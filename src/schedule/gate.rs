//! A job's gate: the probe it runs, the predicate applied to the result, and why a fire was
//! withheld.

use super::*;

/// How a gate obtains the value it judges.
///
/// Split from [`GatePredicate`] because the two answer different questions and only one of them is
/// shell-shaped. Welded together, an exit-code predicate is meaningless for anything but a command,
/// and a tool result containing a timestamp can only ever be described as "changed".
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GateProbe {
    /// A shell command, run unsandboxed. Requires `unrestricted`; see
    /// [`crate::permission::Permission::allows_unattended_shell`].
    Shell { command: String },
    /// A tool call, by the same name the model would use (`mcp__server__tool`, or a built-in).
    ///
    /// Deliberately *not* held to `unrestricted`: a structured call to a server the operator
    /// configured is not a bare `sh -c` with meka's environment, so the bar is the tool's own
    /// level. See [`gate_probe_is_authorized`].
    Tool {
        name: String,
        #[serde(default)]
        arguments: serde_json::Value,
    },
}
impl GateProbe {
    /// Discriminant stored in `scheduled_jobs.gate_kind`, mirroring [`Schedule::kind_str`] so a row
    /// can be read for its shape without parsing the spec.
    pub(crate) fn kind_str(&self) -> &'static str {
        match self {
            Self::Shell { .. } => "shell",
            Self::Tool { .. } => "tool",
        }
    }

    /// How the probe reads in a listing, short enough for a one-line summary.
    ///
    /// A tool's arguments are deliberately absent. They can be long and can carry a token the
    /// caller pasted into a gate, and this feeds a `Check` column and an HTTP field. Where the
    /// arguments matter, use [`Self::detail`].
    pub(crate) fn summary(&self) -> String {
        match self {
            Self::Shell { command } => command.clone(),
            Self::Tool { name, .. } => name.clone(),
        }
    }

    /// The probe with its kind named and a tool's arguments attached, for the one reader that needs
    /// them.
    ///
    /// `schedule_list` is that reader: the model wrote those arguments and cannot otherwise read
    /// back what it created, so a gate it built with the wrong `since` looks identical to a correct
    /// one. Every other surface stays on [`Self::summary`], because the operator's listing and the
    /// HTTP view are read by parties who did not author the job.
    ///
    /// The kind is named because the two are otherwise indistinguishable where a tool's name would
    /// also be a valid command: `fetch_url` as a shell gate and `fetch_url` as a tool gate rendered
    /// identically, and they are an unsandboxed `sh -c` and a structured call. The model needs the
    /// difference to recreate the job it is reading back.
    pub(crate) fn detail(&self) -> String {
        match self {
            Self::Shell { command } => format!("shell {command}"),
            Self::Tool { name, arguments } => match arguments {
                // An omitted or empty argument object is the common case and adds nothing.
                serde_json::Value::Null => format!("tool {name}"),
                serde_json::Value::Object(fields) if fields.is_empty() => format!("tool {name}"),
                other => format!("tool {} {}", name, truncate_gate_output(&other.to_string())),
            },
        }
    }
}
/// Which value a [`GatePredicate::At`] test is applied to, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PointerTest {
    /// A non-empty array, object or string, or any non-null scalar.
    NotEmpty,
    /// The inverse, including a pointer that resolves to nothing.
    Empty,
    /// The pointed-at value differs from the previous evaluation's.
    Changed,
}
/// What the probe's result has to look like for the job to fire.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GatePredicate {
    /// The whole output differs from the previous evaluation's. Edge-triggered: "tell me when the
    /// build *finishes*", not "tell me every 30s while it is running".
    Changed,
    /// The probe reported success: a shell command exiting 0, or a tool call that did not come back
    /// as an error. Level-triggered, for "is this true yet".
    Succeeded,
    /// The output matches a regular expression.
    Matches { pattern: String },
    /// A JSON pointer into the result satisfies `is`.
    ///
    /// The reason this exists. A structured result carrying anything self-moving (a `checked_at`, a
    /// request id) is different on every single call, so [`Self::Changed`] over the whole of it
    /// fires every interval and costs exactly the turns a gate is supposed to save. Pointing at the
    /// part that matters is the only honest way to watch one.
    At { pointer: String, is: PointerTest },
}
impl GateProbe {
    /// Parse the `check` half of a gate request.
    ///
    /// Hand-written rather than derived because the request shape and the stored shape answer to
    /// different readers. Storage is meka talking to itself and uses serde's tagging; a request is
    /// authored by a model or typed into a `curl`, so it reads `{"command": ...}` rather than
    /// `{"shell": {"command": ...}}`, and a wrong one has to say what was wrong.
    pub(crate) fn parse_request(
        value: Option<&serde_json::Value>,
    ) -> std::result::Result<Self, String> {
        let Some(object) = value.and_then(|value| value.as_object()) else {
            return Err("`check` must be an object naming either `command` or `tool`".to_string());
        };
        let command = object.get("command").filter(|value| !value.is_null());
        let tool = object.get("tool").filter(|value| !value.is_null());
        match (command, tool) {
            (Some(command), None) => {
                let command = command
                    .as_str()
                    .ok_or_else(|| "`check.command` must be a string".to_string())?;
                if command.trim().is_empty() {
                    return Err("`check.command` cannot be empty".to_string());
                }
                Ok(Self::Shell {
                    command: command.to_string(),
                })
            }
            (None, Some(tool)) => {
                let name = tool
                    .as_str()
                    .ok_or_else(|| "`check.tool` must be a tool name".to_string())?;
                if name.trim().is_empty() {
                    return Err("`check.tool` cannot be empty".to_string());
                }
                // Checked against the shape the tool schema declares. Anything else reaches the
                // tool as null arguments, so the gate errors on every interval instead of being
                // refused once, here, by the door that could have said which field was wrong.
                let arguments = match object.get("arguments") {
                    None | Some(serde_json::Value::Null) => serde_json::json!({}),
                    Some(value) if value.is_object() => value.clone(),
                    Some(_) => {
                        return Err("`check.arguments` must be an object".to_string());
                    }
                };
                Ok(Self::Tool {
                    name: name.to_string(),
                    arguments,
                })
            }
            // Naming both is refused rather than resolved by precedence: the two run entirely
            // different things, and guessing which was meant is how a gate ends up watching
            // something nobody asked it to watch.
            (Some(_), Some(_)) => {
                Err("`check` names both `command` and `tool`; use one".to_string())
            }
            (None, None) => Err("`check` must name either `command` or `tool`".to_string()),
        }
    }
}
impl GatePredicate {
    /// Parse the `when` half of a gate request. Absent means [`Self::Changed`], which is the
    /// predicate most watchers want.
    pub(crate) fn parse_request(
        value: Option<&serde_json::Value>,
    ) -> std::result::Result<Self, String> {
        const EXPECTED: &str = "expected \"changed\", \"succeeded\", {\"matches\": \"<regex>\"} or \
                                {\"at\": \"<json pointer>\", \"is\": \"not-empty\"|\"empty\"|\"changed\"}";

        let Some(value) = value.filter(|value| !value.is_null()) else {
            return Ok(Self::Changed);
        };
        if let Some(word) = value.as_str() {
            return match word {
                "changed" => Ok(Self::Changed),
                "succeeded" => Ok(Self::Succeeded),
                other => Err(format!(
                    "{}; {EXPECTED}",
                    crate::text::unknown_name("gate condition", other, ["changed", "succeeded"])
                )),
            };
        }
        let Some(object) = value.as_object() else {
            return Err(format!("`when` is not a condition; {EXPECTED}"));
        };
        // Refused rather than resolved, exactly as `check` refuses naming both `command` and
        // `tool`. Taking `matches` and ignoring `at` gave the model a gate watching something it
        // did not ask for, silently, at both creation doors -- and the two halves of a `when` that
        // names both are usually meant as *different* conditions, so neither reading is safe.
        if object.contains_key("matches") && object.contains_key("at") {
            return Err(format!(
                "`when` names both `matches` and `at`; give exactly one. {EXPECTED}"
            ));
        }
        if let Some(pattern) = object.get("matches") {
            let pattern = pattern
                .as_str()
                .ok_or_else(|| "`when.matches` must be a regular expression".to_string())?;
            // Compiled here so a bad pattern is refused by the door that accepted it, rather than
            // becoming a gate that silently never fires.
            regex::Regex::new(pattern)
                .map_err(|error| format!("`when.matches` does not compile: {error}"))?;
            return Ok(Self::Matches {
                pattern: pattern.to_string(),
            });
        }
        if let Some(pointer) = object.get("at") {
            let pointer = pointer
                .as_str()
                .ok_or_else(|| "`when.at` must be a JSON pointer such as \"/chats\"".to_string())?;
            if !pointer.is_empty() && !pointer.starts_with('/') {
                return Err(format!(
                    "`when.at` must be a JSON pointer starting with '/', got '{pointer}'"
                ));
            }
            let is = match object.get("is").and_then(|value| value.as_str()) {
                Some("not-empty") | None => PointerTest::NotEmpty,
                Some("empty") => PointerTest::Empty,
                Some("changed") => PointerTest::Changed,
                Some(other) => {
                    return Err(format!(
                        "unknown `when.is` '{other}'; expected 'not-empty', 'empty' or 'changed'"
                    ));
                }
            };
            return Ok(Self::At {
                pointer: pointer.to_string(),
                is,
            });
        }
        Err(format!("`when` is not a condition; {EXPECTED}"))
    }

    /// How the predicate reads in a listing.
    pub(crate) fn summary(&self) -> String {
        match self {
            Self::Changed => "changed".to_string(),
            Self::Succeeded => "succeeded".to_string(),
            Self::Matches { pattern } => format!("matches /{pattern}/"),
            Self::At { pointer, is } => format!("{} {}", pointer, match is {
                PointerTest::NotEmpty => "not-empty",
                PointerTest::Empty => "empty",
                PointerTest::Changed => "changed",
            }),
        }
    }
}
/// The cheap check that decides whether a due job spends a model turn.
///
/// This is the whole reason a 30-second cadence is affordable: without it, watching something costs
/// one model turn per interval whether or not anything happened.
#[derive(Debug, Clone)]
pub(crate) struct Gate {
    pub(crate) probe: GateProbe,
    pub(crate) predicate: GatePredicate,
    /// The comparison baseline from the last evaluation, for the predicates that need one. `None`
    /// until the first run, at which point the job fires: with nothing to compare against,
    /// "changed" is the honest answer, and it also proves the gate works rather than leaving
    /// it silently untested.
    ///
    /// Not always the same bytes the turn saw. [`GatePredicate::At`] with [`PointerTest::Changed`]
    /// stores the *pointed-at* value, because storing the whole result would re-admit the moving
    /// field the pointer was chosen to exclude.
    pub(crate) last_output: Option<String>,
    /// The permission level the creating session held when this gate was authorized.
    ///
    /// Creation checks the level, but creation is a moment and the row outlives it: the session
    /// drops to `read`, or `meka serve --permission read` restarts and inherits the job, and
    /// without this field nothing downstream can tell that the authority behind the gate is
    /// gone. Carrying the level on the row is what lets [`crate::scheduler::prepare`] re-check it
    /// at fire time instead of trusting a decision made days ago.
    pub(crate) permission: crate::permission::Permission,
}
/// The parts of a gate that round-trip through `scheduled_jobs.gate_spec_json` as one JSON value.
///
/// `last_output` and `permission` stay in their own columns: the first is rewritten on every
/// evaluation and the second is read by the fire-time authority check, and neither wants a
/// parse-and-reserialize to touch it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct GateSpec {
    #[serde(flatten)]
    pub(crate) probe: GateProbe,
    pub(crate) when: GatePredicate,
}
impl Gate {
    /// The `gate_spec_json` column: probe and predicate as one JSON value.
    pub(crate) fn spec(&self) -> String {
        // Both halves are meka's own types, so the only way this fails is a serde bug. An empty
        // spec would decode as a corrupt row and refuse the gate, which is the safe direction.
        serde_json::to_string(&GateSpec {
            probe: self.probe.clone(),
            when: self.predicate.clone(),
        })
        .unwrap_or_default()
    }

    /// Rebuild a gate from its columns.
    ///
    /// `kind` is validated against the spec rather than trusted: the two are written together, so
    /// disagreement means a hand-edited or damaged row, and a gate whose stored shape cannot be
    /// read must not resolve to some other shape that happens to parse.
    pub(crate) fn from_stored(
        kind: &str,
        spec: &str,
        last_output: Option<String>,
        permission: crate::permission::Permission,
    ) -> std::result::Result<Self, String> {
        let parsed: GateSpec =
            serde_json::from_str(spec).map_err(|error| format!("unreadable gate spec: {error}"))?;
        if parsed.probe.kind_str() != kind {
            return Err(format!(
                "gate_kind '{}' does not match its spec, which describes a '{}' gate",
                kind,
                parsed.probe.kind_str()
            ));
        }
        Ok(Self {
            probe: parsed.probe,
            predicate: parsed.when,
            last_output,
            permission,
        })
    }
}
/// Ceiling on gate stdout carried into a turn. A gate is meant to yield a status line, not a
/// payload; anything past this is truncated so a runaway command cannot push the prompt over the
/// context window.
pub(crate) const GATE_OUTPUT_LIMIT: usize = 8 * crate::text::KIB;
/// Ceiling on a probe result meka will parse as JSON, which is a different question from how much
/// of it the turn is shown.
///
/// Separate from [`GATE_OUTPUT_LIMIT`] because the two bound different costs. That one bounds the
/// prompt; this one bounds the work an evaluation does, which is the larger number: a `serde_json`
/// `Value` runs several times the size of its input, and a pointer predicate re-serializes the
/// part it judges. A megabyte is far past any status a gate should be reading and far short of
/// what a runaway command can emit.
pub(crate) const GATE_PARSE_LIMIT: usize = crate::text::MIB;
/// What a gate evaluation decided.
#[derive(Debug, Clone)]
pub(crate) struct GateOutcome {
    /// Whether to spend a model turn.
    pub(crate) fired: bool,
    /// The probe's result, trimmed and truncated. Handed to the turn as context when `fired`.
    pub(crate) output: String,
    /// What to persist as the next evaluation's comparison baseline.
    ///
    /// Usually the same as `output`, and separate from it for one predicate: [`GatePredicate::At`]
    /// with [`PointerTest::Changed`] compares the pointed-at value, so storing the whole result
    /// would re-admit the moving field the pointer exists to exclude and the gate would fire every
    /// interval.
    pub(crate) baseline: String,
}
/// What a probe produced, before any predicate is applied to it.
#[derive(Debug, Clone)]
pub(crate) struct ProbeOutcome {
    /// The result as text, trimmed and capped at [`GATE_OUTPUT_LIMIT`]. What the turn is shown.
    pub(crate) text: String,
    /// The machine-readable result, when there is one: a tool's `structuredContent`, or whatever
    /// the untruncated text parsed as.
    pub(crate) structured: Option<serde_json::Value>,
    /// Whether the probe itself reported success: exit 0, or a tool call that was not an error.
    pub(crate) succeeded: bool,
}
impl ProbeOutcome {
    /// Assemble a result, parsing before truncating.
    ///
    /// The order is the point. `text` is capped at [`GATE_OUTPUT_LIMIT`] and gains a truncation
    /// marker, and [`crate::scheduler::pointed_at`] falls back to parsing that text whenever there
    /// is no structured value -- which is the path every shell probe takes, and every MCP
    /// server that returns its JSON as text content, which is most of them. A document over the
    /// cap therefore never parsed again, so an `at` gate over it failed permanently with "the
    /// probe did not return JSON". It did; meka truncated it.
    ///
    /// Parsing `raw` and keeping the result means the cap goes on being what it is for -- bounding
    /// what a runaway probe can push into the turn's context -- without deciding what the gate is
    /// allowed to judge.
    pub(crate) fn new(raw: &str, structured: Option<serde_json::Value>, succeeded: bool) -> Self {
        // Applied to a value the caller already parsed, not only to the fallback below. An MCP
        // server's `structuredContent` arrives as a `Value` and took the `or_else` branch's cap
        // with it -- which is to say the cap covered shell probes and text-only servers, and
        // missed the path the feature was built for.
        //
        // Serializing to measure looks circular and is not: it happens once, here, against a
        // predicate that would otherwise re-serialize the same value on every evaluation
        // (`canonical_json(...).to_string()` in the `At` arm). What this cannot do is un-receive
        // the value: the MCP layer parsed it before meka saw it, so the peak allocation has
        // already been paid. The bound is on what meka keeps and keeps re-doing.
        let structured = structured.filter(|value| {
            serde_json::to_string(value).is_ok_and(|rendered| rendered.len() <= GATE_PARSE_LIMIT)
        });
        let structured = structured.or_else(|| {
            // Bounded separately from the display cap, because relaxing that cap quietly removed
            // the only bound on this. `text` is capped so a runaway probe cannot push the prompt
            // over the context window; parsing what the cap had already trimmed *also* meant every
            // allocation downstream was bounded by 8 KiB. Parsing `raw` instead is what makes a
            // large result readable, and it hands a probe that returns hundreds of megabytes a
            // `Value` several times that size -- built, and for a pointer predicate re-serialized
            // whole, on the scheduler's own task, on every evaluation.
            //
            // A megabyte covers any result a gate has business judging while keeping the cost of
            // a hostile or runaway one flat. Past it there is no structured value, so a pointer
            // predicate declines and says the probe did not return JSON, which is the same answer
            // it gives for a result it genuinely cannot read.
            (raw.len() <= GATE_PARSE_LIMIT)
                .then(|| serde_json::from_str::<serde_json::Value>(raw.trim()).ok())
                .flatten()
        });
        Self {
            text: truncate_gate_output(raw),
            structured,
            succeeded,
        }
    }
}
/// Why a gate may not run at a given level.
///
/// One type so the doors that ask -- `schedule_create`, `POST /v1/sessions/{id}/schedule`, and the
/// fire-time re-check in [`crate::scheduler::prepare`] -- give the same answer for the same state.
/// Phrased separately at each door, one of them ends up naming a level that does not exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GateRefusal {
    /// A shell gate is a bare `sh -c` on a timer with nobody watching.
    ShellNeedsUnrestricted,
    /// The tool is not registered, or its server is not connected, right now.
    ToolUnavailable,
    /// The tool resolves above `read`. A gate asks a question; a tool that can act is not one.
    ToolNotReadOnly(crate::permission::Permission),
    /// The tool is read-only, but the session is not even at `read`.
    SessionBelowTool,
}
impl GateRefusal {
    /// The user- and model-facing reason, naming the level actually held.
    pub(crate) fn explain(
        &self,
        probe: &GateProbe,
        level: crate::permission::Permission,
    ) -> String {
        match self {
            Self::ShellNeedsUnrestricted => format!(
                "a gate command runs unattended with no sandbox, so it needs `unrestricted` \
                 (currently {level})"
            ),
            // Deliberately not "right now". That reads as transient, and the common cause is not:
            // a name that does not exist, or a session-scoped tool a gate could never reach, is
            // permanent, and a model told "right now" will keep the job and wait. The
            // genuinely-transient case is a server still connecting, which the reporting surfaces
            // decline to mention at all until it settles.
            Self::ToolUnavailable => format!(
                "no gate tool named '{}'. A gate can call a read-only tool that does not depend on \
                 the session (an MCP tool, or one of `read_file`, `find_files`, \
                 `search_contents`, `fetch_url`, `search_web`, and `execute_command` where a \
                 sandbox is available), or the server providing it is not connected",
                probe.summary()
            ),
            Self::ToolNotReadOnly(required) => format!(
                "gate tool '{}' requires '{}'; a gate may only call a tool that needs `read` or less",
                probe.summary(),
                required
            ),
            Self::SessionBelowTool => format!(
                "gate tool '{}' needs `read` (currently '{}')",
                probe.summary(),
                level
            ),
        }
    }
}
/// Whether `level` may author or fire this probe, re-resolving the tool every time it is asked.
///
/// Both halves are checked, and both are checked *now* rather than trusted from creation. A job
/// authored at `unrestricted` must stop firing its command once the session drops, or a daemon runs
/// an unsandboxed command after the user lowered the level and is entitled to be surprised. And a
/// tool that resolved to `read` when the job was written but resolves higher today must stop being
/// a gate, because the operator retuned `tool_permissions` and meant it.
pub(crate) fn gate_probe_is_authorized(
    probe: &GateProbe,
    level: crate::permission::Permission,
    tools: Option<&dyn GateTools>,
) -> std::result::Result<(), GateRefusal> {
    match probe {
        GateProbe::Shell { .. } => {
            if level.allows_unattended_shell() {
                Ok(())
            } else {
                Err(GateRefusal::ShellNeedsUnrestricted)
            }
        }
        GateProbe::Tool { name, .. } => {
            // No dispatcher means this process cannot resolve the name, which is the same answer as
            // a disconnected server: not right now.
            let Some(required) = tools.and_then(|tools| tools.resolve(name)) else {
                return Err(GateRefusal::ToolUnavailable);
            };
            // At most `read`, not exactly `read`: a tool an operator pinned to `none` asks even
            // less of the session, and refusing it read as a complaint that it was too dangerous.
            if !crate::permission::Permission::Read.allows(required) {
                return Err(GateRefusal::ToolNotReadOnly(required));
            }
            if !level.allows(crate::permission::Permission::Read) {
                return Err(GateRefusal::SessionBelowTool);
            }
            Ok(())
        }
    }
}
/// Why `gate` will not fire right now, and the level that answer was reached at.
///
/// One function for three readers: the fire door in [`crate::scheduler::prepare`], the
/// `[Scheduled]` index the model sees every turn, and `schedule_list`. Before this the fire door
/// was the only one that asked, so a held-back job was reported to the operator's log and to nobody
/// else: the model saw a job that looked healthy, could not tell a gate that had said "no" from one
/// that was never consulted, and had nothing to act on. It can cancel a job it cannot fire, so the
/// asymmetry was worth closing.
///
/// The live level is tried first because it is the one that can be put back. A refusal that only
/// the *recorded* level produces means a row nothing can currently restore, which is a different
/// thing to say.
pub(crate) fn gate_withheld_reason(
    gate: &Gate,
    live: crate::permission::Permission,
    tools: Option<&dyn GateTools>,
) -> Option<(GateRefusal, crate::permission::Permission)> {
    if let Err(refusal) = gate_probe_is_authorized(&gate.probe, live, tools) {
        return Some((refusal, live));
    }
    if let Err(refusal) = gate_probe_is_authorized(&gate.probe, gate.permission, tools) {
        return Some((refusal, gate.permission));
    }
    None
}
/// Where a gate's [`GateProbe::Tool`] call is dispatched.
///
/// A trait rather than a concrete handle so `schedule` does not take a dependency on the tool and
/// MCP stacks, which would be circular. `src/tools.rs` supplies the implementation.
#[async_trait::async_trait]
pub(crate) trait GateTools: Send + Sync + std::fmt::Debug {
    /// Look up a tool by the name the model would use, and report the permission it currently
    /// resolves to.
    ///
    /// `None` when the name is unknown *or* its server is not connected. Both are the same answer
    /// for a gate: it cannot be evaluated right now, so it has not passed.
    fn resolve(&self, name: &str) -> Option<crate::permission::Permission>;

    /// Whether this name might still resolve once its server finishes connecting.
    ///
    /// Only the *reporting* surfaces ask. Authority does not: a gate whose server is mid-handshake
    /// genuinely cannot run, and [`Self::resolve`] returning `None` is the right answer there. But
    /// saying "not available right now" in the model's `[Scheduled]` block during startup marks a
    /// healthy job as dead and then announces it alive again a turn later, which is worse than
    /// saying nothing for the second it takes.
    ///
    /// Defaulted to `false` so a dispatcher with no notion of connecting -- every test stub, and
    /// any future non-MCP one -- keeps the plain behavior.
    fn is_still_connecting(&self, _name: &str) -> bool {
        false
    }

    /// Call it, in the creating session's directory. Only reached once [`Self::resolve`] has
    /// answered and the authority check has passed.
    ///
    /// `cwd` is here and not on `resolve` because only the call needs it: what a tool *requires* is
    /// a property of the tool, while where it runs is a property of the job.
    async fn call(
        &self,
        name: &str,
        arguments: &serde_json::Value,
        timeout: Duration,
        cwd: Option<&std::path::Path>,
        session_id: Option<uuid::Uuid>,
    ) -> Result<ProbeOutcome, String>;
}
/// One line of a probe's result, short enough to sit inside a warning.
pub(crate) fn elide_for_message(text: &str) -> String {
    const LIMIT: usize = 120;
    let first = text.trim().lines().next().unwrap_or_default();
    match first.char_indices().nth(LIMIT) {
        Some((cut, _)) => format!("{}…", &first[..cut]),
        None => first.to_string(),
    }
}
/// Trim and cap a probe's result. Trimming matters for correctness, not tidiness: most commands
/// emit a trailing newline, and comparing untrimmed output would be fine, but a command whose
/// trailing whitespace varies run to run would fire a `changed` gate forever.
pub(crate) fn truncate_gate_output(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.len() <= GATE_OUTPUT_LIMIT {
        return trimmed.to_string();
    }
    // Cut on a character boundary so the result is still valid UTF-8.
    let mut end = GATE_OUTPUT_LIMIT;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[gate output truncated]", &trimmed[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        schedule::tests::{FixedTools, GATE_BUDGET, StillConnecting, gate, probed, tool_probe},
        scheduler::*,
    };

    /// A pointer into something that is not a JSON document is an error, not an answer.
    ///
    /// The process ran fine, so this is not a spawn failure -- but the predicate describes a shape
    /// the result does not have, so nothing was measured. Declining silently is survivable for
    /// `not-empty` and ruinous for `empty`: a missing value reads as empty, so a server that starts
    /// returning prose or an error string would fire the job every interval, indefinitely. Both
    /// directions are covered here because the asymmetry is the point.
    #[test]
    fn a_pointer_into_something_that_is_not_json_is_an_error() {
        for is in [
            PointerTest::NotEmpty,
            PointerTest::Empty,
            PointerTest::Changed,
        ] {
            let predicate = GatePredicate::At {
                pointer: "/chats".to_string(),
                is,
            };
            let error = apply_predicate(&predicate, &probed("upstream is down", None, true), None)
                .expect_err("prose is not a document to point into");
            assert!(error.contains("/chats"), "{error}");
            assert!(
                error.contains("upstream is down"),
                "the message has to carry what it actually got: {error}"
            );
        }
    }

    /// Every shape survives the round trip through the two columns, including the arguments a tool
    /// probe carries. A gate that stored but did not reload would fire on whatever the decode
    /// happened to produce.
    #[test]
    fn every_gate_shape_round_trips_through_its_columns() {
        let shapes = [
            (
                GateProbe::Shell {
                    command: "gh pr checks".to_string(),
                },
                GatePredicate::Changed,
            ),
            (
                GateProbe::Shell {
                    command: "curl -f https://example.test".to_string(),
                },
                GatePredicate::Succeeded,
            ),
            (
                GateProbe::Tool {
                    name: "mcp__bridge__unseen".to_string(),
                    arguments: serde_json::json!({"folder": "inbox"}),
                },
                GatePredicate::At {
                    pointer: "/chats".to_string(),
                    is: PointerTest::NotEmpty,
                },
            ),
            (
                GateProbe::Tool {
                    name: "fetch_url".to_string(),
                    arguments: serde_json::json!({"url": "https://example.test/health"}),
                },
                GatePredicate::Matches {
                    pattern: "ok".to_string(),
                },
            ),
        ];

        for (probe, predicate) in shapes {
            let gate = Gate {
                probe: probe.clone(),
                predicate: predicate.clone(),
                last_output: Some("baseline".to_string()),
                permission: crate::permission::Permission::Read,
            };
            let restored = Gate::from_stored(
                probe.kind_str(),
                &gate.spec(),
                gate.last_output.clone(),
                gate.permission,
            )
            .expect("a gate meka wrote must be one meka can read");
            assert_eq!(restored.probe, probe);
            assert_eq!(restored.predicate, predicate);
            assert_eq!(restored.last_output.as_deref(), Some("baseline"));
        }
    }

    /// The three request-parser refusals, each of which would otherwise resolve silently or fail
    /// late.
    #[test]
    fn the_request_parsers_refuse_what_they_cannot_answer() {
        // Naming both halves of a `when` is an ambiguity, not a precedence question. Resolving it
        // to `matches` gave the model a gate watching something it did not ask for.
        let both = serde_json::json!({"matches": "x", "at": "/y", "is": "changed"});
        let error = GatePredicate::parse_request(Some(&both))
            .expect_err("`when` naming both is refused, as `check` naming both is");
        assert!(error.contains("both"), "{error}");

        // `arguments` reaches a tool, so it has to be the shape the tool schema declares.
        let scalar = serde_json::json!({"tool": "t", "arguments": "oops"});
        let error = GateProbe::parse_request(Some(&scalar))
            .expect_err("a string is not an argument object");
        assert!(error.contains("`check.arguments`"), "{error}");

        // The shapes that are fine stay fine, so the guards above are not just refusing everything.
        assert!(GatePredicate::parse_request(Some(&serde_json::json!({"matches": "x"}))).is_ok());
        assert!(
            GateProbe::parse_request(Some(&serde_json::json!({"tool": "t"})))
                .is_ok_and(|probe| matches!(probe, GateProbe::Tool { .. }))
        );
    }

    /// A job carrying `gate` and nothing else of interest, for the predicate tests that never touch
    /// a store. A fresh id every call, so the process-global ledgers keyed by job id cannot carry
    /// one test's state into another's.
    fn job_carrying(gate: Option<Gate>) -> ScheduledJob {
        ScheduledJob {
            attempts: 0,
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::nil(),
            schedule: Schedule::parse_every("1h").expect("parses"),
            prompt: "watch the thing".to_string(),
            gate,
            created_at: Utc::now(),
            last_fired_at: None,
            next_fire_at: Utc::now(),
        }
    }

    /// A gate whose server has not finished connecting is not reported as dead.
    ///
    /// Between process start and `Connected`, and again on every reconnect, the tool is absent from
    /// the snapshot. Marking that in the model's `[Scheduled]` block says a healthy job is dead and
    /// then announces it alive a turn later -- churn the model may act on. Authority is unchanged:
    /// the fire door still declines, because the probe genuinely cannot run.
    #[test]
    fn a_gate_whose_server_is_still_connecting_is_not_reported_as_dead() {
        let gate = Gate {
            probe: tool_probe(),
            predicate: GatePredicate::Succeeded,
            last_output: None,
            permission: crate::permission::Permission::Read,
        };
        let level = crate::permission::Permission::Read;

        let job = job_carrying(Some(gate.clone()));

        assert_eq!(
            job_withheld_reason(
                &SchedulerMemory::default(),
                &job,
                level,
                Some(&StillConnecting)
            ),
            None,
            "a handshake in progress is not a verdict"
        );
        assert!(
            job_withheld_reason(
                &SchedulerMemory::default(),
                &job,
                level,
                Some(&FixedTools(None))
            )
            .is_some(),
            "but a server that is simply not there still is"
        );
        assert!(
            gate_probe_is_authorized(&gate.probe, level, Some(&StillConnecting)).is_err(),
            "and the authority check refuses either way, since the probe cannot run"
        );
        assert_eq!(
            job_withheld(&SchedulerMemory::default(), &job, Some(level), None),
            Withheld::Undetermined,
            "a reader with no dispatcher has not established that the job is fine; it has \
             established nothing, and `meka schedule list` renders that as `?` rather than as the \
             blank cell that means healthy"
        );
    }

    /// The point of the whole permission split. A read-only tool call is not a bare `sh -c`, so
    /// holding it to `unrestricted` would leave gating unavailable to everyone below it -- which,
    /// with `workspace` now the default rung, is most people.
    #[test]
    fn a_read_only_tool_gate_is_allowed_at_read() {
        let tools = FixedTools(Some(crate::permission::Permission::Read));
        assert!(
            gate_probe_is_authorized(
                &tool_probe(),
                crate::permission::Permission::Read,
                Some(&tools)
            )
            .is_ok()
        );
    }

    /// The user's second scenario: a tool that resolved to `read` when the job was written but
    /// resolves higher today. Re-resolving at fire time is what catches it; trusting the level
    /// recorded at creation would keep calling it.
    #[test]
    fn a_tool_that_now_resolves_above_read_stops_being_a_gate() {
        let tools = FixedTools(Some(crate::permission::Permission::Unrestricted));
        let refusal = gate_probe_is_authorized(
            &tool_probe(),
            crate::permission::Permission::Unrestricted,
            Some(&tools),
        )
        .expect_err("a tool that can act is not a question");
        assert_eq!(
            refusal,
            GateRefusal::ToolNotReadOnly(crate::permission::Permission::Unrestricted)
        );
        // At most `read`, not exactly `read`: a tool pinned to `none` asks even less.
        assert!(
            gate_probe_is_authorized(
                &tool_probe(),
                crate::permission::Permission::Unrestricted,
                Some(&FixedTools(Some(crate::permission::Permission::None))),
            )
            .is_ok(),
            "a tool that needs nothing is a question a gate may ask"
        );
    }

    /// The user's first scenario, for the tool half: the session dropped below what the tool needs.
    #[test]
    fn a_tool_gate_stops_once_the_session_falls_below_read() {
        let tools = FixedTools(Some(crate::permission::Permission::Read));
        let refusal = gate_probe_is_authorized(
            &tool_probe(),
            crate::permission::Permission::None,
            Some(&tools),
        )
        .expect_err("`none` cannot call even a read-only tool");
        assert_eq!(refusal, GateRefusal::SessionBelowTool);
    }

    /// An unknown name and a disconnected server are the same answer, and both decline rather than
    /// fire. A gate that could not be evaluated has not passed.
    #[test]
    fn an_unresolvable_tool_declines_rather_than_fires() {
        for tools in [FixedTools(None), FixedTools(None)] {
            let refusal = gate_probe_is_authorized(
                &tool_probe(),
                crate::permission::Permission::Unrestricted,
                Some(&tools),
            )
            .expect_err("nothing to resolve against");
            assert_eq!(refusal, GateRefusal::ToolUnavailable);
        }
        // And a process with no dispatcher at all reads the same way.
        let refusal = gate_probe_is_authorized(
            &tool_probe(),
            crate::permission::Permission::Unrestricted,
            None,
        )
        .expect_err("this process cannot dispatch tools");
        assert_eq!(refusal, GateRefusal::ToolUnavailable);
    }

    /// The shell bar is unchanged, and unchanged for its own reason: a bare `sh -c` on a timer with
    /// meka's environment is not made safer by the probe split.
    #[test]
    fn a_shell_gate_still_needs_unrestricted() {
        let probe = GateProbe::Shell {
            command: "true".to_string(),
        };
        for level in [
            crate::permission::Permission::None,
            crate::permission::Permission::Read,
            crate::permission::Permission::Workspace,
        ] {
            assert_eq!(
                gate_probe_is_authorized(&probe, level, None),
                Err(GateRefusal::ShellNeedsUnrestricted),
                "a shell gate must not run at {level}"
            );
        }
        assert!(
            gate_probe_is_authorized(&probe, crate::permission::Permission::Unrestricted, None)
                .is_ok()
        );
    }

    /// `gate_kind` and `gate_spec_json` are written together, so disagreement means a hand-edited
    /// or damaged row. Resolving it to whichever half happens to parse would run a gate the
    /// operator did not write.
    #[test]
    fn a_gate_kind_that_contradicts_its_spec_is_refused() {
        let spec = r#"{"shell":{"command":"true"},"when":"changed"}"#;
        let error = Gate::from_stored("tool", spec, None, crate::permission::Permission::Read)
            .expect_err("the two columns disagree");
        assert!(error.contains("does not match its spec"), "{error}");
    }

    /// The model almost always authors a gate right after verifying the same command through
    /// `execute_command`, which runs in the session cwd. Under a `meka serve` unit the host process
    /// sits somewhere else entirely (`/`, or wherever systemd put it), so a gate that ignores the
    /// session cwd silently stops matching the command the user watched succeed. Nothing caught
    /// this: the `cwd` argument threads all the way through `prepare` with no assertion on it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_runs_in_its_sessions_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        // Resolved because macOS hands out `/var/...` symlinked to `/private/var/...`, and `pwd`
        // in the child reports the resolved form.
        let directory = temp.path().canonicalize().expect("canonicalize");

        let outcome = evaluate_gate(
            &gate("pwd", GatePredicate::Changed, None),
            GATE_BUDGET,
            Some(&directory),
            None,
            None,
        )
        .await
        .expect("gate ran");

        assert_eq!(
            outcome.output,
            directory.to_string_lossy(),
            "the gate ran in the host's directory instead of the session's"
        );
    }

    #[tokio::test]
    async fn gate_output_is_trimmed_so_trailing_newlines_do_not_flap() {
        // `echo` appends a newline. Comparing untrimmed, a gate whose command varied its trailing
        // whitespace would fire forever.
        let outcome = evaluate_gate(
            &gate("echo spaced", GatePredicate::Changed, None),
            GATE_BUDGET,
            None,
            None,
            None,
        )
        .await
        .expect("gate ran");
        assert_eq!(outcome.output, "spaced");
    }

    #[test]
    fn truncate_gate_output_caps_and_marks() {
        let short = truncate_gate_output("  brief  ");
        assert_eq!(short, "brief");

        let long = truncate_gate_output(&"x".repeat(GATE_OUTPUT_LIMIT * 2));
        assert!(long.len() < GATE_OUTPUT_LIMIT * 2);
        assert!(long.ends_with("[gate output truncated]"));
    }

    /// Truncation cuts by byte offset, so a multi-byte character straddling the limit would panic a
    /// naive slice.
    #[test]
    fn truncate_gate_output_cuts_on_a_character_boundary() {
        // Three bytes wide, and the limit is not a multiple of three, so the cut lands
        // mid-character and the walk-back actually runs. A two-byte character would divide
        // the even limit exactly and never exercise it.
        assert_ne!(GATE_OUTPUT_LIMIT % 3, 0, "fixture relies on a ragged cut");
        let multibyte = "☃".repeat(GATE_OUTPUT_LIMIT);
        let truncated = truncate_gate_output(&multibyte);
        assert!(truncated.ends_with("[gate output truncated]"));
    }
}
