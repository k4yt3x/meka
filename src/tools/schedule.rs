//! The `schedule_*` tools: the agent arranging its own future turns ([`crate::schedule`]).
//!
//! Creating and listing gate at [`Permission::Read`], matching `memory_*` and `scratchpad_*`: a
//! scheduled prompt writes to a store Meka owns, and the turn it eventually produces is permission
//! -checked when it runs, so scheduling one grants nothing the session did not already have.
//!
//! The `gate` field is the exception, and is checked inside [`ScheduleCreateTool::execute`]
//! against [`crate::schedule::gate_probe_is_authorized`]. A gate runs unattended, on a timer, until
//! someone cancels it, which a tool call inside a turn is not. The bar depends on what the gate
//! runs: a shell command needs
//! `unrestricted` because it is unsandboxed, while a read-only tool call needs only what that tool
//! needs. [`Tool::required_permission`] is per-tool and cannot vary by argument, so the check has
//! to live in the body either way.
//!
//! Sub-agents deliberately get none of these tools. A sub-agent's session is ephemeral, so a
//! schedule keyed to it would outlive the only conversation that could run it.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use super::{
    Tool, ToolOutput,
    util::{optional_object, optional_str, require_str, resolve_session_id},
};
use crate::{
    config::ResolvedScheduleConfig,
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
    schedule::{Gate, GatePredicate, GateProbe, Schedule, Timing},
    store::Store,
};

/// Shared by all three tools.
pub(super) struct ScheduleContext {
    store: Store,
    site: crate::session::ToolSite,
}

/// Absolute local time plus a relative offset, e.g. `Wed 2026-08-12 08:57 +02:00 (in 17h 35m)`.
///
/// Both halves earn their place: the absolute form is what the user can check against a calendar,
/// and the relative form is what catches a timing that parsed successfully but means something
/// other than intended. A model that writes `0 9 * * 1-5` believing it fires "in a few minutes"
/// sees `in 17h 35m` and can correct itself before the user ever finds out.
fn describe_fire_time(at: chrono::DateTime<Utc>) -> String {
    let delta = at - Utc::now();
    let relative = match delta.to_std() {
        Ok(std) => format!(
            "in {}",
            humantime_serde::re::humantime::format_duration(std::time::Duration::from_secs(
                std.as_secs()
            ))
        ),
        // Negative: the instant has already passed, which the scheduler will treat as due.
        Err(_) => "overdue".to_string(),
    };
    format!(
        "{} ({})",
        crate::text::format_timestamp(at, crate::text::Precision::WeekdayMinutes),
        relative
    )
}

pub(super) struct ScheduleCreateTool {
    /// The dispatcher a tool gate resolves against, for the `[Scheduled]` reporting; `None` where
    /// the host has none.
    pub(crate) gate_tools: Option<Arc<dyn crate::schedule::GateTools>>,
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    pub(crate) config: ResolvedScheduleConfig,
}

#[async_trait]
impl Tool for ScheduleCreateTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "schedule_create".to_string(),
            description: "Schedule a self-contained prompt for this session. Give exactly one of `at`, `every`, or `cron`. A fire arrives as a message in this conversation: a running turn reads it at its next pause, and an idle session gets a turn for it. A running scheduler must serve the session for the schedule to fire; a saved schedule alone does not keep Meka running. Write the prompt as an unattended instruction, including the context it needs.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What to do when the schedule fires. Self-contained: the \
                                        conversation that created it may be long gone."
                    },
                    "at": {
                        "type": "string",
                        "description": "One-shot. Either a duration from now ('20m', '2h', \
                                        '1h 30m') or an RFC 3339 timestamp. Note 'm' is minutes \
                                        and 'M' is months. Fires once, then deletes itself."
                    },
                    "every": {
                        "type": "string",
                        "description": "Recurring fixed interval ('30m', '1h', '1d'). Runs until \
                                        canceled."
                    },
                    "cron": {
                        "type": "string",
                        "description": "Recurring 5-field cron expression, read in `timezone` \
                                        ('0 9 * * 1-5' = 09:00 on weekdays). No seconds field."
                    },
                    "timezone": {
                        "type": "string",
                        "description": timezone_description()
                    },
                    "gate": {
                        "type": "object",
                        "description": "Optional. Check something first and fire only if it says \
                                        something happened. Turns an expensive poll into a cheap \
                                        one, so a short `every` becomes affordable.",
                        "properties": {
                            "check": {
                                "type": "object",
                                "description": "What to run. Either {\"command\": \"...\"} for a \
                                                shell command, which needs `unrestricted` because \
                                                it runs unsandboxed, or {\"tool\": \"name\", \
                                                \"arguments\": {...}} to call a read-only tool, \
                                                which needs only what that tool needs. Prefer the \
                                                tool form when one exists: it is available at lower \
                                                permission and returns structured data you can \
                                                point `when.at` into.",
                                "properties": {
                                    "command": {"type": "string"},
                                    "tool": {"type": "string"},
                                    "arguments": {"type": "object"}
                                }
                            },
                            "when": {
                                "default": "changed",
                                "description": "What counts as 'something happened'. \"changed\" \
                                                (default) fires when the whole result differs from \
                                                last time. \"succeeded\" fires while the command \
                                                exits 0 or the tool call does not error. \
                                                {\"matches\": \"regex\"} fires when the result \
                                                matches. {\"at\": \"/json/pointer\", \"is\": \
                                                \"not_empty\"|\"empty\"|\"changed\"} judges one \
                                                field. Use `at` for anything returning JSON: a \
                                                result carrying a timestamp or an id differs on \
                                                every call, so \"changed\" over the whole of it \
                                                fires every tick and costs the turns the gate is \
                                                meant to save."
                            }
                        },
                        "required": ["check"]
                    }
                },
                "required": ["prompt"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let tool_name = "schedule_create";
        let session_id = resolve_session_id(&self.site.session_id, tool_name)?;
        let prompt = require_str(&input, "prompt", tool_name)?;

        let now = Utc::now();
        let timing = parse_timing(&input, now, tool_name)?;
        let gate = self.parse_gate(&input, tool_name)?;

        let next_fire_at = timing
            .next_after(now)
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: "that time has no next occurrence. A one-shot time must be in the \
                          future."
                    .to_string(),
            })?;

        let schedule = Schedule {
            id: Uuid::new_v4().to_string(),
            session_id,
            timing,
            prompt: prompt.to_string(),
            gate,
            created_at: now,
            last_fired_at: None,
            next_fire_at,
            attempts: 0,
        };
        // The cap is held where the row is written, in one transaction with the count, so two
        // calls racing for the last slot cannot both be told yes.
        if let crate::store::schedule::ScheduleCreation::AtCapacity { existing } = self
            .store
            .schedule_store()
            .create_schedule(&schedule, self.config.max_schedules)
            .await?
        {
            return Err(MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: format!(
                    "this session already has {existing} schedules (the limit). Cancel one with \
                     schedule_cancel first."
                ),
            });
        }

        let short_id = schedule.short_id();
        let timing = schedule.timing.describe();
        let next_fire = next_fire_at.to_rfc3339();
        tracing::info!("schedule {short_id} ({timing}), next fire {next_fire}");

        let mut summary = format!(
            "Created schedule {} ({}). Next fire: {}.",
            schedule.short_id(),
            schedule.timing.describe(),
            describe_fire_time(next_fire_at)
        );
        if schedule.gate.is_some() {
            summary.push_str(" Gated: it fires only when the gate says so.");
        }
        summary.push_str(&format!(
            " Cancel with schedule_cancel(\"{}\").",
            schedule.short_id()
        ));
        Ok(ToolOutput::text(summary, false))
    }
}

impl ScheduleCreateTool {
    fn parse_gate(&self, input: &serde_json::Value, tool_name: &str) -> Result<Option<Gate>> {
        let Some(raw) = optional_object(input, "gate", tool_name)? else {
            return Ok(None);
        };
        let refuse = |message: String| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message,
        };
        let probe = GateProbe::parse_request(raw.get("check")).map_err(refuse)?;
        let predicate = GatePredicate::parse_request(raw.get("when")).map_err(refuse)?;

        // A gate outlives the turn that created it and runs with no supervision, which is a
        // stronger grant than a tool call inside a turn. Checked here rather than through
        // `required_permission` so an ungated reminder still works at read, and because the bar
        // depends on the probe: a shell command needs `unrestricted`, a read-only tool call needs
        // only what that tool needs.
        let permission = self.site.permission.get();
        if let Err(refusal) = crate::schedule::gate_probe_is_authorized(
            &probe,
            permission,
            self.gate_tools.as_deref(),
        ) {
            return Err(refuse(format!(
                "{}. Create the schedule without `gate` and check the condition inside the prompt \
                 instead.",
                refusal.explain(&probe, permission)
            )));
        }

        Ok(Some(Gate {
            probe,
            predicate,
            last_output: None,
            // Recorded, not re-derived. The check above proves the level *now*; the row will be
            // executed by some other process on some later day, and `prepare` re-checks both this
            // and the live level before running anything.
            permission,
        }))
    }
}

/// The timing the tool input names, read by the one rule every door that creates a schedule uses.
fn parse_timing(
    input: &serde_json::Value,
    now: chrono::DateTime<Utc>,
    tool_name: &str,
) -> Result<Timing> {
    Timing::parse_request(
        optional_str(input, "at", tool_name)?,
        optional_str(input, "every", tool_name)?,
        optional_str(input, "cron", tool_name)?,
        optional_str(input, "timezone", tool_name)?,
        now,
    )
    .map_err(|message| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message,
    })
}

/// What `timezone` says, naming the machine's zone so the model knows what a pattern without one
/// is read in.
fn timezone_description() -> String {
    match crate::schedule::machine_zone_name() {
        Some(name) => {
            format!("IANA time zone a `cron` pattern is read in. Default: the machine's, {name}.")
        }
        None => "IANA time zone a `cron` pattern is read in. Default: the machine's.".to_string(),
    }
}

pub(super) struct ScheduleListTool {
    /// Carried only to resolve a gate's tool when reporting whether the gate can still fire;
    /// `None` where the host has no dispatcher.
    pub(crate) gate_tools: Option<Arc<dyn crate::schedule::GateTools>>,
    pub(crate) context: ScheduleContext,
}

#[async_trait]
impl Tool for ScheduleListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "schedule_list".to_string(),
            description: "List this session's schedules, with their next fire times, \
                gates, and full prompts. Your per-turn context already carries a short index, so \
                reach for this when you need a schedule's exact prompt or gate check."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let session_id = resolve_session_id(&self.context.site.session_id, "schedule_list")?;
        let schedules = self
            .context
            .store
            .schedule_store()
            .list_schedules(session_id)
            .await?;
        if schedules.is_empty() {
            return Ok(ToolOutput::text(
                "No schedules in this session.".to_string(),
                false,
            ));
        }

        let mut rendered = String::new();
        for schedule in &schedules {
            rendered.push_str(&format!(
                "{} ({})\n  next: {}\n",
                schedule.short_id(),
                schedule.timing.describe(),
                describe_fire_time(schedule.next_fire_at)
            ));
            if let Some(fired) = schedule.last_fired_at {
                rendered.push_str(&format!(
                    "  last fired: {}\n",
                    crate::text::format_timestamp(fired, crate::text::Precision::Minutes)
                ));
            }
            if let Some(gate) = &schedule.gate {
                // `detail`, not `summary`: this is the surface the model authored the schedule on,
                // and it cannot otherwise read back the arguments it wrote.
                rendered.push_str(&format!(
                    "  gate ({}): {}\n",
                    gate.predicate.summary(),
                    gate.probe.detail()
                ));
            }
            // A held schedule and a healthy one look identical without this line: both simply never
            // fire, and `last fired` is absent for a brand-new schedule too. Outside the `gate`
            // branch because an ungated schedule on a session at `none` is held as well. Reported
            // to the model because it can act on it: `schedule_cancel` needs only `read`.
            if let Some(reason) = crate::schedule::schedule_withheld_reason(
                self.context.store.scheduler_memory(),
                schedule,
                self.context.site.permission.get(),
                self.gate_tools.as_deref(),
            ) {
                rendered.push_str(&format!("  NOT FIRING: {reason}\n"));
            }
            rendered.push_str(&format!("  prompt: {}\n", schedule.prompt));
        }
        Ok(ToolOutput::text(rendered, false))
    }
}

pub(super) struct ScheduleCancelTool {
    pub(crate) context: ScheduleContext,
}

#[async_trait]
impl Tool for ScheduleCancelTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "schedule_cancel".to_string(),
            description: "Cancel a schedule so it stops firing. Takes the id from \
                schedule_create or schedule_list; a unique prefix is enough."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Schedule id, or any unique prefix of one."
                    }
                },
                "required": ["id"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let tool_name = "schedule_cancel";
        let session_id = resolve_session_id(&self.context.site.session_id, tool_name)?;
        let id = require_str(&input, "id", tool_name)?;

        match self
            .context
            .store
            .schedule_store()
            .cancel_schedule(session_id, &id)
            .await?
        {
            Some(canceled) => {
                let canceled_id = &canceled.id;
                tracing::info!("canceled schedule {canceled_id}");
                // The fire the schedule had waiting went with it, and whoever follows the session
                // was told an item was queued.
                for item_id in canceled.withdrawn {
                    context
                        .frontend
                        .emit(crate::frontend::FrontendEvent::InboxWithdrawn { item_id })
                        .await;
                }
                Ok(ToolOutput::text(
                    format!(
                        "Canceled schedule {}.",
                        &canceled_id[..8.min(canceled_id.len())]
                    ),
                    false,
                ))
            }
            None => Ok(ToolOutput::text(
                format!(
                    "No schedule matching '{id}' in this session. Use schedule_list to see \
                     what is there."
                ),
                false,
            )),
        }
    }
}

/// Build the three tools. Kept here so `crate::tools` does not need to know their field shapes.
pub(super) fn build(
    store: Store,
    site: crate::session::ToolSite,
    config: ResolvedScheduleConfig,
    gate_tools: Option<Arc<dyn crate::schedule::GateTools>>,
) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(ScheduleCreateTool {
            gate_tools: gate_tools.clone(),
            store: store.clone(),
            config,
            site: site.clone(),
        }),
        Arc::new(ScheduleListTool {
            gate_tools,
            context: ScheduleContext {
                store: store.clone(),
                site: site.clone(),
            },
        }),
        Arc::new(ScheduleCancelTool {
            context: ScheduleContext { store, site },
        }),
    ]
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;

    async fn harness() -> (ScheduleCreateTool, crate::session::SharedSessionId, Store) {
        let manager = Store::for_test().await;
        let session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let session_id = crate::session::SharedSessionId::new(Some(session));
        let tool = ScheduleCreateTool {
            store: manager.clone(),
            config: ResolvedScheduleConfig::default(),
            gate_tools: None,
            site: crate::session::ToolSite::for_test()
                .with_permission(crate::permission::SharedPermission::new(
                    Permission::Unrestricted,
                    crate::permission::EnabledPermissions::ALL,
                ))
                .with_session_id(session_id.clone()),
        };
        (tool, session_id, manager)
    }

    #[tokio::test]
    async fn create_reports_the_resolved_fire_time() {
        let (tool, _session_id, _manager) = harness().await;
        let output = tool
            .execute(
                serde_json::json!({"prompt": "check the deploy", "at": "20m"}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("creates");
        let text = output.text_content();
        assert!(text.contains("Next fire:"), "{text}");
        // The relative half is the part that catches a timing meaning something other than
        // intended, so it must actually be present.
        assert!(text.contains("in 19m"), "{text}");
    }

    #[tokio::test]
    async fn create_requires_exactly_one_timing() {
        let (tool, _session_id, _manager) = harness().await;
        let none = tool
            .execute(
                serde_json::json!({"prompt": "x"}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect_err("no timing is an error");
        assert!(none.to_string().contains("one of"), "{none}");

        let both = tool
            .execute(
                serde_json::json!({"prompt": "x", "at": "20m", "every": "1h"}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect_err("two timings is an error");
        assert!(both.to_string().contains("exactly one"), "{both}");
    }

    /// The permission rule that cannot live in `required_permission`, since that is per-tool.
    #[tokio::test]
    async fn a_gate_is_refused_below_write_but_the_reminder_is_not() {
        // Every level below `unrestricted`, not just `read`. `workspace` is the one that matters:
        // letting it pass this door is a one-call escape, since `schedule_create` with a gate runs
        // arbitrary unconfined commands from inside the confined level within one poll interval.
        for level in [Permission::None, Permission::Read, Permission::Workspace] {
            refuses_a_gate_at(level).await;
        }
    }

    async fn refuses_a_gate_at(level: Permission) {
        let (mut tool, _session_id, _manager) = harness().await;
        tool.site.permission = crate::permission::SharedPermission::new(
            level,
            crate::permission::EnabledPermissions::ALL,
        );

        let refused = tool
            .execute(
                serde_json::json!({
                    "prompt": "x",
                    "every": "1h",
                    "gate": {"check": {"command": "true"}}
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect_err("a gate needs an unattended-write level");
        assert!(
            refused.to_string().contains("`unrestricted`"),
            "the refusal at {level} must name the level a gate needs: {refused}"
        );

        // The same schedule without a gate is fine at read: scheduling a prompt grants nothing,
        // since the turn it produces is permission-checked when it runs.
        tool.execute(
            serde_json::json!({"prompt": "x", "every": "1h"}),
            crate::tools::ToolContext::for_test(CancellationToken::new()),
        )
        .await
        .unwrap_or_else(|error| panic!("an ungated reminder is allowed at {level}: {error}"));
    }

    /// A gate spelled as a JSON string is refused naming `gate`, not read as one with no `check`.
    #[tokio::test]
    async fn a_gate_that_is_not_an_object_is_refused_by_name() {
        let (tool, _session_id, _manager) = harness().await;
        let refused = tool
            .execute(
                serde_json::json!({
                    "prompt": "x",
                    "every": "1h",
                    "gate": "{\"check\": {\"command\": \"true\"}}"
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect_err("a string is not a gate");
        assert!(
            refused.to_string().contains("'gate' must be an object"),
            "{refused}"
        );
    }

    #[tokio::test]
    async fn create_enforces_the_schedule_ceiling() {
        let (mut tool, _session_id, _manager) = harness().await;
        tool.config = ResolvedScheduleConfig {
            max_schedules: 1,
            ..ResolvedScheduleConfig::default()
        };
        tool.execute(
            serde_json::json!({"prompt": "first", "every": "1h"}),
            crate::tools::ToolContext::for_test(CancellationToken::new()),
        )
        .await
        .expect("first fits");
        let error = tool
            .execute(
                serde_json::json!({"prompt": "second", "every": "1h"}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect_err("second exceeds the ceiling");
        assert!(error.to_string().contains("limit"), "{error}");
    }

    #[tokio::test]
    async fn create_rejects_a_one_shot_in_the_past() {
        let (tool, _session_id, _manager) = harness().await;
        let error = tool
            .execute(
                serde_json::json!({"prompt": "x", "at": "2020-01-01T00:00:00Z"}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect_err("a past instant has no next occurrence");
        assert!(error.to_string().contains("future"), "{error}");
    }

    /// The model can read back the arguments it wrote into a tool gate.
    ///
    /// `schedule_list` is the only surface that shows them: the HTTP view and `meka schedule list`
    /// are read by parties who did not author the schedule, and arguments are where a pasted
    /// credential would sit. Without this the model could not tell a gate it built with the wrong
    /// `folder` from a correct one.
    #[tokio::test]
    async fn list_shows_a_tool_gate_s_arguments_to_the_model_that_wrote_them() {
        let (_tool, session_id, manager) = harness().await;
        // Planted through the store rather than `schedule_create`: the creation door needs a live
        // tool dispatcher to authorize a tool gate, and what is under test here is the rendering.
        let id = session_id.get().expect("harness made a session");
        let timing = Timing::parse_every("1h").expect("parses");
        let now = Utc::now();
        manager
            .schedule_store()
            .create_schedule(
                &Schedule {
                    attempts: 0,
                    id: uuid::Uuid::new_v4().to_string(),
                    session_id: id,
                    timing: timing.clone(),
                    prompt: "watch it".to_string(),
                    gate: Some(Gate {
                        probe: GateProbe::Tool {
                            name: "mcp__bridge__unseen".to_string(),
                            arguments: serde_json::json!({"folder": "sentinel-9c3f"}),
                        },
                        predicate: GatePredicate::Succeeded,
                        last_output: None,
                        permission: Permission::Read,
                    }),
                    created_at: now,
                    last_fired_at: None,
                    next_fire_at: timing.next_after(now).expect("has a next fire"),
                },
                usize::MAX,
            )
            .await
            .expect("plants the schedule");

        let list = ScheduleListTool {
            context: ScheduleContext {
                store: manager.clone(),
                site: crate::session::ToolSite::for_test()
                    .with_session_id(session_id.clone())
                    .with_permission(crate::permission::SharedPermission::new(
                        Permission::Unrestricted,
                        crate::permission::EnabledPermissions::DEFAULT,
                    )),
            },
            gate_tools: None,
        };
        let listed = list
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("lists")
            .text_content();
        assert!(listed.contains("mcp__bridge__unseen"), "{listed}");
        assert!(
            listed.contains("sentinel-9c3f"),
            "the arguments the model wrote must come back to it: {listed}"
        );
    }

    /// `schedule_list` is where the model looks when it wants detail, so it is where the reason a
    /// schedule is dead belongs. Showing the gate's *definition* and stopping there reads as
    /// healthy however long the gate has been refused.
    #[tokio::test]
    async fn list_reports_a_gate_that_cannot_currently_fire() {
        let (tool, session_id, manager) = harness().await;
        tool.execute(
            serde_json::json!({
                "prompt": "watch the build",
                "every": "1h",
                "gate": {"check": {"command": "gh pr checks"}, "when": "changed"}
            }),
            crate::tools::ToolContext::for_test(CancellationToken::new()),
        )
        .await
        .expect("creates at unrestricted");

        // What the operator did afterwards: dropped the session to `read`.
        let list = ScheduleListTool {
            context: ScheduleContext {
                store: manager.clone(),
                site: crate::session::ToolSite::for_test()
                    .with_session_id(session_id.clone())
                    .with_permission(crate::permission::SharedPermission::new(
                        Permission::Read,
                        crate::permission::EnabledPermissions::DEFAULT,
                    )),
            },
            gate_tools: None,
        };
        let listed = list
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("lists")
            .text_content();
        assert!(
            listed.contains("NOT FIRING"),
            "a withheld gate must say so: {listed}"
        );
        assert!(
            listed.contains("unrestricted"),
            "and name what it needs: {listed}"
        );
    }

    #[tokio::test]
    async fn list_and_cancel_round_trip() {
        let (tool, session_id, manager) = harness().await;
        tool.execute(
            serde_json::json!({"prompt": "watch the build", "every": "1h"}),
            crate::tools::ToolContext::for_test(CancellationToken::new()),
        )
        .await
        .expect("creates");

        let list = ScheduleListTool {
            context: ScheduleContext {
                store: manager.clone(),
                site: crate::session::ToolSite::for_test()
                    .with_session_id(session_id.clone())
                    .with_permission(crate::permission::SharedPermission::new(
                        Permission::Unrestricted,
                        crate::permission::EnabledPermissions::DEFAULT,
                    )),
            },
            gate_tools: None,
        };
        let listed = list
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("lists")
            .text_content();
        assert!(listed.contains("watch the build"), "{listed}");

        let short = listed
            .split_whitespace()
            .next()
            .expect("id is the first token")
            .to_string();
        let cancel = ScheduleCancelTool {
            context: ScheduleContext {
                store: manager,
                site: crate::session::ToolSite::for_test().with_session_id(session_id),
            },
        };
        let canceled = cancel
            .execute(
                serde_json::json!({"id": short}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("cancels")
            .text_content();
        assert!(canceled.contains("Canceled schedule"), "{canceled}");

        let after = list
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("lists")
            .text_content();
        assert!(after.contains("No schedules"), "{after}");
    }

    #[tokio::test]
    async fn cancel_reports_a_miss_rather_than_failing() {
        let (_tool, session_id, manager) = harness().await;
        let cancel = ScheduleCancelTool {
            context: ScheduleContext {
                store: manager,
                site: crate::session::ToolSite::for_test().with_session_id(session_id),
            },
        };
        let text = cancel
            .execute(
                serde_json::json!({"id": "deadbeef"}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("a miss is not an error")
            .text_content();
        assert!(text.contains("No schedule matching"), "{text}");
    }

    /// Canceling a schedule whose fire still waits tells the frontend the item was taken back, so
    /// a host that announced the fire can say what became of it.
    #[tokio::test]
    async fn cancel_announces_the_waiting_fire_it_withdraws() {
        let (_tool, session_id, manager) = harness().await;
        let now = Utc::now();
        let timing = Timing::parse_every("1h").expect("parses");
        let next_fire_at = timing.next_after(now).expect("has a next fire");
        let schedule = Schedule {
            attempts: 0,
            id: Uuid::new_v4().to_string(),
            session_id: session_id.get().expect("the session"),
            timing,
            prompt: "check the deploy".to_string(),
            gate: None,
            created_at: now,
            last_fired_at: None,
            next_fire_at,
        };
        let schedules = manager.schedule_store();
        schedules
            .create_schedule(&schedule, usize::MAX)
            .await
            .expect("create schedule");
        assert!(
            schedules
                .claim_occurrence(
                    &schedule.id,
                    next_fire_at,
                    "test",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim"),
            "the occurrence is takeable"
        );
        let crate::store::schedule::ClaimClosed::Yes(waiting) = schedules
            .queue_fire(
                &schedule,
                "test",
                Some(next_fire_at + chrono::Duration::hours(1)),
                None,
                crate::store::inbox::Fire {
                    schedule_id: schedule.id.clone(),
                    gate_output: None,
                    occurrences: 1,
                    due_at: next_fire_at,
                    fired_at: now,
                },
            )
            .await
            .expect("queue the fire")
        else {
            panic!("the fire is queued");
        };

        let frontend = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        let cancel = ScheduleCancelTool {
            context: ScheduleContext {
                store: manager,
                site: crate::session::ToolSite::for_test().with_session_id(session_id),
            },
        };
        let text = cancel
            .execute(
                serde_json::json!({"id": schedule.id}),
                crate::tools::ToolContext {
                    frontend: frontend.clone(),
                    ..crate::tools::ToolContext::for_test(CancellationToken::new())
                },
            )
            .await
            .expect("cancels")
            .text_content();
        assert!(text.contains("Canceled schedule"), "{text}");
        let withdrawn: Vec<Uuid> = frontend
            .events()
            .into_iter()
            .filter_map(|event| match event {
                crate::frontend::FrontendEvent::InboxWithdrawn { item_id } => Some(item_id),
                _ => None,
            })
            .collect();
        assert_eq!(withdrawn, [waiting.item_id]);
    }
}
