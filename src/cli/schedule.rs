//! `meka schedule`: inspecting and canceling the schedules the agent created for itself.
//!
//! Read-and-cancel only. Creating a schedule needs a session for the resulting turn to run in, and
//! picking one from the command line would mean guessing which conversation the user meant; the
//! agent creates schedules through `schedule_create`, where the session is unambiguous.

use crate::{
    cli::ScheduleAction,
    error::{MekaError, Result},
    store::Store,
};

/// Ceiling on the timing column: a spec is shown for information, and `show` has the whole one.
const TIMING_TRUNCATE: usize = 24;

/// Where the `Next` column stops counting and starts saying "not soon".
///
/// Chosen so the cell stays eight columns wide. `format_duration_short` renders `{days}d {hours}h`,
/// which is nine columns once the day count reaches four digits and the hour two, so the clamp has
/// to land before four digits, not at them. `999d 23h` is eight and `>999d` is five.
const NEVER_SOON_DAYS: i64 = 999;
const NEVER_SOON: chrono::TimeDelta = chrono::TimeDelta::days(NEVER_SOON_DAYS);

pub(crate) async fn run(
    store: &Store,
    action: &ScheduleAction,
    cli_args: &crate::cli::Cli,
) -> anyhow::Result<()> {
    match action {
        ScheduleAction::List { session, format } => {
            // Read for its refusal, not its contents: this listing answers about schedules that a
            // differently-configured host will run, and a table rendered over an unreadable config
            // is a listing of what may already be wrong.
            crate::config::ResolvedConfig::resolve(cli_args.overrides())
                .require_readable_config()?;
            list(store, session.as_deref(), *format).await
        }
        ScheduleAction::Show { id, format } => {
            // Contents as well as refusal here, since `withheld` is decided against
            // `[permissions].enabled` and answering off defaults would report a schedule as able to
            // fire that the running host declines on every sweep.
            let config = crate::config::ResolvedConfig::resolve(cli_args.overrides());
            config.require_readable_config()?;
            show_schedule(
                store,
                id,
                &config.schedule,
                None,
                *format,
                crate::render::Stream::Stdout,
            )
            .await
        }
        ScheduleAction::Cancel { id } => cancel(store, id).await,
    }
    .map_err(Into::into)
}

/// Pair each schedule with the level its session holds now, which is what decides whether it can
/// fire.
///
/// One row read per schedule, and only for a command the user typed.
///
/// The fire door's own answer, [`crate::scheduler::live_permission`], so the two cannot drift. A
/// row records what a session was *set* to, not what this installation still permits, and the two
/// diverge the moment an operator narrows `[permissions].enabled`: the fire door reads the clamped
/// level and refuses, and a listing that read the unclamped row would render the schedule as
/// healthy. A row that records no level, or one this installation does not enable, is `none` here
/// as it is there: nothing fires on a level nobody set.
///
/// A row that could not be read at all leaves this `None`, which [`withheld_summary`] reports as
/// unknown. The host level is deliberately *not* used as a fallback: the host that eventually runs
/// this schedule is some other process, and its `--permission` is not something this command can
/// see. "I cannot tell" is the honest answer.
pub(crate) async fn with_levels(
    store: &Store,
    config: &crate::config::ResolvedScheduleConfig,
    schedules: Vec<crate::schedule::Schedule>,
) -> Vec<(
    crate::schedule::Schedule,
    Option<crate::permission::Permission>,
)> {
    let mut out = Vec::with_capacity(schedules.len());
    for schedule in schedules {
        let level = match store.session_info(schedule.session_id).await {
            Ok(info) => Some(crate::scheduler::live_permission(
                crate::scheduler::SessionLookup::Read(info.as_ref()),
                config,
                schedule.session_id,
            )),
            Err(error) => {
                tracing::debug!(
                    "failed to read session {session_id}: {error}",
                    session_id = schedule.session_id
                );
                None
            }
        };
        out.push((schedule, level));
    }
    out
}

async fn list(
    store: &Store,
    session: Option<&str>,
    format: crate::cli::OutputFormat,
) -> Result<()> {
    let schedules = match session {
        // A prefix, as `meka -r` accepts and as the `Session` column prints.
        Some(raw) => {
            let id = store.resolve_session_id(raw).await?;
            store.schedule_store().list_schedules(id).await?
        }
        None => store.schedule_store().list_all_schedules().await?,
    };

    // No levels read: the unscoped table has no `Held` column to spend them on, and reading one
    // session row per schedule across every session is not free.
    let schedules: Vec<_> = schedules
        .into_iter()
        .map(|schedule| (schedule, None))
        .collect();
    // Every schedule and every session, not the rows above. `--session` narrows this listing while
    // `show` and `cancel` still scan the whole store, so an id sized to the filtered rows is one
    // they would refuse. Read only when the listing is filtered, since otherwise it is the rows.
    let resolvable = match session {
        Some(_) => Resolvable {
            schedules: store
                .schedule_store()
                .list_all_schedules()
                .await?
                .into_iter()
                .map(|schedule| schedule.id)
                .collect(),
            sessions: store.all_session_ids().await?,
        },
        None => Resolvable::of(&schedules),
    };
    render(
        store.scheduler_memory(),
        &schedules,
        Layout::Unscoped,
        None,
        crate::render::Stream::Stdout,
        &resolvable,
        format,
    )
}

/// The table's header, beside the row builder so the two cannot drift apart in width or order.
///
/// `Session` only where it can differ. `meka schedule list` spans every session, so a row without
/// it names no owner; `/schedule` on the screen is already scoped to one conversation, where the
/// column would repeat the same id on every line.
const COLUMNS: [crate::text::Column; 6] = [
    crate::text::Column::content("ID"),
    crate::text::Column::content("Session"),
    crate::text::Column::capped("Timing", TIMING_TRUNCATE),
    crate::text::Column::content("Next"),
    crate::text::Column::content("Gate"),
    crate::text::Column::remainder("Prompt"),
];

/// [`COLUMNS`] with `Held` in place of `Session`: one is unanswerable on the surface that has room
/// for the other. It sits last-but-one rather than second, because dropping a column shifts the row
/// rather than blanking a cell, and `Held` reads beside the gate it is a verdict about.
const SESSION_SCOPED_COLUMNS: [crate::text::Column; 6] = [
    crate::text::Column::content("ID"),
    crate::text::Column::capped("Timing", TIMING_TRUNCATE),
    crate::text::Column::content("Next"),
    crate::text::Column::content("Gate"),
    crate::text::Column::content("Held"),
    crate::text::Column::remainder("Prompt"),
];

/// Which of the two tables is being built.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Layout {
    /// `meka schedule list`: every session, so each row must name its own.
    Unscoped,
    /// `/schedule`: one session, and a host that can resolve a gate.
    SessionScoped,
}

/// What a printed prefix will be resolved against: every schedule, and every session that owns one.
///
/// Not the rows being drawn. `meka schedule list --session <id>` shows one conversation's schedules
/// while `show` and `cancel` scan all of them, so a width sized to the rows printed an id those
/// commands then refused as ambiguous.
struct Resolvable {
    schedules: Vec<String>,
    sessions: Vec<String>,
}

impl Resolvable {
    /// The rows as their own universe, which is the truth for an unfiltered listing and what a
    /// test about layout wants.
    fn of(
        schedules: &[(
            crate::schedule::Schedule,
            Option<crate::permission::Permission>,
        )],
    ) -> Self {
        Self {
            schedules: schedules
                .iter()
                .map(|(schedule, _)| schedule.id.clone())
                .collect(),
            sessions: schedules
                .iter()
                .map(|(schedule, _)| schedule.session_id.to_string())
                .collect(),
        }
    }
}

/// One row per schedule, separated from printing so it can be tested.
///
/// The `Prompt` cell carries text Meka did not write, possibly composed from a page the agent
/// fetched, so it goes through [`crate::text::sanitize_to_line`]. The separation from `render`
/// matters for exactly that: `render` prints, so sanitizing done there has no way to be asserted.
///
/// `Timing` goes through the same helper without ever having been a way in: everything
/// `Timing::describe` can emit is a formatted timestamp, a `humantime` duration, or a cron
/// pattern, and croner's five-field grammar admits no control characters at either the creation or
/// the rehydration door. It is sanitized so the guarantee this function makes does not quietly
/// depend on a parser two modules away staying that strict. The rest of the row is an id, a
/// duration, and a word from an enum.
fn rows_for(
    memory: &crate::schedule::SchedulerMemory,
    schedules: &[(
        crate::schedule::Schedule,
        Option<crate::permission::Permission>,
    )],
    layout: Layout,
    tools: Option<&dyn crate::schedule::GateTools>,
    resolvable: &Resolvable,
) -> Vec<Vec<String>> {
    let now = chrono::Utc::now();
    // Widened only as far as resolving requires, and sized against `resolvable` rather than the
    // rows, because `--session` narrows the listing while `show` and `cancel` still scan every
    // schedule.
    let schedule_width = crate::text::unique_prefix_len_within(
        schedules.iter().map(|(schedule, _)| schedule.id.as_str()),
        resolvable.schedules.iter().map(String::as_str),
    )
    .max(crate::text::ID_PREFIX);
    let sessions: Vec<String> = schedules
        .iter()
        .map(|(schedule, _)| schedule.session_id.to_string())
        .collect();
    let session_width = crate::text::unique_prefix_len_within(
        sessions.iter().map(String::as_str),
        resolvable.sessions.iter().map(String::as_str),
    )
    .max(crate::text::ID_PREFIX);

    schedules
        .iter()
        .zip(sessions.iter())
        .map(|((schedule, level), session)| {
            let mut row = vec![
                schedule
                    .id
                    .get(..schedule_width)
                    .unwrap_or(&schedule.id)
                    .to_string(),
            ];
            if layout == Layout::Unscoped {
                row.push(session.get(..session_width).unwrap_or(session).to_string());
            }
            row.push(schedule.timing.describe());
            // Relative, because the absolute instant cost 23 columns to answer a question a reader
            // asks in the relative form. An occurrence already due reads as `due` rather than as
            // `0s`: a host that was down through it has not missed it, and the two are different
            // states. `show` carries the instant.
            row.push(match schedule.next_fire_at - now {
                remaining if remaining <= chrono::TimeDelta::zero() => "due".to_string(),
                // Bounded, unlike a duration rendered straight. `Timing::parse_at` takes any RFC
                // 3339 instant, so a schedule dated 3121 reads `399999d 23h` and spends 11 of the 8
                // columns this table budgeted for it, pushing the row past 120. Nobody reads a
                // four-digit day count as anything but "not soon", and `show` carries the instant.
                remaining if remaining >= NEVER_SOON => format!(">{NEVER_SOON_DAYS}d"),
                remaining => crate::text::format_duration_short(remaining.num_seconds()),
            });
            // The kind alone. What the gate *runs* is model-authored text that no column can hold
            // legibly, and `shell` against `tool` is the distinction worth a column: an unsandboxed
            // `sh -c` against a structured call. A fixed word from an enum, so it needs no
            // sanitizing.
            row.push(match &schedule.gate {
                Some(gate) => gate.probe.kind_str().to_string(),
                None => "-".to_string(),
            });
            // Three answers, all spelled: a blank cell reads as a missing field to anything that
            // splits the row on whitespace.
            if layout == Layout::SessionScoped {
                row.push(
                    match crate::schedule::schedule_withheld(memory, schedule, *level, tools) {
                        crate::schedule::Withheld::Yes(_) => "yes".to_string(),
                        crate::schedule::Withheld::Undetermined => "unknown".to_string(),
                        crate::schedule::Withheld::No => "no".to_string(),
                    },
                );
            }
            row.push(crate::text::prose_cell(&schedule.prompt));
            row
        })
        .collect()
}

fn render(
    memory: &crate::schedule::SchedulerMemory,
    schedules: &[(
        crate::schedule::Schedule,
        Option<crate::permission::Permission>,
    )],
    layout: Layout,
    tools: Option<&dyn crate::schedule::GateTools>,
    stream: crate::render::Stream,
    resolvable: &Resolvable,
    format: crate::cli::OutputFormat,
) -> Result<()> {
    if format == crate::cli::OutputFormat::Json {
        let views: Vec<crate::view::ScheduleView> = schedules
            .iter()
            .map(|(schedule, level)| schedule_view(memory, schedule, *level, tools))
            .collect();
        crate::cli::write_json_listing("schedules", &views)?;
        return Ok(());
    }
    if schedules.is_empty() {
        // stderr: an empty list is a status note, not the data a script asked for. A caller piping
        // this still gets a clean empty stdout.
        crate::streams::write_stderr_line("No schedules.");
        return Ok(());
    }

    let columns: &[crate::text::Column] = match layout {
        Layout::Unscoped => &COLUMNS,
        Layout::SessionScoped => &SESSION_SCOPED_COLUMNS,
    };
    stream.write(crate::text::format_table(
        columns,
        &rows_for(memory, schedules, layout, tools, resolvable),
    ))?;
    Ok(())
}

/// `meka schedule show <id>`: one schedule, in full.
///
/// The table exists to be scanned, so every cell in it is bounded. This is the surface that answers
/// what a schedule actually does (the whole prompt, the whole command a gate runs, the session it
/// wakes), none of which survives a column. It is also the only place `Held` is explained rather
/// than encoded, which matters because a withheld schedule is one that will never fire.
pub(crate) async fn show(
    store: &Store,
    id_prefix: &str,
    config: &crate::config::ResolvedScheduleConfig,
    scope: Option<uuid::Uuid>,
) -> Result<()> {
    show_schedule(
        store,
        id_prefix,
        config,
        scope,
        crate::cli::OutputFormat::Plain,
        crate::render::Stream::Stderr,
    )
    .await
}

/// [`show`] under a chosen format; the screen's `/schedule show` has no format flag and takes the
/// door above.
async fn show_schedule(
    store: &Store,
    id_prefix: &str,
    config: &crate::config::ResolvedScheduleConfig,
    scope: Option<uuid::Uuid>,
    format: crate::cli::OutputFormat,
    stream: crate::render::Stream,
) -> Result<()> {
    let schedule = resolve_schedule(store, id_prefix, scope).await?;
    let level = with_levels(store, config, vec![schedule]).await;
    let Some((schedule, level)) = level.into_iter().next() else {
        return Err(MekaError::Usage(format!(
            "schedule '{id_prefix}' was already gone"
        )));
    };
    if format == crate::cli::OutputFormat::Json {
        crate::cli::write_json(&schedule_view(
            store.scheduler_memory(),
            &schedule,
            level,
            None,
        ))?;
        return Ok(());
    }

    let mut fields = vec![
        ("id", schedule.id.clone()),
        ("session", schedule.session_id.to_string()),
        ("timing", sanitize_untruncated(&schedule.timing.describe())),
        (
            "next fire",
            crate::text::format_timestamp(schedule.next_fire_at, crate::text::Precision::Minutes),
        ),
        ("last fired", match schedule.last_fired_at {
            Some(at) => crate::text::format_timestamp(at, crate::text::Precision::Minutes),
            None => "never".to_string(),
        }),
        (
            "created",
            crate::text::format_timestamp(schedule.created_at, crate::text::Precision::Minutes),
        ),
        // Every value below is model-authored or server-chosen, so all of it is sanitized.
        // Untruncated is the point of this command; `sanitize_to_line` still collapses anything
        // that could forge a line, which is the part that was never about width.
        (
            "withheld",
            withheld_summary(store.scheduler_memory(), &schedule, level, None).to_string(),
        ),
    ];
    match &schedule.gate {
        Some(gate) => {
            fields.push(("gate when", sanitize_untruncated(&gate.predicate.summary())));
            fields.push((
                "gate check",
                format!(
                    "{} {}",
                    gate.probe.kind_str(),
                    sanitize_untruncated(&gate.probe.summary())
                ),
            ));
        }
        None => fields.push(("gate", "none (fires on schedule)".to_string())),
    }
    // Line structure preserved and indented, as `/task show` does for a task's output: a schedule's
    // prompt is prose the model wrote for its own future self and is routinely multi-line, and the
    // indent is what stops a line in it reading as another field.
    fields.push(("prompt", String::new()));
    let mut out = crate::text::format_fields(&fields);
    for line in schedule.prompt.lines() {
        out.push_str("  ");
        out.push_str(&sanitize_untruncated(line));
        out.push('\n');
    }

    stream.write(&out)?;
    Ok(())
}

/// A schedule as `--format json` prints it. The operator at this terminal can read every schedule
/// in the store, so the gate's command is always given. `withheld` is present only with a reason
/// the schedule cannot fire, as on the wire: this process cannot resolve a tool gate, and "cannot
/// tell" is not a reason.
fn schedule_view(
    memory: &crate::schedule::SchedulerMemory,
    schedule: &crate::schedule::Schedule,
    level: Option<crate::permission::Permission>,
    tools: Option<&dyn crate::schedule::GateTools>,
) -> crate::view::ScheduleView {
    let withheld = match crate::schedule::schedule_withheld(memory, schedule, level, tools) {
        crate::schedule::Withheld::Yes(reason) => Some(reason),
        crate::schedule::Withheld::No | crate::schedule::Withheld::Undetermined => None,
    };
    crate::view::ScheduleView::new(schedule, true, withheld)
}

/// Whether a schedule can fire, spelled out.
///
/// Three answers, because this command has no MCP manager and so cannot resolve a tool gate at all.
/// Encoded as `yes` / `?` / blank, a blank would be a verdict that reads as missing data; with a
/// line to itself each one can say which it is.
fn withheld_summary(
    memory: &crate::schedule::SchedulerMemory,
    schedule: &crate::schedule::Schedule,
    level: Option<crate::permission::Permission>,
    tools: Option<&dyn crate::schedule::GateTools>,
) -> &'static str {
    match crate::schedule::schedule_withheld(memory, schedule, level, tools) {
        crate::schedule::Withheld::Yes(_) => "yes: this schedule cannot fire",
        // `Undetermined` has two causes and they send an operator to different places, so the one
        // this reader is in is named rather than the commoner one guessed at. A missing level is
        // its own answer: the row records no permission, or the one it records is not in
        // `[permissions] enabled` any more, which is the case that makes a schedule stop firing
        // after a config change and has nothing to do with gates.
        crate::schedule::Withheld::Undetermined if level.is_none() => {
            "unknown: this session's permission level could not be established"
        }
        crate::schedule::Withheld::Undetermined => {
            "unknown: this process cannot resolve a tool gate"
        }
        crate::schedule::Withheld::No => "no",
    }
}

/// Sanitize without a ceiling, for [`show`], whose whole job is to not truncate.
fn sanitize_untruncated(text: &str) -> String {
    crate::text::sanitize_to_line(text, usize::MAX)
}

/// Find the one schedule whose id starts with `id_prefix`, for `show`, or say why not.
///
/// `scope` is `None` for the CLI, which scans every session because the operator has an id, not a
/// session, and requiring them to find the session first would make the id useless on its own. It
/// is `Some` for `/schedule show`, which answers inside one conversation like everything else on
/// that surface. [`cancel`] resolves over the stored ids instead, through the store's resolver,
/// so it reaches a row this build cannot decode, which has nothing to show.
async fn resolve_schedule(
    store: &Store,
    id_prefix: &str,
    scope: Option<uuid::Uuid>,
) -> Result<crate::schedule::Schedule> {
    if !crate::text::is_usable_id_prefix(id_prefix) {
        return Err(no_schedule_matching(id_prefix));
    }
    let schedules = match scope {
        // `/schedule` on the screen is one conversation's view, and its `cancel` resolves against
        // that session alone. A `show` that reached outside it would answer about a schedule the
        // listing above it never mentioned and the `cancel` beside it would refuse.
        Some(session) => store.schedule_store().list_schedules(session).await?,
        None => store.schedule_store().list_all_schedules().await?,
    };
    let wanted = crate::text::id_prefix_for_matching(id_prefix);
    let mut matches = schedules
        .into_iter()
        .filter(|schedule| schedule.id.starts_with(&wanted));
    match (matches.next(), matches.next()) {
        (None, _) => Err(no_schedule_matching(id_prefix)),
        (Some(schedule), None) => Ok(schedule),
        (Some(first), Some(second)) => Err(ambiguous_schedule_id(id_prefix, &first.id, &second.id)),
    }
}

fn no_schedule_matching(id_prefix: &str) -> MekaError {
    MekaError::Usage(format!("no schedule matching '{id_prefix}'"))
}

/// Full ids, as `resolve_session_id` reports: every match shares the prefix, so echoing it back
/// twice names nothing the caller could retype.
fn ambiguous_schedule_id(id_prefix: &str, first: &str, second: &str) -> MekaError {
    MekaError::Usage(format!(
        "ambiguous schedule id '{id_prefix}' matches at least: {first}, {second}"
    ))
}

async fn cancel(store: &Store, id_prefix: &str) -> Result<()> {
    let id = match store
        .schedule_store()
        .schedule_ids_matching(id_prefix, None)
        .await?
        .as_slice()
    {
        [] => return Err(no_schedule_matching(id_prefix)),
        [id] => id.clone(),
        [first, second, ..] => return Err(ambiguous_schedule_id(id_prefix, first, second)),
    };
    // Reported from the delete's own row count, not from the resolution that found the schedule: a
    // scheduler sweep can retire it in between, and `ok: canceled` about a schedule this command
    // did not cancel is indistinguishable from the real thing.
    match store.schedule_store().delete_schedule(&id).await? {
        Some(_) => {
            tracing::info!("canceled schedule {id}");
            Ok(())
        }
        None => Err(MekaError::Usage(format!(
            "schedule '{}' was already gone",
            id.get(..crate::text::ID_PREFIX).unwrap_or(&id)
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        permission::Permission,
        schedule::{Gate, GatePredicate, GateProbe, Schedule, Timing},
    };

    /// `/schedule show` answers inside the conversation, like everything else on that surface.
    ///
    /// `/schedule` lists one session's schedules and `/schedule cancel` resolves against that
    /// session alone. A `show` that scanned every session would print a schedule the listing above
    /// it never mentioned and the `cancel` beside it would refuse, and would leak another
    /// conversation's prompt and gate command into this one. The CLI passes `None` and does scan
    /// everything, because there the operator has an id and no session.
    #[tokio::test]
    async fn the_repl_resolves_a_schedule_only_inside_its_own_session() {
        let manager = crate::store::Store::open(
            Some(std::path::Path::new(":memory:")),
            &crate::store::migrations::Context::adopting(None),
        )
        .await
        .expect("open");
        let mine = manager
            .create_session(None, "p".to_string())
            .await
            .expect("session");
        let theirs = manager
            .create_session(None, "p".to_string())
            .await
            .expect("session");

        let mut schedule = schedule_with(None);
        schedule.session_id = theirs;
        manager
            .schedule_store()
            .create_schedule(&schedule, usize::MAX)
            .await
            .expect("create");

        assert!(
            resolve_schedule(&manager, &schedule.id[..8], Some(mine))
                .await
                .is_err(),
            "another session's schedule must not resolve here"
        );
        assert_eq!(
            resolve_schedule(&manager, &schedule.id[..8], Some(theirs))
                .await
                .expect("its own session finds it")
                .id,
            schedule.id
        );
        assert_eq!(
            resolve_schedule(&manager, &schedule.id[..8], None)
                .await
                .expect("the CLI scans every session")
                .id,
            schedule.id
        );
    }

    /// `meka schedule cancel` reaches a row this build cannot decode: every listing skips it, and
    /// the store's warning about it names this command as the way to cancel it.
    #[tokio::test]
    async fn the_cli_cancels_a_schedule_this_build_cannot_decode() {
        let manager = crate::store::Store::for_test().await;
        let session = manager
            .create_session(None, "p".to_string())
            .await
            .expect("session");
        let id = uuid::Uuid::new_v4().to_string();
        let (row_id, row_session) = (id.clone(), session.to_string());
        let connection = manager.schedule_store().connection.clone();
        connection
            .call(move |connection| {
                connection.execute(
                    "INSERT INTO schedules (id, session_id, kind, spec, prompt, created_at, \
                     next_fire_at) VALUES (?1, ?2, 'bogus', 'nothing this build reads', 'p', ?3, ?3)",
                    rusqlite::params![row_id, row_session, chrono::Utc::now().to_rfc3339()],
                )
            })
            .await
            .expect("a row no build decodes");

        cancel(&manager, &id[..8])
            .await
            .expect("cancel resolves the stored id");
        let left: i64 = connection
            .call(move |connection| {
                connection.query_row(
                    "SELECT count(*) FROM schedules WHERE id = ?1",
                    [id],
                    |row| row.get(0),
                )
            })
            .await
            .expect("count");
        assert_eq!(left, 0, "the row is gone");
    }

    fn schedule_with(gate: Option<Gate>) -> Schedule {
        Schedule {
            id: "7f3a1b2c-0000-0000-0000-000000000000".to_string(),
            session_id: uuid::Uuid::nil(),
            timing: Timing::parse_every("1h").expect("parses"),
            prompt: "watch the thing".to_string(),
            gate,
            created_at: chrono::Utc::now(),
            last_fired_at: None,
            next_fire_at: chrono::Utc::now(),
            attempts: 0,
        }
    }

    fn shell_gate(command: &str, predicate: GatePredicate) -> Gate {
        Gate {
            probe: GateProbe::Shell {
                command: command.to_string(),
            },
            predicate,
            last_output: None,
            permission: Permission::Unrestricted,
        }
    }

    /// Nothing the model wrote reaches the terminal as control characters or as extra rows.
    ///
    /// Four cells carry model-authored text: a shell command, a tool name an MCP server chose, a
    /// regular expression or JSON pointer the predicate gained with `matches` and `at`, and the
    /// prompt itself. A `When` column that changes from a fixed enum word to authored text is the
    /// kind of regression a listing that only ever printed could not be asked about.
    ///
    /// An escape goes into every one of them. Looping over `row` while planting in only two would
    /// leave the `Prompt` cell unsanitized under a test that reads as though it covered
    /// everything: the loop honest and the corpus not.
    #[test]
    fn no_cell_can_carry_an_escape_or_forge_a_row() {
        let forged = "/chats\u{1b}[31m\nffffffff  every 1m  now  changed  -  -  harmless";
        let mut schedule = schedule_with(Some(shell_gate(
            "gh pr checks\u{1b}[31m\nsecond line",
            GatePredicate::At {
                pointer: forged.to_string(),
                is: crate::schedule::PointerTest::NotEmpty,
            },
        )));
        // A prompt is prose, and an agent that composed one from a page it fetched is the ordinary
        // way something hostile arrives in this column.
        schedule.prompt =
            "watch the thing\u{1b}[31m\nffffffff  every 1m  now  -  -  -  harmless".to_string();
        let schedules = [(schedule, None)];
        let rows = rows_for(
            &crate::schedule::SchedulerMemory::default(),
            &schedules,
            Layout::Unscoped,
            None,
            &Resolvable::of(&schedules),
        );

        let table = crate::text::format_table(&COLUMNS, &rows);
        assert_eq!(table.lines().count(), 2, "one schedule, one row: {table}");
        for line in table.lines() {
            assert!(
                !line.contains('\u{1b}'),
                "a line reaches a terminal verbatim, so an escape may not survive: {line:?}"
            );
            assert!(
                crate::text::display_width(line) <= crate::text::TABLE_WIDTH,
                "and the row is bounded, or one long spec or a paragraph of prompt wraps it into \
                 unreadability: {line:?}"
            );
        }
    }

    /// The recorded level is clamped by the enabled set before anything is concluded from it.
    ///
    /// Where the fire door reads a session's recorded level it also applies `[permissions].enabled`
    /// and refuses what falls outside it. This listing did not, so narrowing the enabled set and
    /// restarting left the operator's own table reporting a schedule as healthy that the running
    /// host declines on every sweep. Exercised through `with_levels` rather than `rows_for`,
    /// because `rows_for` is handed a level that has already been through this and would pass
    /// either way.
    #[tokio::test]
    async fn a_level_the_enabled_set_excludes_is_not_treated_as_the_session_level() {
        let manager =
            crate::store::Store::open(Some(std::path::Path::new(":memory:")), &Default::default())
                .await
                .expect("open in-memory database");
        let session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        manager
            .update_session(session, crate::store::SessionPatch {
                permission: Some(Permission::Unrestricted),
                ..Default::default()
            })
            .await
            .expect("record the level the session was set to");

        let mut schedule = schedule_with(None);
        schedule.session_id = session;

        let permissive = crate::config::ResolvedScheduleConfig::default();
        assert_eq!(
            with_levels(&manager, &permissive, vec![schedule.clone()]).await[0].1,
            Some(Permission::Unrestricted),
            "with the default enabled set the recorded level is the answer"
        );

        let narrowed = crate::config::ResolvedScheduleConfig {
            enabled_permissions: crate::permission::EnabledPermissions::from_levels([
                Permission::Read,
            ])
            .expect("a single level is a non-empty set"),
            ..crate::config::ResolvedScheduleConfig::default()
        };
        assert_eq!(
            with_levels(&manager, &narrowed, vec![schedule]).await[0].1,
            Some(Permission::None),
            "once the installation stops permitting that level the row runs nothing, which is what \
             the fire door reads too, instead of reporting the schedule as healthy"
        );
    }

    /// A verdict that needs no permission level is still reached without one.
    ///
    /// Parking is decided by the attempt count alone. Matching on the level first and answering
    /// `?` when it is absent would make the one surface an operator opens to find out why a
    /// schedule stopped decline to say, about the one schedule it could have answered for outright.
    #[test]
    fn a_parked_schedule_is_reported_withheld_even_with_no_session_level() {
        let mut schedule = schedule_with(None);
        schedule.attempts = crate::schedule::MAX_CLAIM_ATTEMPTS;
        assert!(
            withheld_summary(
                &crate::schedule::SchedulerMemory::default(),
                &schedule,
                None,
                None
            )
            .starts_with("yes"),
            "a parked schedule is provably dead whatever the session is set to"
        );
    }

    /// A level this command could not establish must not read as one that permits the schedule.
    ///
    /// Two ways to get there, and the second is why the enabled set is applied here at all: the row
    /// carries no level, or it carries one this installation does not permit. The fire door clamps
    /// by `[permissions].enabled` and refuses; a listing that did not would report the schedule as
    /// healthy to the one audience that came here to check.
    #[test]
    fn an_unestablished_level_is_not_reported_as_healthy() {
        let unknown = withheld_summary(
            &crate::schedule::SchedulerMemory::default(),
            &schedule_with(Some(shell_gate("gh pr checks", GatePredicate::Changed))),
            None,
            None,
        );
        assert!(
            unknown.starts_with("unknown"),
            "a session whose level could not be read is unestablished, not healthy: {unknown}"
        );
    }

    /// The verdict answers three things, and each says which it is.
    ///
    /// This listing has no MCP manager, so it cannot resolve a tool gate at all. Rendering that as
    /// blank put "I cannot tell" in the same cell as "it will fire", under a header that reads as
    /// the latter.
    #[test]
    fn the_verdict_distinguishes_cannot_fire_from_cannot_tell() {
        let held = withheld_summary(
            &crate::schedule::SchedulerMemory::default(),
            &schedule_with(Some(shell_gate("gh pr checks", GatePredicate::Changed))),
            Some(Permission::Read),
            None,
        );
        assert!(
            held.starts_with("yes"),
            "a shell gate below `unrestricted` cannot fire: {held}"
        );

        let undetermined = withheld_summary(
            &crate::schedule::SchedulerMemory::default(),
            &schedule_with(Some(Gate {
                probe: GateProbe::Tool {
                    name: "mcp__bridge__unseen".to_string(),
                    arguments: serde_json::json!({}),
                },
                predicate: GatePredicate::Changed,
                last_output: None,
                permission: Permission::Read,
            })),
            Some(Permission::Read),
            None,
        );
        assert!(
            undetermined.starts_with("unknown"),
            "a tool gate this process cannot resolve is unestablished, not healthy: {undetermined}"
        );

        let healthy = withheld_summary(
            &crate::schedule::SchedulerMemory::default(),
            &schedule_with(Some(shell_gate("gh pr checks", GatePredicate::Changed))),
            Some(Permission::Unrestricted),
            None,
        );
        assert_eq!(healthy, "no", "and an authorized gate is not withheld");
    }

    /// The `Held` cell distinguishes a verdict from a shrug.
    ///
    /// `/schedule` is the only surface that renders this cell, and its three values mean different
    /// things: `yes` is "this schedule cannot fire", blank is "it can", and `?` is "this process
    /// cannot establish which". Blank and `?` are the pair worth pinning, because swapping them
    /// turns "I did not check" into "it will fire" on the surface an operator uses to find out why
    /// a schedule is silent.
    #[test]
    fn the_held_cell_separates_a_verdict_from_an_unanswerable_question() {
        let held_cell = |gate: Option<Gate>, level: Option<Permission>| {
            let schedules = [(schedule_with(gate), level)];
            rows_for(
                &crate::schedule::SchedulerMemory::default(),
                &schedules,
                Layout::SessionScoped,
                None,
                &Resolvable::of(&schedules),
            )[0][4]
                .clone()
        };

        assert_eq!(
            held_cell(
                Some(shell_gate("gh pr checks", GatePredicate::Changed)),
                Some(Permission::Read)
            ),
            "yes",
            "a shell gate below `unrestricted` cannot fire, and the column says so"
        );
        assert_eq!(
            held_cell(
                Some(shell_gate("gh pr checks", GatePredicate::Changed)),
                Some(Permission::Unrestricted)
            ),
            "no",
            "an authorized gate will fire, and the column says so rather than `not checked`"
        );
        assert_eq!(
            held_cell(
                Some(shell_gate("gh pr checks", GatePredicate::Changed)),
                None
            ),
            "unknown",
            "a level that could not be established is unanswerable, not healthy"
        );
    }

    /// The table holds 120 columns against the widest cell every column can produce.
    ///
    /// `format_duration_short` renders `{days}d {hours}h`, nine columns once the days reach four
    /// digits, against the eight the budget reserves. Sizing each cell to its own ceiling and never
    /// measuring the row is how a table that fits on every real store overruns on one schedule.
    #[test]
    fn the_table_holds_its_budget_against_the_widest_row_it_can_draw() {
        let mut schedule = schedule_with(Some(shell_gate("gh pr checks", GatePredicate::Changed)));
        schedule.prompt = "x".repeat(400);
        schedule.timing = Timing::parse_every("31536000s").expect("parses");
        // A horizon past the clamp, and one just under it: the second is the wider cell.
        // The middle one is the case that overran: four-digit days with a two-digit hour renders
        // `9990d 14h`, nine columns against the eight the budget reserves. It is only reachable
        // below a clamp set at four digits, which is why the clamp sits at three.
        for offset in [
            chrono::Duration::days(400_000),
            chrono::Duration::days(9_990) + chrono::Duration::hours(14),
            chrono::Duration::days(999) + chrono::Duration::hours(23),
        ] {
            let mut schedule = schedule.clone();
            schedule.next_fire_at = chrono::Utc::now() + offset;
            for (layout, columns) in [
                (Layout::Unscoped, &COLUMNS),
                (Layout::SessionScoped, &SESSION_SCOPED_COLUMNS),
            ] {
                let schedules = [(schedule.clone(), Some(Permission::Read))];
                let table = crate::text::format_table(
                    columns,
                    &rows_for(
                        &crate::schedule::SchedulerMemory::default(),
                        &schedules,
                        layout,
                        None,
                        &Resolvable::of(&schedules),
                    ),
                );
                for row in table.lines() {
                    let width = crate::text::display_width(row);
                    assert!(
                        width <= crate::text::TABLE_WIDTH,
                        "{layout:?} row is {width} columns against a budget of {}: {row}",
                        crate::text::TABLE_WIDTH
                    );
                }
            }
        }
    }

    /// A tool name is also a valid command, so the kind has to be on the row.
    ///
    /// The column carries the kind and nothing else, which is what makes it 5 columns wide instead
    /// of 40. The distinction it exists for survives that: an unsandboxed `sh -c` and a
    /// structured call still cannot read alike. What each one *runs* is [`show`]'s job.
    #[test]
    fn the_gate_column_names_the_kind() {
        let shell = &rows_for(
            &crate::schedule::SchedulerMemory::default(),
            &[(
                schedule_with(Some(shell_gate("web_fetch", GatePredicate::Changed))),
                None,
            )],
            Layout::Unscoped,
            None,
            &Resolvable::of(&[(
                schedule_with(Some(shell_gate("web_fetch", GatePredicate::Changed))),
                None,
            )]),
        )[0][4];
        let tool = &rows_for(
            &crate::schedule::SchedulerMemory::default(),
            &[(
                schedule_with(Some(Gate {
                    probe: GateProbe::Tool {
                        name: "web_fetch".to_string(),
                        arguments: serde_json::json!({}),
                    },
                    predicate: GatePredicate::Changed,
                    last_output: None,
                    permission: Permission::Read,
                })),
                None,
            )],
            Layout::Unscoped,
            None,
            &Resolvable::of(&[(
                schedule_with(Some(Gate {
                    probe: GateProbe::Tool {
                        name: "web_fetch".to_string(),
                        arguments: serde_json::json!({}),
                    },
                    predicate: GatePredicate::Changed,
                    last_output: None,
                    permission: Permission::Read,
                })),
                None,
            )]),
        )[0][4];

        assert_eq!(shell, "shell");
        assert_eq!(tool, "tool");
        assert_ne!(
            shell, tool,
            "an unsandboxed `sh -c` and a structured call must not read alike"
        );
        assert_eq!(
            rows_for(
                &crate::schedule::SchedulerMemory::default(),
                &[(schedule_with(None), None)],
                Layout::Unscoped,
                None,
                &Resolvable::of(&[(schedule_with(None), None)])
            )[0][4],
            "-",
            "and an ungated schedule says so, rather than leaving the cell ambiguous"
        );
    }

    /// `meka schedule list` spans every session, so a row that does not name one names nothing.
    ///
    /// The column is absent from the screen's listing, which is already scoped to one conversation.
    /// Both header sets are asserted against the row width, since a header and a row that disagree
    /// shift every cell under the wrong title without failing anything.
    #[test]
    fn the_session_column_appears_only_where_it_can_differ() {
        let schedule = schedule_with(None);
        let unscoped = &rows_for(
            &crate::schedule::SchedulerMemory::default(),
            &[(schedule.clone(), None)],
            Layout::Unscoped,
            None,
            &Resolvable::of(&[(schedule.clone(), None)]),
        )[0];
        let scoped = &rows_for(
            &crate::schedule::SchedulerMemory::default(),
            &[(schedule.clone(), None)],
            Layout::SessionScoped,
            None,
            &Resolvable::of(&[(schedule.clone(), None)]),
        )[0];

        assert_eq!(unscoped.len(), COLUMNS.len());
        assert_eq!(scoped.len(), SESSION_SCOPED_COLUMNS.len());
        assert_eq!(
            unscoped[1],
            schedule.session_id.to_string()[..8],
            "the session cell is the id's prefix, which is what `--session` now accepts"
        );
        assert_eq!(
            scoped[1], unscoped[2],
            "and dropping it must shift the row, not blank one cell"
        );
    }

    /// An id column shows a first segment until that stops identifying a row.
    ///
    /// A prefix is what the reader retypes into `show`, `cancel` or `--session`, so printing one
    /// that matches two rows hands them a string those commands refuse. Widening only on collision
    /// is what keeps two UUID columns from spending 76 of the 120 available on every other listing.
    #[test]
    fn an_id_widens_only_far_enough_to_stay_unique() {
        let distinct = [
            "4d71eeca-9f21-4c3a-b8e7-1a2b3c4d5e6f",
            "1fb28dc6-0000-0000-0000-000000000000",
        ];
        assert_eq!(
            crate::text::unique_prefix_len(distinct.iter().copied()),
            crate::text::ID_PREFIX,
            "ids that differ in the first segment need no more than the first segment"
        );

        let colliding = [
            "4d71eeca-9f21-4c3a-b8e7-1a2b3c4d5e6f",
            "4d71eeca-0000-0000-0000-000000000000",
        ];
        let width = crate::text::unique_prefix_len(colliding.iter().copied());
        assert!(
            width > crate::text::ID_PREFIX,
            "a shared first segment has to widen"
        );
        assert_ne!(
            colliding[0].as_bytes()[width - 1],
            b'-',
            "and must not stop on the hyphen: {:?}",
            &colliding[0][..width]
        );
        assert_ne!(
            &colliding[0][..width],
            &colliding[1][..width],
            "the widened prefixes still have to differ"
        );

        assert_eq!(
            crate::text::unique_prefix_len(
                ["4d71eeca-9f21-4c3a-b8e7-1a2b3c4d5e6f"].iter().copied()
            ),
            crate::text::ID_PREFIX,
            "one row collides with nothing"
        );
    }

    /// Repeating an id is ordinary, and must not be mistaken for a collision.
    ///
    /// One session with several schedules fills the whole `Session` column with itself. Counting
    /// rows rather than distinct ids made that unsatisfiable, so the column widened to a full UUID
    /// to distinguish an id from itself, and charged the prompt 28 columns for it.
    #[test]
    fn one_session_with_several_schedules_does_not_widen_the_session_column() {
        let session = uuid::Uuid::new_v4();
        let schedule = |id: &str| {
            let mut schedule = schedule_with(None);
            schedule.id = id.to_string();
            schedule.session_id = session;
            schedule.prompt = "x".repeat(200);
            schedule
        };
        let rows = rows_for(
            &crate::schedule::SchedulerMemory::default(),
            &[
                (schedule("7f3a1b2c-0000-0000-0000-000000000000"), None),
                (schedule("9e5d4c3b-0000-0000-0000-000000000000"), None),
            ],
            Layout::Unscoped,
            None,
            &Resolvable::of(&[
                (schedule("7f3a1b2c-0000-0000-0000-000000000000"), None),
                (schedule("9e5d4c3b-0000-0000-0000-000000000000"), None),
            ]),
        );

        assert_eq!(
            rows[0][1].chars().count(),
            crate::text::ID_PREFIX,
            "two schedules in one session are not two sessions: {:?}",
            rows[0][1]
        );
        for row in crate::text::format_table(&COLUMNS, &rows).lines().skip(1) {
            assert_eq!(
                crate::text::display_width(row),
                crate::text::TABLE_WIDTH,
                "and the prompt keeps its full width, having paid for nothing: {row}"
            );
        }
    }

    /// Widening an id is paid for out of the prompt, so the table holds its budget either way.
    #[test]
    fn a_widened_id_is_paid_for_out_of_the_prompt() {
        let long = "x".repeat(200);
        let mut first = schedule_with(None);
        let mut second = schedule_with(None);
        first.id = "4d71eeca-9f21-4c3a-b8e7-1a2b3c4d5e6f".to_string();
        second.id = "4d71eeca-0000-0000-0000-000000000000".to_string();
        first.prompt = long.clone();
        second.prompt = long;

        let schedules = [(first, None), (second, None)];
        let rows = rows_for(
            &crate::schedule::SchedulerMemory::default(),
            &schedules,
            Layout::SessionScoped,
            None,
            &Resolvable::of(&schedules),
        );
        let widened = rows[0][0].chars().count();
        assert!(widened > crate::text::ID_PREFIX);
        for row in crate::text::format_table(&SESSION_SCOPED_COLUMNS, &rows)
            .lines()
            .skip(1)
        {
            assert_eq!(
                crate::text::display_width(row),
                crate::text::TABLE_WIDTH,
                "the prompt gives back exactly what the id took: {row}"
            );
        }
    }

    /// An occurrence that is already due is a different state from one due in no time at all.
    ///
    /// A host that was down through a fire time leaves the schedule overdue, and
    /// `format_duration_short` clamps a negative duration to `0s`, which reads as "about to fire"
    /// for a schedule that should already have.
    #[test]
    fn an_overdue_occurrence_reads_as_due() {
        let mut schedule = schedule_with(None);
        schedule.next_fire_at = chrono::Utc::now() - chrono::Duration::hours(3);
        assert_eq!(
            rows_for(
                &crate::schedule::SchedulerMemory::default(),
                &[(schedule, None)],
                Layout::Unscoped,
                None,
                &Resolvable::of(&[])
            )[0][3],
            "due"
        );

        let mut soon = schedule_with(None);
        soon.next_fire_at = chrono::Utc::now() + chrono::Duration::minutes(90);
        assert_eq!(
            rows_for(
                &crate::schedule::SchedulerMemory::default(),
                &[(soon, None)],
                Layout::Unscoped,
                None,
                &Resolvable::of(&[])
            )[0][3],
            "1h 29m"
        );
    }
}
