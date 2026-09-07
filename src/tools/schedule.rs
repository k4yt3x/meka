//! The `schedule_*` tools: the agent arranging its own future turns ([`crate::schedule`]).
//!
//! Creating and listing gate at [`Permission::Read`], matching `memory_*` and `scratchpad_*`: a
//! scheduled prompt writes to a store meka owns, and the turn it eventually produces is permission
//! -checked when it runs, so scheduling one grants nothing the session did not already have.
//!
//! The `gate` field is the exception, and is checked inside [`ScheduleCreateTool::execute`]
//! against [`crate::schedule::gate_probe_is_authorized`]. A gate runs unattended, on a timer, until
//! someone cancels it -- persistent in a way a tool call inside a turn is not, since that at least
//! ends with the turn that made it. The bar depends on what the gate runs: a shell command needs
//! `unrestricted` because it is unsandboxed, while a read-only tool call needs only what that tool
//! needs. [`Tool::required_permission`] is per-tool and cannot vary by argument, so the check has
//! to live in the body either way.
//!
//! Sub-agents deliberately get none of these tools. A sub-agent's session is ephemeral, so a job
//! keyed to it would outlive the only conversation that could run it.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use super::{
    Tool, ToolOutput,
    util::{require_str, resolve_session_id},
};
use crate::{
    config::ResolvedScheduleConfig,
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
    schedule::{Gate, GatePredicate, GateProbe, Schedule, ScheduledJob},
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
/// and the relative form is what catches a schedule that parsed successfully but means something
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
            description: "Arrange for a prompt to be delivered to you at a future time, so you can \
                act without the user asking again. Use for reminders (\"remind me in 20 minutes\"), \
                recurring work (\"summarize my calendar every weekday morning\"), and watching \
                something change. Give exactly one of `at`, `every`, or `cron`. The prompt is \
                delivered as a turn with no human present, so write it as an instruction to \
                yourself, including any context you will need and no longer have."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What to do when the job fires. Self-contained: the \
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
                        "description": "Recurring 5-field cron expression in local time \
                                        ('0 9 * * 1-5' = 09:00 on weekdays). No seconds field."
                    },
                    "gate": {
                        "type": "object",
                        "description": "Optional. Check something first and only take a turn if it \
                                        says something happened. Turns an expensive poll into a \
                                        cheap one, so a short `every` becomes affordable.",
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
                                                \"not-empty\"|\"empty\"|\"changed\"} judges one \
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
        let schedule = parse_schedule(&input, now, tool_name)?;
        let gate = self.parse_gate(&input, tool_name)?;

        let existing = self
            .store
            .schedule_store()
            .list_scheduled_jobs(session_id)
            .await?
            .len();
        if existing >= self.config.max_jobs {
            return Err(MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: format!(
                    "this session already has {existing} scheduled jobs (the limit). Cancel one with \
                     schedule_cancel first."
                ),
            });
        }

        let next_fire_at = schedule
            .next_after(now)
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: "that schedule has no next occurrence. A one-shot time must be in the \
                          future."
                    .to_string(),
            })?;

        let job = ScheduledJob {
            id: Uuid::new_v4().to_string(),
            session_id,
            schedule,
            prompt: prompt.to_string(),
            gate,
            created_at: now,
            last_fired_at: None,
            next_fire_at,
            attempts: 0,
        };
        self.store
            .schedule_store()
            .create_scheduled_job(&job)
            .await?;

        let short_id = job.short_id();
        let schedule = job.schedule.describe();
        let next_fire = next_fire_at.to_rfc3339();
        tracing::info!("scheduled job {short_id} ({schedule}), next fire {next_fire}");

        let mut summary = format!(
            "Created job {} ({}). Next fire: {}.",
            job.short_id(),
            job.schedule.describe(),
            describe_fire_time(next_fire_at)
        );
        if job.gate.is_some() {
            summary.push_str(" Gated: a turn happens only when the gate says so.");
        }
        summary.push_str(&format!(
            " Cancel with schedule_cancel(\"{}\").",
            job.short_id()
        ));
        Ok(ToolOutput::text(summary, false))
    }
}

impl ScheduleCreateTool {
    fn parse_gate(&self, input: &serde_json::Value, tool_name: &str) -> Result<Option<Gate>> {
        let Some(raw) = input.get("gate").filter(|value| !value.is_null()) else {
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
                "{}. Create the job without `gate` and check the condition inside the prompt \
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

/// Pull exactly one of `at` / `every` / `cron` out of the tool input.
fn parse_schedule(
    input: &serde_json::Value,
    now: chrono::DateTime<Utc>,
    tool_name: &str,
) -> Result<Schedule> {
    let field = |key: &str| {
        input
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let given: Vec<(&str, &str)> = ["at", "every", "cron"]
        .into_iter()
        .filter_map(|key| field(key).map(|value| (key, value)))
        .collect();

    let (kind, value) = match given.as_slice() {
        [single] => *single,
        [] => {
            return Err(MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: "give one of 'at' (once), 'every' (interval), or 'cron' (expression)"
                    .to_string(),
            });
        }
        // Ambiguity is refused rather than resolved by precedence: silently honoring one and
        // dropping the other would produce a job that fires on a schedule nobody asked for.
        several => {
            return Err(MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: format!(
                    "give exactly one schedule, got {}",
                    several
                        .iter()
                        .map(|(key, _)| *key)
                        .collect::<Vec<_>>()
                        .join(" and ")
                ),
            });
        }
    };

    let parsed = match kind {
        "at" => Schedule::parse_at(value, now),
        "every" => Schedule::parse_every(value),
        _ => Schedule::parse_cron(value),
    };
    parsed.map_err(|message| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message,
    })
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
            description: "List the scheduled jobs for this session, with their next fire times, \
                gates, and full prompts. Your per-turn context already carries a short index, so \
                reach for this when you need a job's exact prompt or gate check."
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
        let jobs = self
            .context
            .store
            .schedule_store()
            .list_scheduled_jobs(session_id)
            .await?;
        if jobs.is_empty() {
            return Ok(ToolOutput::text(
                "No scheduled jobs in this session.".to_string(),
                false,
            ));
        }

        let mut rendered = String::new();
        for job in &jobs {
            rendered.push_str(&format!(
                "{} ({})\n  next: {}\n",
                job.short_id(),
                job.schedule.describe(),
                describe_fire_time(job.next_fire_at)
            ));
            if let Some(fired) = job.last_fired_at {
                rendered.push_str(&format!(
                    "  last fired: {}\n",
                    crate::text::format_timestamp(fired, crate::text::Precision::Minutes)
                ));
            }
            if let Some(gate) = &job.gate {
                // `detail`, not `summary`: this is the surface the model authored the job on, and
                // it cannot otherwise read back the arguments it wrote.
                rendered.push_str(&format!(
                    "  gate ({}): {}\n",
                    gate.predicate.summary(),
                    gate.probe.detail()
                ));
            }
            // A job that cannot fire is the difference between one quietly waiting and one that is
            // dead, and the two look identical without this line: both simply never fire, and
            // `last fired` is absent for a brand-new job too. Outside the `gate` branch because an
            // ungated job on a session at `none` is held as well. Reported to the model because it
            // can act on it -- `schedule_cancel` needs only `read` -- where until now the only
            // trace was a `warn!` in the operator's log.
            if let Some(reason) = crate::schedule::job_withheld_reason(
                self.context.store.scheduler_memory(),
                job,
                self.context.site.permission.get(),
                self.gate_tools.as_deref(),
            ) {
                rendered.push_str(&format!("  NOT FIRING: {reason}\n"));
            }
            rendered.push_str(&format!("  prompt: {}\n", job.prompt));
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
            description: "Cancel a scheduled job so it stops firing. Takes the id from \
                schedule_create or schedule_list; a unique prefix is enough."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Job id, or any unique prefix of one."
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
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let tool_name = "schedule_cancel";
        let session_id = resolve_session_id(&self.context.site.session_id, tool_name)?;
        let id = require_str(&input, "id", tool_name)?;

        match self
            .context
            .store
            .schedule_store()
            .cancel_scheduled_job(session_id, &id)
            .await?
        {
            Some(canceled) => {
                tracing::info!("canceled scheduled job {canceled}");
                Ok(ToolOutput::text(
                    format!("Canceled job {}.", &canceled[..8.min(canceled.len())]),
                    false,
                ))
            }
            None => Ok(ToolOutput::text(
                format!(
                    "No scheduled job matching '{id}' in this session. Use schedule_list to see \
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("creates");
        let text = output.text_content();
        assert!(text.contains("Next fire:"), "{text}");
        // The relative half is the part that catches a schedule meaning something other than
        // intended, so it must actually be present.
        assert!(text.contains("in 19m"), "{text}");
    }

    #[tokio::test]
    async fn create_requires_exactly_one_schedule() {
        let (tool, _session_id, _manager) = harness().await;
        let none = tool
            .execute(
                serde_json::json!({"prompt": "x"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("no schedule is an error");
        assert!(none.to_string().contains("one of"), "{none}");

        let both = tool
            .execute(
                serde_json::json!({"prompt": "x", "at": "20m", "every": "1h"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("two schedules is an error");
        assert!(both.to_string().contains("exactly one"), "{both}");
    }

    /// The permission rule that cannot live in `required_permission`, since that is per-tool.
    #[tokio::test]
    async fn a_gate_is_refused_below_write_but_the_reminder_is_not() {
        // Every level below `unrestricted`, not just `read`.
        //
        // `workspace` is the one that matters: letting it *pass* this door is a one-call escape,
        // since `schedule_create` with a gate runs arbitrary unconfined commands from inside the
        // confined level within one poll interval. Exercising only `read` left it unguarded at the
        // tool door.
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a gate needs an unattended-write level");
        assert!(
            refused.to_string().contains("`unrestricted`"),
            "the refusal at {level} must name the level a gate needs: {refused}"
        );

        // The same job without a gate is fine at read: scheduling a prompt grants nothing, since
        // the turn it produces is permission-checked when it runs.
        tool.execute(
            serde_json::json!({"prompt": "x", "every": "1h"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .unwrap_or_else(|error| panic!("an ungated reminder is allowed at {level}: {error}"));
    }

    #[tokio::test]
    async fn create_enforces_the_job_ceiling() {
        let (mut tool, _session_id, _manager) = harness().await;
        tool.config = ResolvedScheduleConfig {
            max_jobs: 1,
            ..ResolvedScheduleConfig::default()
        };
        tool.execute(
            serde_json::json!({"prompt": "first", "every": "1h"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("first fits");
        let error = tool
            .execute(
                serde_json::json!({"prompt": "second", "every": "1h"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a past instant has no next occurrence");
        assert!(error.to_string().contains("future"), "{error}");
    }

    /// The model can read back the arguments it wrote into a tool gate.
    ///
    /// `schedule_list` is the only surface that shows them: the HTTP view and `meka schedule list`
    /// are read by parties who did not author the job, and arguments are where a pasted credential
    /// would sit. Without this the model could not tell a gate it built with the wrong `folder`
    /// from a correct one.
    #[tokio::test]
    async fn list_shows_a_tool_gate_s_arguments_to_the_model_that_wrote_them() {
        let (_tool, session_id, manager) = harness().await;
        // Planted through the store rather than `schedule_create`: the creation door needs a live
        // tool dispatcher to authorize a tool gate, and what is under test here is the rendering.
        let id = session_id.get().expect("harness made a session");
        let schedule = Schedule::parse_every("1h").expect("parses");
        let now = Utc::now();
        manager
            .schedule_store()
            .create_scheduled_job(&ScheduledJob {
                attempts: 0,
                id: uuid::Uuid::new_v4().to_string(),
                session_id: id,
                schedule: schedule.clone(),
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
                next_fire_at: schedule.next_after(now).expect("has a next fire"),
            })
            .await
            .expect("plants the job");

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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
    /// job is dead belongs. Showing the gate's *definition* and stopping there reads as healthy
    /// however long the gate has been refused.
    #[tokio::test]
    async fn list_reports_a_gate_that_cannot_currently_fire() {
        let (tool, session_id, manager) = harness().await;
        tool.execute(
            serde_json::json!({
                "prompt": "watch the build",
                "every": "1h",
                "gate": {"check": {"command": "gh pr checks"}, "when": "changed"}
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
            crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("cancels")
            .text_content();
        assert!(canceled.contains("Canceled job"), "{canceled}");

        let after = list
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("lists")
            .text_content();
        assert!(after.contains("No scheduled jobs"), "{after}");
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a miss is not an error")
            .text_content();
        assert!(text.contains("No scheduled job matching"), "{text}");
    }
}
