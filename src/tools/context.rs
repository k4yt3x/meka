//! The `context_*` tools: the agent managing its own context window.
//!
//! A separate family from `conversation_*`, which reads the archive (the full on-disk log,
//! including turns compaction removed from the window entirely); these three act on the live
//! window. The two families sort adjacently (`cont` precedes `conv`), so the split costs nothing
//! in the catalog.
//!
//! `context_check` exists because the pushed `[Context budget]` block
//! ([`crate::prompt::ContextBudget`]) is rendered once per turn, into the user message at turn
//! start. The counter behind it moves on every provider response including mid-tool-loop, but the
//! rendered text does not, so the gauge is stalest exactly when a tool loop is ingesting large
//! results. Refreshing the block in place would rewrite a message the cached prefix already covers
//! and invalidate it on every iteration; a tool result appends at the tail and is cache-safe by
//! construction.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use async_trait::async_trait;

use super::{Tool, ToolOutput, util::resolve_session_id};
use crate::{
    error::Result,
    permission::Permission,
    provider::ToolDefinition,
    session::{CompactOrigin, CompactRequest, PendingCompaction, compaction_tail_budget},
    store::Store,
};

/// The summary a checkpoint turn submitted, and what it decided about the tail.
pub(crate) struct Submission {
    pub(crate) summary: String,
    pub(crate) keep_recent: Option<bool>,
}

/// Slot `context_replace` writes into. Owned by the checkpoint turn that registered the tool, so a
/// fresh one is created per compaction and never outlives it.
pub(crate) type SubmissionSlot = Arc<std::sync::Mutex<Option<Submission>>>;

/// Live numbers `context_check` reports, kept current by the agent.
#[derive(Clone)]
pub(crate) struct ContextGauge {
    /// Total tokens behind the most recent provider round: the same handle
    /// `Agent::record_context_tokens` writes after every response, so this moves within a turn
    /// rather than only between turns.
    pub(crate) used: Arc<AtomicU64>,
    /// Estimated system prompt + tool schemas, re-stamped by the agent each turn. Separate from
    /// `used` because it is the part compaction *cannot* reclaim, which is what makes it worth
    /// reporting.
    pub(crate) overhead: Arc<AtomicU64>,
    /// The model's window, or zero when meka has no metadata for it.
    ///
    /// A handle rather than a value because a session can move onto another profile mid-run, and a
    /// window frozen when the tools were registered would report the previous profile's size to
    /// the model for the rest of the session. Written by `Agent::set_provider` through
    /// [`crate::provider::PublishedProfile`].
    pub(crate) window: Arc<AtomicU64>,
    /// Tokens whole reads have charged for what they returned since `used` was measured; the same
    /// cell as `SessionCells::context_reserved`, cleared with every measurement.
    pub(crate) reserved: Arc<AtomicU64>,
    /// The share of the window the conversation may fill on its own
    /// (`[session].context_ceiling_percent`): where a whole read stops, and where auto-compaction
    /// fires when it is on.
    pub(crate) ceiling_percent: u64,
    /// Whether auto-compaction fires past the ceiling. Reported, never consulted for the line
    /// itself: the switch decides what reclaims the space, not how far a read may go.
    pub(crate) auto_compact: bool,
}

impl ContextGauge {
    /// The gauge over a session's own cells, so every reader of it and the agent agree.
    pub(crate) fn new(
        cells: &crate::session::SessionCells,
        ceiling_percent: u64,
        auto_compact: bool,
    ) -> Self {
        Self {
            used: Arc::clone(&cells.context_tokens),
            overhead: Arc::clone(&cells.context_overhead),
            window: cells.profile.window(),
            reserved: Arc::clone(&cells.context_reserved),
            ceiling_percent,
            auto_compact,
        }
    }

    /// The occupancy a read must stay under, in tokens.
    fn ceiling(&self, window: u64) -> u64 {
        crate::session::context_ceiling(window, self.ceiling_percent)
    }

    /// Tokens left under the ceiling now: what was measured plus what has been reserved since.
    /// `None` when the window is unknown, which is not a small window but no window at all.
    pub(crate) fn headroom(&self) -> Option<u64> {
        (self.window.load(Ordering::Relaxed) > 0)
            .then(|| self.headroom_with(self.reserved.load(Ordering::Relaxed)))
    }

    /// Reserve room for `text` and return how many of its bytes may go into context: all of it
    /// where its token bound fits under the ceiling, the longest prefix that does where it does
    /// not, and at least the inline bound, granted and charged like the rest so a read is never a
    /// worse tool than any other; the clamp governs only the excess. `None` when the window is
    /// unknown: nothing can be sized against it, so the caller leaves the result for the spill to
    /// bound as it bounds every other tool's.
    ///
    /// Sized by `tokens::bound_text` rather than by bytes, because this is the one place an
    /// estimate is acted on before a measurement can correct it, and a byte count under-reads
    /// digits, symbols and non-ASCII text by two to three times. The grant is charged to the
    /// reservation before it is returned, so two reads in one round share the headroom rather
    /// than each taking all of it.
    pub(crate) fn reserve(&self, text: &str) -> Option<usize> {
        if self.window.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let floor =
            text.floor_char_boundary(super::scratchpad::MAX_INLINE_RESULT_BYTES.min(text.len()));
        let mut reserved = self.reserved.load(Ordering::Relaxed);
        loop {
            let headroom = self.headroom_with(reserved);
            let granted = crate::tokens::prefix_within(text, headroom).max(floor);
            let charged = reserved.saturating_add(crate::tokens::bound_text(&text[..granted]));
            match self.reserved.compare_exchange(
                reserved,
                charged,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(granted),
                Err(current) => reserved = current,
            }
        }
    }

    /// [`Self::headroom`] against a reservation the caller read, which is what the compare-and-swap
    /// in [`Self::reserve`] needs; a zero window reads as no room, and only callers that checked
    /// the window take that as a number.
    fn headroom_with(&self, reserved: u64) -> u64 {
        let window = self.window.load(Ordering::Relaxed);
        self.ceiling(window)
            .saturating_sub(self.used.load(Ordering::Relaxed).saturating_add(reserved))
    }

    /// A gauge over nothing: no window, so nothing is sized, nothing is reserved, and a read's
    /// reply is left for the spill.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self {
            used: Arc::new(AtomicU64::new(0)),
            overhead: Arc::new(AtomicU64::new(0)),
            window: Arc::new(AtomicU64::new(0)),
            reserved: Arc::new(AtomicU64::new(0)),
            ceiling_percent: crate::config::DEFAULT_CONTEXT_CEILING_PERCENT,
            auto_compact: false,
        }
    }
}

pub(super) struct ContextCheckTool {
    pub(crate) gauge: ContextGauge,
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
}

#[async_trait]
impl Tool for ContextCheckTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "context_check".to_string(),
            description: "Measure your context window now. Unlike the `[Context budget]` line, \
                          which is fixed at the start of the turn, this is live. Call it before \
                          reading a large file, starting a long stretch of tool calls, or \
                          deciding whether a task fits."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
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
        let used = self.gauge.used.load(Ordering::Relaxed);
        let overhead = self.gauge.overhead.load(Ordering::Relaxed);
        let window = self.gauge.window.load(Ordering::Relaxed);

        let mut report = String::new();

        // `checked_div` is the zero-window test: without a denominator there is no occupancy to
        // report, and a percentage of an unknown total is worse than silence. Same call
        // `ContextBudget::render` makes.
        match used.saturating_mul(100).checked_div(window) {
            None => report.push_str(
                "Context window: unknown for this model, so occupancy cannot be reported.\n",
            ),
            Some(percent) => {
                report.push_str(&format!("Using {used} of {window} tokens ({percent}%)"));
                // Named beside the measurement, or the headroom below would not add up to it.
                let reserved = self.gauge.reserved.load(Ordering::Relaxed);
                if reserved > 0 {
                    report.push_str(&format!(", plus {reserved} reserved by this round's reads"));
                }
                report.push_str(".\n");
                // The same figure a whole `scratchpad_read` is sized against, so what this
                // reports as room is what a read may take.
                let headroom = self.gauge.headroom().unwrap_or(0);
                report.push_str(&format!(
                    "Headroom: {headroom} tokens before the context ceiling at {}%. ",
                    self.gauge.ceiling_percent
                ));
                report.push_str(if self.gauge.auto_compact {
                    "Auto-compaction fires there, between turns or between two of your tool \
                     rounds.\n"
                } else {
                    "Auto-compaction is off, so nothing reclaims the space past it and a request \
                     past the window fails the turn.\n"
                });
                report.push_str(&format!(
                    "Kept verbatim on compaction: about {} tokens of the most recent turns; \
                     everything older is replaced by a summary.\n",
                    compaction_tail_budget(window)
                ));
            }
        }

        if overhead > 0 {
            report.push_str(&format!(
                "Fixed overhead: about {overhead} tokens of system prompt and tool schemas (estimated). \
                 Compaction does not reclaim this.\n"
            ));
            if used > overhead {
                report.push_str(&format!(
                    "Conversation: about {} tokens, which is the part compaction acts on.\n",
                    used.saturating_sub(overhead)
                ));
            }
        }

        // Best-effort: a session that has not been created yet, or a read that fails, should not
        // fail the whole call over a line that is context rather than the answer.
        if let Ok(session_id) = resolve_session_id(&self.site.session_id, "context_check")
            && let Ok(generation) = self.store.count_compactions(session_id).await
        {
            report.push_str(&match generation {
                0 => "Compactions so far: none, so nothing has been summarized away yet.\n".into(),
                1 => "Compactions so far: 1. Detail from before it survives only as a summary; \
                      `conversation_search` reaches the original turns.\n"
                    .to_string(),
                count => format!(
                    "Compactions so far: {count}. Each one summarizes the previous summary, so early \
                     detail is now several removes from the original; write anything that must \
                     last to memory rather than trusting it to survive another pass.\n"
                ),
            });
        }

        Ok(ToolOutput::text(report, false))
    }
}

pub(super) struct ContextCompactTool {
    pub(crate) pending: PendingCompaction,
    /// Whether a checkpoint turn will actually run (`[session].compact_checkpoint`).
    ///
    /// Carried so this tool can tell the truth about what happens next. The difference is not
    /// cosmetic: with a checkpoint the agent gets a chance to save durable notes *after* asking to
    /// compact, so it can reasonably defer that work; without one the summary is written by a
    /// separate call with no tools, and anything not already saved is simply gone. An agent told
    /// it would get a checkpoint that never comes would skip the one action that mattered.
    pub(crate) checkpoint_enabled: bool,
}

#[async_trait]
impl Tool for ContextCompactTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "context_compact".to_string(),
            // Branched for the same reason the result below is: this is the only text the model
            // reads *before* it decides. Told it will get a checkpoint on an installation that
            // has none, it defers the one action that had to happen first, and the deferral is
            // unrecoverable because the summary is written without it.
            description: if self.checkpoint_enabled {
                "Compact this conversation before your next step. Earlier turns become a summary \
                 you write, after a checkpoint for saving anything that must outlive them, and \
                 this turn then carries on against it. Use it when a stretch of work is done \
                 rather than waiting for auto-compaction mid-task. `conversation_search` still \
                 reaches the full history."
                    .to_string()
            } else {
                "Compact this conversation before your next step. Earlier turns become a summary \
                 written without you, and this turn then carries on against it. There is no \
                 checkpoint on this installation, so save anything that must outlive this \
                 conversation to memory in the same batch as this call: afterwards is too late. \
                 Use it when a stretch of work is done rather than waiting for auto-compaction \
                 mid-task. `conversation_search` still reaches the full history."
                    .to_string()
            },
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": "What to preserve or drop, e.g. \"keep the design decisions, drop the debugging\". Takes precedence over the default summary sections."
                    },
                    "keep_recent": {
                        "type": "boolean",
                        "default": true,
                        "description": "Whether to keep the most recent turns verbatim after the summary. Default: true. Set false to start clean, only when the summary and what you have saved cover everything, such as when closing out a day's work."
                    }
                },
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
        let request = CompactRequest {
            origin: CompactOrigin::Requested,
            instructions: input["instructions"]
                .as_str()
                .map(str::trim)
                .filter(|instructions| !instructions.is_empty())
                .map(str::to_string),
            keep_recent: input["keep_recent"].as_bool(),
            prompt_id: context.prompt_id,
            request_in_flight: None,
        };
        let mut pending = crate::sync::lock(&self.pending);
        // Last call wins rather than first, within the batch the loop drains as a unit: a model
        // that asks twice before the drain most likely refined what it wanted, and keeping the
        // first would silently apply the stale instructions. Across batches the decision is not
        // here but at the drain, which acts on one request per turn and drops the rest.
        *pending = Some(request);
        drop(pending);
        Ok(ToolOutput::text(
            if self.checkpoint_enabled {
                "Compaction runs once this batch of tool calls finishes, and then this turn \
                 continues against the summary. You will get a checkpoint first, to save anything \
                 durable. One compaction per turn; to compact again, ask on a later turn."
                    .to_string()
            } else {
                "Compaction runs once this batch of tool calls finishes, and then this turn \
                 continues against the summary. There is no checkpoint on this installation, so \
                 the summary is written without you and anything not already in memory is gone \
                 from your context. One compaction per turn; to compact again, ask on a \
                 later turn."
                    .to_string()
            },
            false,
        ))
    }
}

/// The checkpoint turn's terminal call. Registered only for that turn, so it never appears in the
/// ordinary catalog and is deliberately absent from `BUILTIN_TOOL_NAMES`: listing it there would
/// let a `disabled_tools` entry silently downgrade every compaction to the fallback summarizer.
pub(super) struct ContextReplaceTool {
    pub(crate) slot: SubmissionSlot,
}

#[async_trait]
impl Tool for ContextReplaceTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "context_replace".to_string(),
            description: "Replace your context with the summary you have written. Call this once, \
                          last, to end the checkpoint. Everything before the kept tail is \
                          discarded from your window, so the summary has to carry whatever the \
                          work still depends on."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The text that will replace the earlier turns. Write it for yourself, in your own voice, as the record you would want to pick the work back up from."
                    },
                    "keep_recent": {
                        "type": "boolean",
                        "default": true,
                        "description": "Whether to keep the most recent turns verbatim after the summary. Default: true. Set false only when your summary and what you have saved fully cover them."
                    }
                },
                "required": ["summary"]
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
        let summary = input["summary"].as_str().unwrap_or_default().trim();
        if summary.is_empty() {
            // A tool error rather than a hard failure: the loop gets another iteration to try
            // again, and if it never does, the fallback ladder catches it.
            return Ok(ToolOutput::text(
                "The 'summary' parameter is required and cannot be empty. It becomes your entire \
                 context, so an empty one would erase the conversation."
                    .to_string(),
                true,
            ));
        }
        let mut slot = crate::sync::lock(&self.slot);
        *slot = Some(Submission {
            summary: summary.to_string(),
            keep_recent: input["keep_recent"].as_bool(),
        });
        drop(slot);
        Ok(ToolOutput::text("Checkpoint accepted.".to_string(), false))
    }
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;

    fn gauge(used: u64, overhead: u64, window: u64, auto_compact: bool) -> ContextGauge {
        ContextGauge {
            used: Arc::new(AtomicU64::new(used)),
            overhead: Arc::new(AtomicU64::new(overhead)),
            window: Arc::new(AtomicU64::new(window)),
            reserved: Arc::new(AtomicU64::new(0)),
            ceiling_percent: 80,
            auto_compact,
        }
    }

    async fn check(gauge: ContextGauge) -> String {
        let tool = ContextCheckTool {
            gauge,
            store: Store::for_test().await,
            site: crate::session::ToolSite::for_test()
                .with_session_id(crate::session::SharedSessionId::default()),
        };
        let output = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("context_check");
        output.text_content()
    }

    #[tokio::test]
    async fn reports_headroom_to_the_ceiling() {
        let report = check(gauge(40_000, 5_000, 200_000, true)).await;
        assert!(report.contains("40000 of 200000 tokens (20%)"), "{report}");
        // 80% of 200k is 160k, so 120k of headroom is left.
        assert!(report.contains("Headroom: 120000 tokens"), "{report}");
        assert!(report.contains("Auto-compaction fires there"), "{report}");
        assert!(
            report.contains("about 5000 tokens of system prompt"),
            "{report}"
        );
        assert!(
            report.contains("Conversation: about 35000 tokens"),
            "{report}"
        );
    }

    /// The switch changes what happens past the line, not where the line is: a read is still cut
    /// there, so the headroom reported is the same.
    #[tokio::test]
    async fn reports_headroom_to_the_ceiling_when_auto_compaction_is_off() {
        let report = check(gauge(40_000, 0, 200_000, false)).await;
        assert!(report.contains("Headroom: 120000 tokens"), "{report}");
        assert!(report.contains("Auto-compaction is off"), "{report}");
    }

    /// An unknown window must not divide by zero, and must not invent a percentage either.
    #[tokio::test]
    async fn suppresses_occupancy_when_the_window_is_unknown() {
        let report = check(gauge(40_000, 0, 0, true)).await;
        assert!(report.contains("unknown for this model"), "{report}");
        assert!(!report.contains('%'), "{report}");
    }

    /// The room a read may take is what is left under the ceiling after what this round has
    /// already reserved, and the grant is charged before it is returned, so two reads issued
    /// together split the headroom instead of each taking all of it.
    #[test]
    fn a_reservation_takes_from_the_headroom_and_never_less_than_the_inline_bound() {
        let gauge = gauge(100_000, 0, 200_000, true);
        assert_eq!(gauge.headroom(), Some(60_000));

        // Digits bound at one token each, so the grant reads in bytes: 60k of a 250k ask.
        let digits = "7".repeat(250_000);
        assert_eq!(gauge.reserve(&digits), Some(60_000));
        assert_eq!(gauge.headroom(), Some(0));
        // Spent, so the floor is all that is left, and it is still charged.
        assert_eq!(
            gauge.reserve(&digits),
            Some(super::super::scratchpad::MAX_INLINE_RESULT_BYTES)
        );
        assert_eq!(gauge.reserved.load(Ordering::Relaxed), 90_000);

        // A small ask is granted whole and charged for what it took: four letters, one token.
        let gauge = self::gauge(0, 0, 200_000, true);
        assert_eq!(gauge.reserve("abcd"), Some(4));
        assert_eq!(gauge.headroom(), Some(159_999));

        // Letters merge, so the same headroom holds five times the bytes of prose than of digits.
        let gauge = self::gauge(100_000, 0, 200_000, true);
        assert_eq!(gauge.reserve(&"x".repeat(400_000)), Some(300_000));
    }

    /// The switch decides what reclaims the space past the ceiling, not how far a read may go:
    /// with it off the read stops at the same line rather than running to the window's edge,
    /// where nothing would be left for the reply.
    #[test]
    fn a_read_stops_at_the_ceiling_whether_or_not_compaction_is_on() {
        let digits = "7".repeat(100_000);
        let on = gauge(100_000, 0, 200_000, true);
        let off = gauge(100_000, 0, 200_000, false);
        assert_eq!(on.headroom(), Some(60_000));
        assert_eq!(off.headroom(), Some(60_000));
        assert_eq!(on.reserve(&digits), Some(60_000));
        assert_eq!(off.reserve(&digits), Some(60_000));
    }

    /// No window is no bound: nothing is sized and nothing is charged, since a percentage of an
    /// unknown total is not a small number but no number, and the caller is told so rather than
    /// handed the whole text as if it had been sized.
    #[test]
    fn an_unknown_window_sizes_no_read() {
        let gauge = gauge(100_000, 0, 0, true);
        assert_eq!(gauge.headroom(), None);
        assert_eq!(gauge.reserve(&"7".repeat(500_000)), None);
        assert_eq!(gauge.reserved.load(Ordering::Relaxed), 0);
    }

    /// What `context_check` calls headroom is the same figure a read is sized against, so a
    /// model that checks and then reads sees the reservation its earlier read made.
    #[tokio::test]
    async fn reports_headroom_net_of_this_rounds_reads() {
        let gauge = gauge(40_000, 0, 200_000, true);
        gauge.reserve(&"7".repeat(40_000));
        let report = check(gauge).await;
        assert!(report.contains("Headroom: 80000 tokens"), "{report}");
        assert!(
            report.contains("plus 40000 reserved by this round's reads"),
            "{report}"
        );
    }

    #[tokio::test]
    async fn compact_requests_carry_instructions_and_the_tail_decision() {
        let pending: PendingCompaction = Arc::new(std::sync::Mutex::new(None));
        let tool = ContextCompactTool {
            pending: Arc::clone(&pending),
            checkpoint_enabled: true,
        };
        tool.execute(
            serde_json::json!({"instructions": "keep the decisions", "keep_recent": false}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("context_compact");

        let request = pending.lock().expect("lock").take().expect("recorded");
        assert_eq!(request.origin, CompactOrigin::Requested);
        assert_eq!(request.instructions.as_deref(), Some("keep the decisions"));
        assert_eq!(request.keep_recent, Some(false));
    }

    /// An empty summary would erase the conversation outright, so it has to be refused in a way the
    /// model can recover from rather than accepted.
    #[tokio::test]
    async fn replace_refuses_an_empty_summary() {
        let slot: SubmissionSlot = Arc::new(std::sync::Mutex::new(None));
        let tool = ContextReplaceTool {
            slot: Arc::clone(&slot),
        };
        let output = tool
            .execute(
                serde_json::json!({"summary": "   "}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("context_replace");
        assert!(
            output.is_error,
            "an empty summary must be refused as an error the model can act on"
        );
        assert!(slot.lock().expect("lock").is_none());
    }

    #[tokio::test]
    async fn replace_records_the_summary_and_tail_decision() {
        let slot: SubmissionSlot = Arc::new(std::sync::Mutex::new(None));
        let tool = ContextReplaceTool {
            slot: Arc::clone(&slot),
        };
        tool.execute(
            serde_json::json!({"summary": "what happened", "keep_recent": false}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("context_replace");

        let submission = slot.lock().expect("lock").take().expect("recorded");
        assert_eq!(submission.summary, "what happened");
        assert_eq!(submission.keep_recent, Some(false));
    }
}
