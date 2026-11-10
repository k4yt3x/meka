//! [`Conversation`]: append-only-by-default newtype for the agent's conversation.
//!
//! Built on an event log: each mutation pushes one or more [`Event`]s, and the materialized
//! `&[Message]` view consumed by providers and the scanner is derived from those events. Every
//! destructive operation ([`Conversation::pop_unsaved`], [`Conversation::replace_for_compaction`],
//! [`Conversation::replace_tail`], [`Conversation::pop_repair`], [`Conversation::rewind`],
//! [`Conversation::sanitize_orphans`]) remains an explicit, named method; the compiler refuses
//! casual mutation. The one exception is [`repair_invalid_images`], which runs on every
//! materialization because a rebuild must not be able to reinstate content the provider refuses.
//!
//! On disk, events are stored row-per-event in the `messages` table; the encoding lives in
//! `store::sessions`'s `encode_event_for_db` / `decode_event_from_row` helpers, behind the
//! [`crate::store::Store::save_event`] / [`crate::store::Store::load_events`] API.

use serde::{Deserialize, Serialize};

use crate::image::ImageSource;

/// Whether `message` is the one a turn opens with, rather than something appended inside one.
///
/// A tool round trip persists its results as a `User` message too, so role alone does not separate
/// "somebody asked for something" from "the loop is still running". Shared by
/// [`Conversation::rewind`], which counts turns backwards, and
/// [`Conversation::ends_on_a_turn_opening`], which asks whether a turn produced anything at all:
/// two callers that must agree on where a turn begins.
fn opens_turn(message: &Message) -> bool {
    message.opens_turn()
}

/// What the loop writes to coax a visible reply out of a turn that produced only thinking, or
/// nothing: the text of a [`NudgeKind::VisibleReply`] nudge, kept beside the kind, below every
/// reader that builds one.
pub(crate) const THINKING_ONLY_NUDGE: &str = "[Your previous response contained no visible output. Please \
                                   continue and produce a user-visible response.]";

/// Which strategy wrote a compaction's summary. The two differ in fidelity, not just mechanism:
/// the checkpoint turn is the agent summarizing itself, with its tools, and the standalone
/// summarizer knows nothing but the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum CompactionSource {
    /// The checkpoint turn, via `context_replace`. The intended path.
    Checkpoint,
    /// The standalone summarizer: the emergency path, a disabled checkpoint, or a checkpoint that
    /// ended without calling `context_replace`. The last is reachable on every backend, since
    /// `Provider::complete` carries no `tool_choice` and the call cannot be forced.
    Summarizer,
}

impl CompactionSource {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [CompactionSource; 2] = [Self::Checkpoint, Self::Summarizer];

    /// The value's one spelling, in the log and on every wire.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Checkpoint => "checkpoint",
            Self::Summarizer => "summarizer",
        }
    }
}

impl std::fmt::Display for CompactionSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for CompactionSource {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, String> {
        Self::ALL
            .iter()
            .copied()
            .find(|source| source.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a compaction source. Supported: {}",
                    Self::ALL
                        .iter()
                        .map(|source| source.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
}

impl TryFrom<String> for CompactionSource {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, String> {
        value.parse()
    }
}

impl From<CompactionSource> for String {
    fn from(source: CompactionSource) -> Self {
        source.name().to_string()
    }
}

/// What set a compaction going. Selects the summarization strategy, so it is not merely
/// diagnostic, and is what every surface says about the compaction afterwards.
///
/// Every trigger but [`Self::Emergency`] can afford the checkpoint turn. `Emergency` cannot: it
/// runs *after* the provider rejected the request for exceeding the window, and a checkpoint turn
/// sends the same conversation again, so it would be refused for the same reason. That path needs
/// a call that is deliberately smaller than the one that just failed, which is exactly what
/// `Agent::summarize_via_provider` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum CompactionTrigger {
    /// A measured occupancy crossed the ceiling, between turns or between two tool rounds.
    Reactive,
    /// This turn's projected request would cross it.
    Proactive,
    /// A human ran `/compact` or `POST /compact`.
    Manual,
    /// The agent asked, via `context_compact`.
    Requested,
    /// The provider refused the request as too large.
    Emergency,
    /// A turn in the model's view is older than `[session].turn_retention`, so the sweep that
    /// deletes it needs a summary to stand in for it first.
    Retention,
}

impl CompactionTrigger {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [CompactionTrigger; 6] = [
        Self::Reactive,
        Self::Proactive,
        Self::Manual,
        Self::Requested,
        Self::Emergency,
        Self::Retention,
    ];

    /// The value's one spelling, in the log and on every wire.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Reactive => "reactive",
            Self::Proactive => "proactive",
            Self::Manual => "manual",
            Self::Requested => "requested",
            Self::Emergency => "emergency",
            Self::Retention => "retention",
        }
    }
}

impl std::fmt::Display for CompactionTrigger {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for CompactionTrigger {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, String> {
        Self::ALL
            .iter()
            .copied()
            .find(|trigger| trigger.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a compaction trigger. Supported: {}",
                    Self::ALL
                        .iter()
                        .map(|trigger| trigger.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
}

impl TryFrom<String> for CompactionTrigger {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, String> {
        value.parse()
    }
}

impl From<CompactionTrigger> for String {
    fn from(trigger: CompactionTrigger) -> Self {
        trigger.name().to_string()
    }
}

/// What a compaction was, as its boundary records it and every surface says it: what set it
/// going, which strategy wrote the summary, the memories the checkpoint wrote on the way, and
/// whether the recent turns were kept verbatim after the summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct CompactionRecord {
    pub(crate) trigger: CompactionTrigger,
    pub(crate) source: CompactionSource,
    /// Observed from the checkpoint's `memory_write` calls rather than self-reported, so this
    /// cannot disagree with what landed on disk.
    pub(crate) memories_written: Vec<String>,
    /// The outcome, not the intent: the split yields an empty tail whenever the snapped boundary
    /// lands below the minimum worth summarizing, which is routine in a session whose user turns
    /// are separated by long tool runs.
    pub(crate) kept_recent: bool,
    /// The checkpoint turn's rounds as they happened: each reply as its provider returned it, with
    /// what the provider said about it, and the results Meka sent back, the call that handed the
    /// summary in included. Never sent to a provider; empty when no checkpoint round ran, or none
    /// is recorded. A checkpoint that failed keeps the rounds it completed, beside the summary the
    /// summarizer then wrote; a round cut short is not among them, since no call in it ran and
    /// nobody saw it. What of them a surface shows is the surface's to decide.
    pub(crate) checkpoint: Vec<RecordedMessage>,
    /// The summarizer's replies as their provider returned them, when the summarizer wrote the
    /// summary: one, or two when the first held no words and it was asked again.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) summarizer: Vec<RecordedMessage>,
    /// The summary as the agent or the summarizer wrote it, without the sections compaction
    /// writes around it in the message the boundary puts in place of what it replaced: what the
    /// screen and an editor show of the summary, live and on a replay. Never sent to a provider;
    /// the message is what the model reads, and what an export records.
    pub(crate) written_summary: String,
}

/// A message as it happened, outside the conversation: a round of a compaction's checkpoint or a
/// summarizer's reply, and for a reply what its provider said about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RecordedMessage {
    #[serde(flatten)]
    pub(crate) message: Message,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reply: Option<ReplyRecord>,
}

impl RecordedMessage {
    /// `message`, which no provider returned, or whose record was not kept.
    pub(crate) fn unrecorded(message: Message) -> Self {
        Self {
            message,
            reply: None,
        }
    }
}

/// A compaction that ended without replacing the window: what it did on the way, which stays done
/// (the memories its checkpoint wrote), and what ended it. Changes nothing the model sees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct FailedCompaction {
    pub(crate) trigger: CompactionTrigger,
    pub(crate) memories_written: Vec<String>,
    /// See [`CompactionRecord::checkpoint`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) checkpoint: Vec<RecordedMessage>,
    /// See [`CompactionRecord::summarizer`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) summarizer: Vec<RecordedMessage>,
    /// What it failed on, in the words the live failure said; absent when it was stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

impl CompactionRecord {
    /// The class of the stop its last reply ended on, the summarizer's or else the checkpoint's:
    /// how a turn that ran the compaction alone ended. A reply that recorded none ended as an
    /// answer, and so did one that stopped for its calls, since the compaction ran them and the
    /// turn ended on nothing left to run.
    pub(crate) fn last_stop(&self) -> StopReason {
        match self
            .summarizer
            .iter()
            .rev()
            .chain(self.checkpoint.iter().rev())
            .find_map(|round| round.reply.as_ref())
            .and_then(|reply| reply.stop.as_ref())
            .map(|stop| stop.reason)
        {
            None | Some(StopReason::ToolUse) => StopReason::EndTurn,
            Some(reason) => reason,
        }
    }

    /// What the compaction was, in one line: the sentence for what set it going, then what differs
    /// from the ordinary case. Said where a compaction is one line nobody can open: on the export's
    /// marker and as the one-shot's note. The screen and an editor's card say the
    /// [`Self::headline`] and show the rest when the row or the card is opened.
    ///
    /// The memories are named because they are durable and instance-scoped: notes accumulating
    /// unmentioned under an event whose name suggests it only removes things is the surprise this
    /// reporting exists to prevent. The summarizer is named because the summary is then not the
    /// one the agent chose to write, so a user comparing results across compactions has an
    /// explanation for the difference.
    pub(crate) fn note(&self) -> String {
        let mut line = String::from(self.headline());
        for particular in self.particulars() {
            line.push(' ');
            line.push_str(&particular);
        }
        line
    }

    /// What the [`Self::note`] says past its headline, a sentence each: the departures, then the
    /// memories written.
    pub(crate) fn particulars(&self) -> Vec<String> {
        let mut particulars: Vec<String> =
            self.departures().into_iter().map(String::from).collect();
        if !self.memories_written.is_empty() {
            particulars.push(format!(
                "Wrote {} {}: {}.",
                self.memories_written.len(),
                if self.memories_written.len() == 1 {
                    "memory"
                } else {
                    "memories"
                },
                self.memories_written.join(", "),
            ));
        }
        particulars
    }

    /// The sentence for what set the compaction going.
    pub(crate) const fn headline(&self) -> &'static str {
        match self.trigger {
            CompactionTrigger::Reactive => {
                "Autocompacted the conversation after context usage crossed the ceiling."
            }
            CompactionTrigger::Proactive => {
                "Autocompacted the conversation ahead of a turn that would have crossed the ceiling."
            }
            CompactionTrigger::Manual => "Compacted the conversation on user request.",
            CompactionTrigger::Requested => "The agent compacted the conversation.",
            CompactionTrigger::Emergency => {
                "Compacted the conversation after the provider refused the request as too large."
            }
            CompactionTrigger::Retention => {
                "Autocompacted the conversation to summarize turns older than the retention window."
            }
        }
    }

    /// What differs from the ordinary compaction, a sentence each: that no recent turns were kept
    /// verbatim after the summary, so the summary is all the model has of the conversation, and
    /// that the standalone summarizer wrote it rather than the agent.
    pub(crate) fn departures(&self) -> Vec<&'static str> {
        let mut departures = Vec::new();
        if !self.kept_recent {
            departures.push("No recent turns were kept.");
        }
        if self.source == CompactionSource::Summarizer {
            departures.push("Written by the standalone summarizer.");
        }
        departures
    }
}

#[cfg(test)]
impl CompactionRecord {
    /// A record for a fixture whose test asks nothing about the compaction itself.
    pub(crate) fn for_test() -> Self {
        Self {
            trigger: CompactionTrigger::Reactive,
            source: CompactionSource::Checkpoint,
            memories_written: Vec::new(),
            kept_recent: true,
            written_summary: String::new(),
            checkpoint: Vec::new(),
            summarizer: Vec::new(),
        }
    }
}

/// One entry in the underlying event log of a [`Conversation`]. Persisted as a single row in the
/// `messages` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Event {
    /// Adds a message to the materialized view.
    Append(Message),
    /// Marks a compaction boundary: when materializing, drop the last `replaced_count` materialized
    /// messages and push `summary` instead. Subsequent `Append` events extend the new tail. Carries
    /// the set of deferred tools that were active at compaction time so `extract_loaded_tool_names`
    /// can recover them after the boundary (otherwise compaction would silently un-load them), and
    /// the checklist as it stood, for the same reason: the calls that built it are among the rows
    /// the boundary replaces, and the replay that recovers the list reads the view. The record is
    /// what a replay says about the compaction, in the words the live row used.
    CompactBoundary {
        summary: Message,
        replaced_count: usize,
        loaded_tools_snapshot: std::collections::BTreeSet<String>,
        checklist_snapshot: crate::checklist::ChecklistState,
        #[serde(flatten)]
        record: CompactionRecord,
    },
    /// Replaces the last `replaced_count` materialized messages with `messages`. An empty
    /// `messages` therefore means "drop them", which is what [`Conversation::rewind`] emits.
    ///
    /// Position-relative rather than index-addressed, deliberately: like
    /// [`Self::CompactBoundary`] it replays correctly no matter what precedes it.
    /// The corollary is an invariant on producers: emit it only while the messages it replaces are
    /// still the trailing materialized entries.
    Repair {
        replaced_count: usize,
        messages: Vec<Message>,
    },
    /// Replaces the images at `images` with a note ([`BUDGET_REDACTION_NOTE`]), once, so the body
    /// that fit the request budget is the body every later request sends and the cache prefix
    /// ahead of the newest turn stops moving. Redacting per request instead would move it on every
    /// send, since each request re-derives the set from scratch.
    ///
    /// Tail-relative like [`Self::Repair`], and for the same reason: the producer,
    /// `Agent::run_turn`, records it before appending the round's own messages, so the view it
    /// addresses is the view the request was built from.
    Redact { images: Vec<RedactedImage> },
    /// A compaction that ended without replacing the window; see [`FailedCompaction`]. The view
    /// is as it was.
    CompactionFailed(FailedCompaction),
    /// Takes the thinking Claude checks out of every message the view holds at this point (see
    /// [`Message::without_claude_thinking`]), after Claude rejected a thinking block's signature:
    /// a rejected block fails every later request that carries it, wherever it sits. Recorded by
    /// reference, like [`Self::Redact`], rather than as a [`Self::Repair`] carrying copies: the
    /// copies would be every message since the last compaction, dated now, and a turn retention
    /// sweep would keep them for a window past the rows they copied. A message appended after it
    /// keeps its own thinking.
    Unsign,
}

pub(crate) use crate::image::RedactedImage;

/// Meka's class of how a reply ended, the one the loop and every surface read: the provider's own
/// word for it is [`Stop::word`], recorded beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum StopReason {
    /// The model finished its reply.
    EndTurn,
    /// The model asked for tools.
    ToolUse,
    /// The reply reached the output limit and was cut there.
    MaxTokens,
    /// The reply reached the context window and was cut there.
    ContextWindow,
    /// The model, or its provider's classifier, declined the request.
    Refusal,
    /// The provider's content filter withheld the rest of the reply.
    ContentFilter,
    /// A stop the provider named that Meka has no class for; the loop treats it as an end.
    Unknown,
}

impl StopReason {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [StopReason; 7] = [
        Self::EndTurn,
        Self::ToolUse,
        Self::MaxTokens,
        Self::ContextWindow,
        Self::Refusal,
        Self::ContentFilter,
        Self::Unknown,
    ];

    /// The class's one spelling: the turn row, the stored reply record and every surface.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::ToolUse => "tool_use",
            Self::MaxTokens => "max_tokens",
            Self::ContextWindow => "context_window",
            Self::Refusal => "refusal",
            Self::ContentFilter => "content_filter",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for StopReason {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|reason| reason.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a stop reason. Supported: {}",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

impl TryFrom<String> for StopReason {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<StopReason> for String {
    fn from(reason: StopReason) -> Self {
        reason.name().to_string()
    }
}

/// How a reply ended, as recorded: Meka's class of it, the word the provider said it with, and
/// what the provider said about it beside the word, verbatim.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Stop {
    pub(crate) reason: StopReason,
    /// The provider's own stop value as it sent it (Claude's `stop_reason`, Chat Completions'
    /// `finish_reason`, the Responses `status`); absent when the reply stated none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) word: Option<String>,
    /// What the provider said about the stop beside its word, under its own names: Claude's
    /// `stop_details` and `stop_sequence`, the Responses `incomplete_details`, OpenRouter's
    /// `native_finish_reason`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl Stop {
    /// A stop of class `reason` with no word of the provider's beside it: one Meka decides, or a
    /// test's.
    pub(crate) fn of(reason: StopReason) -> Self {
        Self {
            reason,
            word: None,
            details: None,
        }
    }

    /// `self`, classed a refusal when the reply carries the words the model declined in and
    /// otherwise ended as an answer: OpenAI says so in a field of the reply, not in its stop.
    pub(crate) fn with_refusal_of(mut self, content: &[ContentBlock]) -> Self {
        let refused = content
            .iter()
            .any(|block| matches!(block, ContentBlock::Refusal { .. }));
        if refused && self.reason == StopReason::EndTurn {
            self.reason = StopReason::Refusal;
        }
        self
    }
}

/// What a provider said about a reply beside its content, recorded with the reply's row: how it
/// stopped or was cut, the model the response names, the profile that sent the request, and what
/// the response reported spending. A fact about the row, as when it was said is, so a message stays
/// what a wire maps from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ReplyRecord {
    /// How the reply ended, as the provider said; absent on one cut before it said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) stop: Option<Stop>,
    /// What cut the reply short, on one that did not run to its stop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cut: Option<ReplyCut>,
    /// The model the response names; absent when it names none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<String>,
    /// The profile that sent the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) profile: Option<String>,
    /// What the response reported spending, as Meka counts it; absent when it reported nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<crate::stats::TokenUsage>,
    /// The response's fields nothing else holds, verbatim and in place: its id, the raw `usage`
    /// Meka's counts are read from, a choice's `logprobs`, and whatever else the provider said
    /// about the completion. Never what restates the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) provider_fields: Option<ProviderFields>,
}

impl ReplyRecord {
    /// How the reply ended when it did not end as an answer or a round of calls, as a reader of
    /// the conversation labels it: `[cut: <cut>]` for one cut short, `[stop: <class>]` for another
    /// stop.
    pub(crate) fn unfinished_label(&self) -> Option<String> {
        if let Some(cut) = self.cut {
            return Some(format!("[cut: {cut}]"));
        }
        let reason = self.stop.as_ref()?.reason;
        (!matches!(reason, StopReason::EndTurn | StopReason::ToolUse))
            .then(|| format!("[stop: {reason}]"))
    }
}

/// What cut a reply short before its provider said how it ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum ReplyCut {
    /// The turn was stopped.
    Stopped,
    /// An inbox interrupt cut the reply, and the turn went on.
    Interrupted,
    /// The stream failed.
    Failed,
}

impl ReplyCut {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [ReplyCut; 3] = [Self::Stopped, Self::Interrupted, Self::Failed];

    /// The value's one spelling.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
            Self::Interrupted => "interrupted",
            Self::Failed => "failed",
        }
    }
}

impl std::fmt::Display for ReplyCut {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for ReplyCut {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|cut| cut.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a reply cut. Supported: {}",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

impl TryFrom<String> for ReplyCut {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<ReplyCut> for String {
    fn from(cut: ReplyCut) -> Self {
        cut.name().to_string()
    }
}

/// Characters a session title keeps on every wire before it is cut, and the most a user may set,
/// so a surface that fits one fits the other; the screen's session picker, which lays the title
/// out itself, asks for as many as its row holds.
pub(crate) const TITLE_CHARS: usize = 200;

/// Append-only conversation: an event log, plus the materialized message view derived from it.
#[derive(Debug, Default, Clone)]
pub(crate) struct Conversation {
    events: Vec<Event>,
    /// Materialized view kept in lockstep with `events`. Rebuilt by `rebuild_materialized` after
    /// every mutation; reads are zero-cost.
    materialized: Vec<Message>,
    /// The checklist the boundary whose summary heads the view recorded, when one does; see
    /// [`Self::checklist_baseline`]. Kept in lockstep with `materialized` by the same rebuild.
    baseline_checklist: Option<crate::checklist::ChecklistState>,
    /// Whether a compaction's summary heads the view, as it does from the boundary until a rewind
    /// before it. Kept in lockstep with `materialized` by the same rebuild.
    summary_heads: bool,
    /// Images replaced since the last full rebuild, for
    /// [`Conversation::invalid_images_replaced`].
    invalid_images_replaced: usize,
    /// Whether this log came off disk and the model has not been told yet.
    ///
    /// Set by [`Self::from_events`] and consumed once by [`Self::take_resumed_notice`]. Lives here
    /// rather than on the agent because the conversation is the thing that was hydrated, and
    /// hydration has four entry points (screen resume, two ACP paths, serve reattach) that all
    /// reach `from_events`. A flag set at those call sites instead would be a flag the fifth
    /// one forgets.
    resumed_undisclosed: bool,
}

impl Conversation {
    /// An empty log.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Hydrate from a sequence of events (typically loaded from the session DB on resume). The
    /// materialized view is computed once and cached.
    pub(crate) fn from_events(events: Vec<Event>) -> Self {
        let mut log = Self {
            events,
            ..Self::default()
        };
        log.rebuild_materialized();
        // Only when something was actually restored. Resuming a session that never got a turn
        // leaves the model nothing to hold a stale belief about, and the notice would be a warning
        // against trusting a history that does not exist.
        log.resumed_undisclosed = !log.materialized.is_empty();
        log
    }

    /// Hydrate from a flat `Vec<Message>`; every entry becomes an `Event::Append`. Test-only: every
    /// hydration site in the tree goes through [`Self::from_events`], and a `Vec<Message>` cannot
    /// express the boundary and repair events a real log carries, so a production caller would be
    /// silently discarding them.
    #[cfg(test)]
    pub(crate) fn from_vec(entries: Vec<Message>) -> Self {
        let events = entries.into_iter().map(Event::Append).collect();
        Self::from_events(events)
    }

    /// The session's title, the same on every surface that labels one: the first words a user
    /// said, their whitespace collapsed to single spaces, cut to [`TITLE_CHARS`] with an ellipsis.
    /// Empty until a user has said something.
    ///
    /// Read from the log rather than the view, because a compaction replaces the view's first user
    /// message with its summary and a label that changed when the session was reopened would name
    /// nothing. The words are the first block of a user `Append` that carries a person's words by
    /// [`ContentBlock::spoken_words`]: typed text, an inbox item's body, a fire's prompt, an
    /// invocation's words or name; a turn's context block and a note Meka wrote carry none. The
    /// store's `title_of_first_user_row` selects the same row in SQL. A log pruned by
    /// [`Self::prune_compacted_events`] starts at its last boundary and no longer holds the first
    /// turn; every surface that labels a stored session reads the store's rows, which do.
    pub(crate) fn title(&self) -> String {
        self.title_within(TITLE_CHARS)
    }

    /// [`Self::title`] cut to `chars` rather than [`TITLE_CHARS`], for a surface that lays the
    /// title out itself and has more room than the wire carries, as the screen's session picker
    /// does on a wide terminal.
    pub(crate) fn title_within(&self, chars: usize) -> String {
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::Append(message) if message.role == Role::User => Some(message),
                _ => None,
            })
            .flat_map(|message| &message.content)
            .find_map(|block| {
                let words = block
                    .spoken_words()?
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                (!words.is_empty()).then_some(words)
            })
            .map(|words| {
                if words.chars().count() <= chars {
                    words
                } else {
                    let kept: String = words.chars().take(chars).collect();
                    format!("{}…", kept.trim_end())
                }
            })
            .unwrap_or_default()
    }

    /// Read the underlying event log (e.g. for persistence or scanning).
    pub(crate) fn events(&self) -> &[Event] {
        &self.events
    }

    /// Whether this turn is the first since the log was restored from disk, clearing the flag.
    ///
    /// Told once. The conversation the model reads back is a record of what happened, not proof
    /// that any of it still holds: a tool that was holding something open across those turns has
    /// been restarted along with the process, and nothing else in the context block says so
    /// (permission, cwd and the tool catalog are all restated every turn regardless).
    pub(crate) fn take_resumed_notice(&mut self) -> bool {
        std::mem::take(&mut self.resumed_undisclosed)
    }

    /// Put back a notice taken by a turn that then failed and had its user message popped.
    ///
    /// The notice rides that message and nothing else, so without this a resume whose first turn
    /// errors is never told. Same withdrawal [`Agent::run_turn`] performs on the world snapshot.
    ///
    /// [`Agent::run_turn`]: crate::agent::Agent::run_turn
    pub(crate) fn restore_resumed_notice(&mut self) {
        self.resumed_undisclosed = true;
    }

    /// The only canonical mutation. Push a fully-formed message onto the log as a new
    /// `Event::Append`.
    pub(crate) fn append(&mut self, message: Message) {
        self.materialized.push(message.clone());
        self.events.push(Event::Append(message));
        // Pushed straight onto the view rather than going through `rebuild_materialized`, so the
        // repair has to be applied to the new tail explicitly or an appended block would be the one
        // thing materialization never checks.
        if let Some(tail) = self.materialized.last_mut() {
            self.invalid_images_replaced += repair_invalid_images(std::slice::from_mut(tail));
        }
    }

    /// Read-only borrow of the materialized view: what the model is about to be shown, which is
    /// what providers and the token scanner consume. Anything asking what *happened* reads
    /// [`Self::events`] instead, because a compaction or a repair can take a record out of this
    /// view while leaving it in the log.
    pub(crate) fn as_slice(&self) -> &[Message] {
        &self.materialized
    }

    /// How many messages the model is about to be shown.
    pub(crate) fn len(&self) -> usize {
        self.materialized.len()
    }

    /// Whether the model is about to be shown nothing.
    pub(crate) fn is_empty(&self) -> bool {
        self.materialized.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn last(&self) -> Option<&Message> {
        self.materialized.last()
    }

    /// How many events the log holds.
    ///
    /// For a caller that appended something and wants to know whether *anything at all* has
    /// happened since. Every mutation goes through the event log, so an unchanged count is a
    /// much stronger statement than any inspection of the materialized tail: compaction, a
    /// repair, a thinking-only nudge and a tool round all move it, including the ones that
    /// leave a tail still shaped like a turn-opening prompt.
    pub(crate) fn events_len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log ends with an appended message that opened a turn, i.e. a turn got as far as
    /// its prompt and no further: no assistant reply, no tool results.
    ///
    /// `run_turn` uses this to decide whether a failure is safe to withdraw. Counting messages
    /// would not do: both compaction paths and the `InvalidRequest` repair move `len()` under
    /// the turn that is running.
    ///
    /// **Both halves are load-bearing, and the event half is the subtle one.** A compaction summary
    /// is itself a plain `User` message, so [`Self::replace_for_compaction`] with an empty tail
    /// leaves a materialized view whose last entry satisfies [`opens_turn`], and withdrawing
    /// *that* would delete the summary standing in for the whole conversation. Requiring a trailing
    /// [`Event::Append`] says "the message on the end is one somebody appended", which a summary
    /// carried by a `CompactBoundary` is not. [`Self::pop_unsaved`] guards its own removal the same
    /// way and for the same reason.
    ///
    /// A reply cut short that the context keeps none of is recorded and taken back at once, an
    /// `Append` of the model's and a repair of it to nothing, which leaves the opening standing as
    /// it stood: such pairs at the end are looked past.
    pub(crate) fn ends_on_a_turn_opening(&self) -> bool {
        let mut events = self.events.as_slice();
        while let [
            earlier @ ..,
            Event::Append(reply),
            Event::Repair {
                replaced_count: 1,
                messages,
            },
        ] = events
            && reply.role == Role::Assistant
            && messages.is_empty()
        {
            events = earlier;
        }
        matches!(events.last(), Some(Event::Append(_)))
            && self.materialized.last().is_some_and(opens_turn)
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, Message> {
        self.materialized.iter()
    }

    /// The words of the most recent `Role::Assistant` message, its text and refusal blocks a blank
    /// line apart, or `None` when there is no assistant message or it holds no words. Walks
    /// backward, which is necessary because a turn that ended via tool-use leaves a `Role::User`
    /// tool-result trailer in the conversation, hiding the assistant's final text from a look at
    /// the last message alone.
    pub(crate) fn last_assistant_text(&self) -> Option<String> {
        let reply = self
            .materialized
            .iter()
            .rev()
            .find(|message| matches!(message.role, crate::conversation::Role::Assistant))?;
        let words = reply
            .shown_content()
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } | ContentBlock::Refusal { text, .. } => {
                    Some(text.trim())
                }
                _ => None,
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        (!words.is_empty()).then_some(words)
    }

    /// Roll back an [`Conversation::append`] that did not reach the persistence layer. Used by
    /// `Agent::run_turn`'s failure arm when the prompt's eager persist failed and nothing was
    /// persisted after it. Returns the popped message for diagnostics.
    ///
    /// Removes only a trailing `Event::Append`. If the last event is a `Event::CompactBoundary`
    /// (which can only be true after a successful compaction round-trip), this is a programmer
    /// error and the call returns `None` without mutating the log.
    pub(crate) fn pop_unsaved(&mut self) -> Option<Message> {
        match self.events.last() {
            Some(Event::Append(_)) => {}
            _ => return None,
        }
        let popped = match self.events.pop() {
            Some(Event::Append(message)) => message,
            #[allow(
                clippy::unreachable,
                reason = "the match above returned unless the last event is an `Append`, and nothing ran between"
            )]
            _ => unreachable!("checked Append above"),
        };
        // Mirror the in-memory removal in the materialized view.
        self.materialized.pop();
        Some(popped)
    }

    /// Replace the visible window with `summary` followed by `tail`. Used by `compact_session`:
    /// appends one [`Event::CompactBoundary`] (which tells the materializer to truncate the prior
    /// tail and push the summary), then appends each kept tail message as an [`Event::Append`]. The
    /// events log itself is *only ever appended to*; pre-compaction events stay untouched in the
    /// log and on disk.
    ///
    /// `loaded_tools_snapshot` is the active deferred-tool set captured from the conversation
    /// *before* the boundary is appended. Carried so `extract_loaded_tool_names_from_events` can
    /// recover deferred tools after the boundary; otherwise a session that loaded a tool, then
    /// compacted, would fall back to the deferred state.
    pub(crate) fn replace_for_compaction(
        &mut self,
        summary: Message,
        tail: Vec<Message>,
        loaded_tools_snapshot: std::collections::BTreeSet<String>,
        checklist_snapshot: crate::checklist::ChecklistState,
        record: CompactionRecord,
    ) {
        let replaced_count = self.materialized.len();
        self.events.push(Event::CompactBoundary {
            summary,
            replaced_count,
            loaded_tools_snapshot,
            checklist_snapshot,
            record,
        });
        for message in tail {
            self.events.push(Event::Append(message));
        }
        self.rebuild_materialized();
    }

    /// Undo the most recent [`Self::replace_for_compaction`], restoring the pre-compaction view.
    ///
    /// The counterpart to [`Self::pop_repair`], and it exists for the same reason: compaction
    /// rewrites the in-memory conversation *before* it persists, so a failed write would otherwise
    /// leave the model reasoning from a summary the database has never heard of, while the store
    /// still holds the full context. The caller was told the compaction failed, so the two must
    /// agree that it did.
    ///
    /// Only correct while the caller still holds the lock it compacted under, and before
    /// [`Self::prune_compacted_events`] has dropped the superseded events: this truncates back to
    /// the boundary on the assumption that everything from there on is what the compaction just
    /// pushed. Returns `false` when there is no boundary to undo.
    pub(crate) fn pop_compaction(&mut self) -> bool {
        let Some(boundary) = self
            .events
            .iter()
            .rposition(|event| matches!(event, Event::CompactBoundary { .. }))
        else {
            return false;
        };
        self.events.truncate(boundary);
        self.rebuild_materialized();
        true
    }

    /// Replace the trailing `replaced_count` materialized messages with `messages`, appending one
    /// [`Event::Repair`]. Returns that event so the caller can persist it; the log is only ever
    /// appended to, so the originals stay in memory and on disk for `meka session export`.
    ///
    /// Used by `Agent::run_turn` when the provider rejects content it has just appended, and by
    /// [`Self::rewind`] with an empty `messages`. Callers must satisfy [`Event::Repair`]'s
    /// invariant: the replaced messages have to be the current tail.
    pub(crate) fn replace_tail(&mut self, replaced_count: usize, messages: Vec<Message>) -> Event {
        let event = Event::Repair {
            replaced_count,
            messages,
        };
        self.events.push(event.clone());
        self.rebuild_materialized();
        event
    }

    /// Record that a request budget redacted `images`, replacing each with Meka's note in the
    /// view. Returns the event so the caller can persist it; see [`Event::Redact`] for why it is
    /// recorded rather than redone per request.
    pub(crate) fn redact_images(&mut self, images: Vec<RedactedImage>) -> Event {
        let event = Event::Redact { images };
        self.events.push(event.clone());
        self.rebuild_materialized();
        event
    }

    /// Record that the thinking Claude checks is no longer sent for anything the view holds, when
    /// it holds some. Returns the event so the caller can persist it; see [`Event::Unsign`].
    pub(crate) fn unsign_thinking(&mut self) -> Option<Event> {
        if !self
            .materialized
            .iter()
            .any(Message::carries_claude_thinking)
        {
            return None;
        }
        let event = Event::Unsign;
        self.events.push(event.clone());
        self.rebuild_materialized();
        Some(event)
    }

    /// Undo the most recent [`Self::replace_tail`] or [`Self::unsign_thinking`], restoring the
    /// messages it changed.
    ///
    /// The inverse of [`Self::pop_unsaved`] for repairs: `run_turn` degrades content, retries, and
    /// calls this when the retry fails too, so a misdiagnosed rejection leaves the conversation
    /// byte-identical instead of permanently losing a good tool result. Returns whether anything
    /// was undone; a trailing event that is neither is left alone.
    pub(crate) fn pop_repair(&mut self) -> bool {
        if !matches!(
            self.events.last(),
            Some(Event::Repair { .. } | Event::Unsign)
        ) {
            return false;
        }
        self.events.pop();
        self.rebuild_materialized();
        true
    }

    /// Drop the last `turns` user turns and everything after them, as one [`Event::Repair`] with an
    /// empty replacement. Returns the event to persist, or `None` when there is nothing to drop.
    ///
    /// The cut snaps to a message that opens a turn (a `User` message carrying no `tool_result`),
    /// so a `tool_use` is never separated from its `tool_result`. A summary heading the view is
    /// no turn and is never taken: it stands for the turns before it, which no rewind gives
    /// back, so taking it would only empty the context, and `POST /rewind` refuses those turns
    /// for the same reason. The rounds after it that open no turn are the rest of the turn the
    /// compaction cut into, and count as that turn.
    pub(crate) fn rewind(&mut self, turns: usize) -> Option<Event> {
        if turns == 0 {
            return None;
        }
        let first = usize::from(self.summary_heads);
        let cut = self
            .materialized
            .iter()
            .enumerate()
            .skip(first)
            .filter(|(index, message)| {
                (self.summary_heads && *index == first) || opens_turn(message)
            })
            .map(|(index, _)| index)
            .nth_back(turns - 1)?;
        let replaced_count = self.materialized.len() - cut;
        Some(self.replace_tail(replaced_count, Vec::new()))
    }

    /// Drop the last `count` messages, as one [`Event::Repair`] that puts nothing back: a rewind
    /// whose cut its caller has placed, as the HTTP door places it before a named turn. `None`
    /// when `count` is zero or more than the view holds.
    pub(crate) fn rewind_messages(&mut self, count: usize) -> Option<Event> {
        if count == 0 || count > self.materialized.len() {
            return None;
        }
        Some(self.replace_tail(count, Vec::new()))
    }

    /// Drop every event preceding the most recent `CompactBoundary`.
    ///
    /// Those events are fully superseded: a `CompactBoundary` truncates all materialized messages
    /// before it and replaces them with its summary, and
    /// [`crate::tools::load_tool::extract_loaded_tool_names_from_events`] reads the boundary's
    /// `loaded_tools_snapshot` rather than the events preceding it. So the materialized view
    /// and the recovered tool set are byte-identical before and after this call; it only stops
    /// the in-memory log from growing unbounded across a long-lived, repeatedly-compacted
    /// session.
    ///
    /// Persistence is unaffected: every event was already written to its own row by `save_event`,
    /// so the on-disk log stays complete. A resume starts from the same place:
    /// `Store::load_view_events` reads the last boundary row and what follows, so a log in memory
    /// begins at a boundary whether this process compacted it or an earlier one did.
    pub(crate) fn prune_compacted_events(&mut self) {
        let last_boundary = self
            .events
            .iter()
            .rposition(|event| matches!(event, Event::CompactBoundary { .. }));
        if let Some(index) = last_boundary
            && index > 0
        {
            self.events.drain(..index);
            self.rebuild_materialized();
        }
    }

    /// Drop assistant messages whose `tool_use` blocks lack matching `tool_result`s in the
    /// immediately-following user message. Returns the dropped messages so callers can log them.
    /// Used at session resume to repair the log after a crash mid-tool-call (the Anthropic API
    /// rejects orphaned `tool_use` blocks).
    ///
    /// The drop is one [`Event::Repair`] of the view's tail, from the message before the first
    /// orphan, putting back everything but the orphans, so it is the log's last event and a caller
    /// with a store saves it: the store's replay then holds the view this does, and a count taken
    /// on either is the count on both. Starting a message early means it puts that message back,
    /// a view opening on a user message, a prompt or a summary, so it never reads as a rewind.
    /// `Event::CompactBoundary` events are never touched (their synthetic summary is a plain user
    /// message that can't be orphaned).
    pub(crate) fn sanitize_orphans(&mut self) -> Vec<Message> {
        let Replayed { placed, .. } = replay(self.events.iter());
        let orphans = orphan_positions(&placed);
        let Some(&first) = orphans.first() else {
            return Vec::new();
        };
        let start = first.saturating_sub(1);
        let (dropped, kept): (Vec<_>, Vec<_>) = placed
            .into_iter()
            .enumerate()
            .skip(start)
            .partition(|(position, _)| orphans.contains(position));
        let replaced_count = dropped.len() + kept.len();
        self.replace_tail(
            replaced_count,
            kept.into_iter().map(|(_, placed)| placed.message).collect(),
        );
        dropped
            .into_iter()
            .map(|(_, placed)| placed.message)
            .collect()
    }

    /// How many images have been replaced because their bytes disagreed with their declared
    /// `media_type`: reset by each full rebuild, then added to by each [`Self::append`]. See
    /// [`repair_invalid_images`] for why that is done at all; callers use this only to log it at
    /// resume, where it is read straight after [`Self::from_events`] and so counts exactly the
    /// images the stored log carried.
    pub(crate) fn invalid_images_replaced(&self) -> usize {
        self.invalid_images_replaced
    }

    fn rebuild_materialized(&mut self) {
        let Replayed { placed, .. } = replay(self.events.iter());
        self.summary_heads = matches!(
            placed.first().map(|placed| placed.source),
            Some(Source::Boundary(_))
        );
        self.baseline_checklist = match placed.first().map(|placed| placed.source) {
            Some(Source::Boundary(index)) => match self.events.get(index) {
                Some(Event::CompactBoundary {
                    checklist_snapshot, ..
                }) => Some(checklist_snapshot.clone()),
                _ => None,
            },
            _ => None,
        };
        self.materialized = placed.into_iter().map(|placed| placed.message).collect();
        self.invalid_images_replaced = repair_invalid_images(&mut self.materialized);
    }

    /// Where a replay of the checklist starts: the list the heading boundary recorded and the
    /// messages after its summary, or an empty list and the whole view when no boundary heads it.
    ///
    /// A boundary clears the view, so its summary is the first message while it stands; a repair
    /// that truncates past it leaves a view that begins with ordinary rows, which then speak for
    /// themselves.
    pub(crate) fn checklist_baseline(&self) -> (crate::checklist::ChecklistState, &[Message]) {
        match &self.baseline_checklist {
            Some(snapshot) => (
                snapshot.clone(),
                self.materialized.get(1..).unwrap_or_default(),
            ),
            None => (
                crate::checklist::ChecklistState::default(),
                self.materialized.as_slice(),
            ),
        }
    }
}

/// The event whose row a materialized message is shown as: the one that placed it, or for a
/// message a repair rewrote in place, the one that placed what it rewrote.
#[derive(Debug, Clone, Copy)]
enum Source {
    Append(usize),
    Boundary(usize),
    Repair(usize),
}

/// One materialized message and where it came from.
struct Placed {
    message: Message,
    source: Source,
    /// The earliest event the message stands for: its own; for a summary, the earliest one a
    /// message it replaced stood for; for a repair's replacement, or a prompt that came back with
    /// more joined to it, the one the message it took the place of stood for. Never decreasing
    /// along the view, since the view's tail is always built from the log's.
    origin: usize,
    marker: Option<CompactionMarker>,
}

/// What a repair that put nothing back took off the view, by the stamps of the events it spans:
/// every event from `first`, the earliest a dropped message stood for, up to `at`, the repair's
/// own. A message a repair rewrote or a summary replaced stands for the events it took the place
/// of, so none of those is left in the view either.
///
/// A rewind is one such repair. A turn writes others: taking its prompt back to join a message
/// that arrived behind it, and withdrawing a prompt the provider refused so its items can be
/// offered again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Rewind<Stamp> {
    pub(crate) first: Stamp,
    pub(crate) at: Stamp,
    /// The turns it dropped, counted as [`Conversation::rewind`] counts them.
    pub(crate) turns: usize,
    /// Whether the one message it took came straight back with more joined to it, as a prompt
    /// does when a message arrives behind it before the model has answered. A summary taken back
    /// so spans the repair alone, since the summary's row stays the compaction it was.
    pub(crate) rejoined: bool,
}

/// A replay of the log: the view and its rewinds by event index.
struct Replayed {
    placed: Vec<Placed>,
    rewinds: Vec<Rewind<usize>>,
    /// Whether every repair found the messages it replaces: false once one reached past the first
    /// event, as one does whose earlier rows a turn sweep took.
    whole: bool,
}

/// Which compaction a summary stands for, as a view of the log reports it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompactionMarker {
    /// How many materialized messages the boundary recorded replacing.
    pub(crate) replaced_count: usize,
    /// Which compaction produced it, counting from 1.
    pub(crate) generation: u64,
    /// What the boundary recorded about the compaction.
    pub(crate) record: CompactionRecord,
}

/// Where in `removed` each of `messages` stands, when `messages` is `removed` with one or more
/// messages left out and nothing else changed; `None` otherwise.
fn kept_in_order(removed: &[Placed], messages: &[Message]) -> Option<Vec<usize>> {
    if messages.is_empty() || messages.len() >= removed.len() {
        return None;
    }
    let mut kept = Vec::with_capacity(messages.len());
    let mut from = 0;
    for message in messages {
        let index = from
            + removed
                .get(from..)?
                .iter()
                .position(|placed| &placed.message == message)?;
        kept.push(index);
        from = index + 1;
    }
    Some(kept)
}

/// Replay `events` into the messages the model sees, with each one's origin, and what each rewind
/// dropped.
///
/// The one statement of the rules. A boundary replaces everything before it: its producer,
/// [`Conversation::replace_for_compaction`], records the whole view as `replaced_count`, but that
/// number was measured against the view in memory, which a resume that dropped orphans has made
/// shorter than what the store replays. Truncating by the count would leave the difference
/// standing above the summary, and every later compaction would widen it. A repair is
/// position-relative by design, since it replaces a trailing run its producer just appended.
fn replay<'a>(events: impl Iterator<Item = &'a Event>) -> Replayed {
    let mut placed: Vec<Placed> = Vec::new();
    let mut generation: u64 = 0;
    let mut rewinds: Vec<Rewind<usize>> = Vec::new();
    let mut whole = true;
    // The one message the last repair took, while the next event may bring it back joined.
    let mut taken: Option<Placed> = None;
    for (index, event) in events.enumerate() {
        let taken_back = taken.take();
        match event {
            Event::Append(message) => {
                let rejoined = taken_back.filter(|taken| {
                    taken.message.role == message.role
                        && message.content.len() > taken.message.content.len()
                        && message.content.starts_with(&taken.message.content)
                });
                if let Some(taken) = &rejoined
                    && let Some(rewind) = rewinds.last_mut()
                {
                    rewind.rejoined = true;
                    // A summary stands for the turns before it, which the message it joined does
                    // not show again: only the row that took it back goes.
                    if matches!(taken.source, Source::Boundary(_)) {
                        rewind.first = rewind.at;
                    }
                }
                // A summary something joined, as an interrupt joins the items that arrived onto
                // a lone summary, still stands for its boundary: the list the boundary recorded
                // is the checklist's baseline, and the marker is what a reader of the log shows.
                let (source, marker) = match &rejoined {
                    Some(taken) if matches!(taken.source, Source::Boundary(_)) => {
                        (taken.source, taken.marker.clone())
                    }
                    _ => (Source::Append(index), None),
                };
                placed.push(Placed {
                    message: message.clone(),
                    source,
                    origin: rejoined.map_or(index, |taken| taken.origin),
                    marker,
                });
            }
            Event::CompactBoundary {
                summary,
                replaced_count,
                record,
                ..
            } => {
                generation = generation.saturating_add(1);
                let origin = placed.first().map_or(index, |first| first.origin);
                placed.clear();
                placed.push(Placed {
                    message: summary.clone(),
                    source: Source::Boundary(index),
                    origin,
                    marker: Some(CompactionMarker {
                        replaced_count: *replaced_count,
                        generation,
                        record: record.clone(),
                    }),
                });
            }
            Event::Repair {
                replaced_count,
                messages,
            } => {
                whole &= *replaced_count <= placed.len();
                let truncate_to = placed.len().saturating_sub(*replaced_count);
                // A repair that rewrites the whole view and puts something back has rewritten the
                // summary heading it, not removed it: the first message it places is that summary
                // as the repair left it, and still stands for the boundary, with the list the
                // boundary recorded. Only a repair that puts nothing back takes the boundary away.
                let heading_boundary = (truncate_to == 0)
                    .then(|| placed.first())
                    .flatten()
                    .filter(|first| matches!(first.source, Source::Boundary(_)))
                    .map(|first| (first.source, first.marker.clone()));
                let mut removed = placed.split_off(truncate_to);
                if messages.is_empty()
                    && let Some(first) = removed.first()
                {
                    rewinds.push(Rewind {
                        first: first.origin,
                        at: index,
                        turns: removed
                            .iter()
                            .filter(|placed| opens_turn(&placed.message))
                            .count(),
                        rejoined: false,
                    });
                    if removed.len() == 1 {
                        taken = removed.pop();
                    }
                }
                // A replacement that is what it replaced with messages left out keeps the rest as
                // they were placed, each still the row it was said in: the orphan drop on resume
                // writes one, and the turns after an orphan stay theirs.
                if let Some(kept) = kept_in_order(&removed, messages) {
                    let mut removed: Vec<Option<Placed>> = removed.into_iter().map(Some).collect();
                    placed.extend(
                        kept.into_iter()
                            .filter_map(|index| removed.get_mut(index).and_then(Option::take)),
                    );
                    continue;
                }
                // A replacement of the same length stands for what it replaced position by
                // position, and is shown as the row it rewrote, as a redacted message is; one of
                // another length stands for the whole run and is the repair's own.
                let same_length = removed.len() == messages.len();
                let origin = |position: usize| {
                    let replaced = if same_length {
                        removed.get(position)
                    } else {
                        removed.first()
                    };
                    replaced.map_or(index, |replaced| replaced.origin)
                };
                let source = |position: usize| {
                    removed
                        .get(position)
                        .filter(|_| same_length)
                        .map_or(Source::Repair(index), |replaced| replaced.source)
                };
                placed.extend(messages.iter().enumerate().map(|(position, message)| {
                    match (position, &heading_boundary) {
                        (0, Some((source, marker))) => Placed {
                            message: message.clone(),
                            source: *source,
                            origin: origin(position),
                            marker: marker.clone(),
                        },
                        _ => Placed {
                            message: message.clone(),
                            source: source(position),
                            origin: origin(position),
                            marker: None,
                        },
                    }
                }));
            }
            Event::Redact { images } => {
                for image in images {
                    let Some(at) = placed.len().checked_sub(image.from_end) else {
                        continue;
                    };
                    if let Some(placed) = placed.get_mut(at) {
                        redact_image(&mut placed.message, image);
                    }
                }
            }
            Event::Unsign => {
                for placed in &mut placed {
                    if let Some(without) = placed.message.without_claude_thinking() {
                        placed.message = without;
                    }
                }
            }
            // Changes nothing in the view, and stands between nothing a repair could rejoin.
            Event::CompactionFailed(_) => taken = taken_back,
        }
    }
    Replayed {
        placed,
        rewinds,
        whole,
    }
}

/// Replace the image `image` names in `message` with [`BUDGET_REDACTION_NOTE`]. A position that
/// names something else is left alone: the event was recorded against a view this one may not
/// be, and removing text on a stale address would be worse than keeping an image.
fn redact_image(message: &mut Message, image: &RedactedImage) {
    let note = || BUDGET_REDACTION_NOTE.to_string();
    match (message.content.get_mut(image.block), image.item) {
        (Some(block @ ContentBlock::Image { .. }), None) => {
            *block = ContentBlock::HarnessNote { note: note() };
        }
        (Some(ContentBlock::ToolResult { content, .. }), Some(item)) => {
            if let Some(entry @ ToolResultContent::Image { .. }) = content.get_mut(item) {
                *entry = ToolResultContent::HarnessNote { note: note() };
            }
        }
        _ => {}
    }
}

/// How many images `messages` hold, attached to a message or inside a tool result: what a profile
/// that does not read images would be sent, and so what a turn on one is refused over.
pub(crate) fn count_images(messages: &[Message]) -> usize {
    messages
        .iter()
        .flat_map(|message| message.content.iter())
        .map(|block| match block {
            ContentBlock::Image { .. } => 1,
            ContentBlock::ToolResult { content, .. } => content
                .iter()
                .filter(|item| matches!(item, ToolResultContent::Image { .. }))
                .count(),
            _ => 0,
        })
        .sum()
}

/// The replayed log with, for each message, when it was written and whether it is a compaction
/// summary. Three vectors of one length, which every arm of the replay maintains.
pub(crate) struct AnnotatedView<Stamp> {
    pub(crate) messages: Vec<Message>,
    /// Each message's stamp: whatever the caller stored beside the event that placed it.
    pub(crate) stamps: Vec<Stamp>,
    pub(crate) markers: Vec<Option<CompactionMarker>>,
    /// Whether the events held everything their repairs replace. A view below a boundary loses
    /// that once a turn sweep takes rows a repair rewrote, and is then no view anything saw.
    pub(crate) whole: bool,
}

/// What `GET /v1/sessions/{id}/context/messages` serves: the same view the model gets, by the same
/// rules, with each message stamped from the row it is shown as. A summary takes its boundary's
/// stamp. A message rewritten in place, by a redaction or by a repair of the same length, keeps the
/// stamp of the row that said it, since it is still that message, and so does one a repair kept as
/// it was; one a repair of another length put in place of others takes the repair's.
pub(crate) fn materialize_annotated<Stamp: Clone>(
    events: &[(Stamp, Event)],
) -> AnnotatedView<Stamp> {
    let Replayed { placed, whole, .. } = replay(events.iter().map(|(_, event)| event));
    let mut messages = Vec::with_capacity(placed.len());
    let mut stamps = Vec::with_capacity(placed.len());
    let mut markers = Vec::with_capacity(placed.len());
    for placed in placed {
        let index = match placed.source {
            Source::Append(index) | Source::Boundary(index) | Source::Repair(index) => index,
        };
        let Some((stamp, _)) = events.get(index) else {
            continue;
        };
        stamps.push(stamp.clone());
        markers.push(placed.marker);
        messages.push(placed.message);
    }
    repair_invalid_images(&mut messages);
    AnnotatedView {
        messages,
        stamps,
        markers,
        whole,
    }
}

/// The rewinds in `events`, by the rules the view is replayed by, each in the stamps the caller
/// stored beside its events.
pub(crate) fn rewinds<Stamp: Clone>(events: &[(Stamp, Event)]) -> Vec<Rewind<Stamp>> {
    replay(events.iter().map(|(_, event)| event))
        .rewinds
        .into_iter()
        .filter_map(|rewind| {
            Some(Rewind {
                first: events.get(rewind.first)?.0.clone(),
                at: events.get(rewind.at)?.0.clone(),
                turns: rewind.turns,
                rejoined: rewind.rejoined,
            })
        })
        .collect()
}

/// Replace every image whose bytes disagree with its declared `media_type` with a text note,
/// returning how many were replaced.
///
/// Providers sniff and answer 400 on a mismatch, and because the block is already committed to the
/// session that 400 repeats on every later request, leaving it unusable. `Agent::run_turn` recovers
/// from a rejection it causes itself, but not from one already on disk: by then the block is
/// outside the window a rejection is allowed to blame. Handling it here heals such a session on the
/// next resume with no provider round trip at all.
///
/// Applied to the materialized view during every rebuild rather than as a one-shot pass, so a later
/// compaction or rewind can't quietly reinstate what it removed. Only the first few base64
/// characters of each image are decoded, so the cost is a fixed handful of bytes per image.
fn repair_invalid_images(messages: &mut [Message]) -> usize {
    let mismatched = |source: &crate::image::ImageSource| {
        // A reference carries no bytes to judge; the bytes are checked once they are inlined.
        let Some(data) = source.base64_data() else {
            return false;
        };
        match crate::image::classify_base64_prefix(data) {
            // Undecodable bytes aren't evidence of a mismatch: an encoding this build can't read
            // may still be one the provider accepts.
            crate::image::ImageHandling::Unsupported => false,
            crate::image::ImageHandling::PassThrough(format)
            | crate::image::ImageHandling::Convert(format) => !format
                .to_mime_type()
                .eq_ignore_ascii_case(source.media_type()),
        }
    };
    let note = |source: &crate::image::ImageSource| {
        format!(
            "An image here was removed: it is declared {} but the bytes are something else, \
             which the provider refuses.",
            source.media_type()
        )
    };

    let mut replaced = 0usize;
    for message in messages {
        for block in &mut message.content {
            match block {
                ContentBlock::Image { source } if mismatched(source) => {
                    replaced += 1;
                    *block = ContentBlock::HarnessNote { note: note(source) };
                }
                ContentBlock::ToolResult {
                    content, is_error, ..
                } => {
                    let mut touched = false;
                    for item in content.iter_mut() {
                        if let crate::conversation::ToolResultContent::Image { source } = item
                            && mismatched(source)
                        {
                            replaced += 1;
                            touched = true;
                            *item = ToolResultContent::HarnessNote { note: note(source) };
                        }
                    }
                    if touched {
                        *is_error = true;
                    }
                }
                _ => {}
            }
        }
    }
    replaced
}

impl<'a> IntoIterator for &'a Conversation {
    type IntoIter = std::slice::Iter<'a, Message>;
    type Item = &'a Message;

    fn into_iter(self) -> Self::IntoIter {
        self.materialized.iter()
    }
}

/// The positions in `placed`, a replay's view, of the assistant messages whose `tool_use` blocks
/// the next message does not answer. The check uses the *materialized* view so a
/// `CompactBoundary` between an orphan and its would-be result correctly counts as orphaned. A
/// message a repair placed takes part in the adjacency scan, since a `tool_result` inside one
/// answers the `tool_use` before it, but only one an `Append` placed is reported.
fn orphan_positions(placed: &[Placed]) -> Vec<usize> {
    let mut orphan = Vec::new();
    for (position, current) in placed.iter().enumerate() {
        if !matches!(current.source, Source::Append(_)) {
            continue;
        }
        let message = &current.message;
        if message.role != Role::Assistant {
            continue;
        }
        let tool_use_ids: Vec<&str> = message
            .content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::ToolUse { id, .. } = block {
                    Some(id.as_str())
                } else {
                    None
                }
            })
            .collect();
        if tool_use_ids.is_empty() {
            continue;
        }
        let next = placed.get(position + 1).map(|placed| &placed.message);
        let has_results = next.is_some_and(|next_message| {
            next_message.role == Role::User
                && tool_use_ids.iter().all(|id| {
                    next_message.content.iter().any(|block| {
                        matches!(block, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == *id)
                    })
                })
        });
        if !has_results {
            orphan.push(position);
        }
    }
    orphan
}

/// Who a message is from, as every provider's wire spells it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum Role {
    User,
    Assistant,
}

impl Role {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [Role; 2] = [Self::User, Self::Assistant];

    /// The role's one spelling, in the log, in an archive and on Meka's own surfaces.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for Role {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|role| role.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a role. Supported: {}",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

impl TryFrom<String> for Role {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<Role> for String {
    fn from(role: Role) -> Self {
        role.name().to_string()
    }
}
/// One item of a tool result: text, an image the tool produced, or a reference to a tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolResultContent {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    /// A deferred tool `tool_load` or `tool_search` made callable, by name. Written only when the
    /// wire declares its deferred tools and loads them in place: Anthropic expands the reference
    /// where it sits, the Responses wire answers with the tool's definition, and a wire that
    /// appends never records one. Everywhere a reader wants text, it reads as the name.
    ToolReference {
        tool_name: String,
    },
    /// Words Meka wrote into a result for the model: a call made without loading its tool's
    /// schema, or what a refused request took out of the result. `note` is the sentence alone;
    /// each wire renders it behind Meka's label, so a reader tells Meka's words from the tool's by
    /// the item rather than by the label.
    HarnessNote {
        note: String,
    },
    /// The line a tool writes ahead of an image it returns, naming where the image came from (a
    /// file read, a page fetched, a render). Its own item rather than a field of the image, so it
    /// stands wherever the image is redacted, repaired or stripped, and a surface that counts the
    /// images leaves it out by its type.
    ImageCaption {
        caption: String,
    },
    /// A result too large to keep inline, saved whole to the scratchpad entry it names: the model
    /// reads where the output went, how large it is and its opening; a surface shows the opening.
    LargeOutput {
        scratchpad_entry: String,
        size_bytes: usize,
        preview: String,
    },
}
/// The most text a tool result keeps inline: a text block larger than this is persisted to the
/// store and replaced with a preview and a handle in the conversation, and a result's recorded
/// [`Change`] keeps no more of its lines than this in the record.
pub(crate) const MAX_INLINE_RESULT_BYTES: usize = 30_000;

/// What a screen draws of a tool result beyond its content, recorded with the result so a replay
/// of the log draws what the live run drew: the log keeps no other trace of how a command ended,
/// what a file held before a call rewrote it, or the list a checklist call left. Never sent to a
/// provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct ResultDisplay {
    /// The code a program the call ran exited with, when it was not 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) exit_code: Option<i32>,
    /// What the call changed in a file, a memory or a skill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) change: Option<Change>,
    /// The checklist as the call's batch left it, every item whatever its status, for a call that
    /// changes the list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) checklist: Option<Vec<crate::checklist::ChecklistItem>>,
    /// The task a call that went to the background runs as, and the sub-agent it drives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) background: Option<BackgroundStart>,
}

/// A call that went to the background, as its result records it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct BackgroundStart {
    /// The task's id, whole or by the prefix a row shows it by, so a reader matches it by prefix.
    pub(crate) task: String,
    /// The sub-agent the task drives, for `agent_spawn` and `agent_followup`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) subagent: Option<uuid::Uuid>,
}

impl BackgroundStart {
    /// The task's id as a row shows it.
    pub(crate) fn short_task(&self) -> &str {
        self.task
            .get(..crate::text::ID_PREFIX)
            .unwrap_or(&self.task)
    }
}

impl ResultDisplay {
    /// `self`, or `None` when it records nothing.
    pub(crate) fn recorded(self) -> Option<Self> {
        (self != Self::default()).then_some(self)
    }
}

/// A change to a text as its diff: every changed line with [`Change::CONTEXT_LINES`] on either
/// side of it, numbered in the old text for what was there and in the new for what is, in hunks
/// apart from each other, up to [`MAX_INLINE_RESULT_BYTES`] of them in the record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Change {
    /// Lines the change added, every one counted whether or not its hunk keeps it.
    pub(crate) added: usize,
    /// Lines the change removed, counted the same way.
    pub(crate) removed: usize,
    /// The changed lines with their context, a group of lines near each other to a hunk.
    pub(crate) hunks: Vec<Vec<ChangeLine>>,
}

/// One line of a [`Change`]: its 1-based number and its text, without the line break.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ChangeLine {
    /// Whether the line stayed, went or came.
    pub(crate) kind: ChangeKind,
    /// The line's number in the old text for one that stayed or went, in the new for one that
    /// came; 0 where neither text has it.
    pub(crate) number: usize,
    /// The line as it reads, without its line break.
    pub(crate) text: String,
}

/// Whether a line of a [`Change`] stayed, went or came.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum ChangeKind {
    /// A line on either side of a change, as it was and is.
    Kept,
    /// A line the change took out, numbered in the old text.
    Removed,
    /// A line the change put in, numbered in the new text.
    Added,
}

impl ChangeKind {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [ChangeKind; 3] = [Self::Kept, Self::Removed, Self::Added];

    /// The one spelling of this value, in the stored result and in an archive.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Kept => "kept",
            Self::Removed => "removed",
            Self::Added => "added",
        }
    }
}

impl std::fmt::Display for ChangeKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for ChangeKind {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|kind| kind.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a change line's kind. Supported: {}",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

impl TryFrom<String> for ChangeKind {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<ChangeKind> for String {
    fn from(kind: ChangeKind) -> Self {
        kind.name().to_string()
    }
}

impl Change {
    /// Lines of unchanged text kept on either side of a change, so a short preview holds the line
    /// before, the line taken out and the line put in.
    pub(crate) const CONTEXT_LINES: usize = 1;
    /// How long the diff may search before it settles for a coarser one: it runs as a call's
    /// result is recorded, and two large texts that differ throughout would otherwise hold the call
    /// up for as long as the search takes.
    const DIFF_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);
    /// What a line costs in the record beside its text: its kind and its number, as the log
    /// writes them, which a line of a few characters or none is mostly made of.
    const LINE_FIELDS_BYTES: usize = 40;

    /// The change from `old` to `new`, either of which is absent for an entry the change created
    /// or deleted.
    pub(crate) fn between(old: Option<&str>, new: Option<&str>) -> Self {
        use similar::{ChangeTag, TextDiff};
        // Lines end at `\n` alone, as `file_read` numbers them; the diff's own splitter also ends
        // one at a lone `\r`.
        let old_lines: Vec<&str> = old.unwrap_or("").split_inclusive('\n').collect();
        let new_lines: Vec<&str> = new.unwrap_or("").split_inclusive('\n').collect();
        let diff = TextDiff::configure()
            .timeout(Self::DIFF_TIMEOUT)
            .diff_slices(&old_lines, &new_lines);
        let (mut added, mut removed) = (0, 0);
        for change in diff.iter_all_changes() {
            match change.tag() {
                ChangeTag::Insert => added += 1,
                ChangeTag::Delete => removed += 1,
                ChangeTag::Equal => {}
            }
        }
        // The hunks keep what a result keeps inline: what a screen shows of a change is recorded
        // with every result, so a rewrite of a large file puts that file in the log once, in the
        // call, rather than again in what it replaced. Lines past it are counted.
        let mut budget = MAX_INLINE_RESULT_BYTES;
        let mut hunks: Vec<Vec<ChangeLine>> = Vec::new();
        // A hunk cut short ends on its last change, and one cut before its first is no hunk.
        let close = |hunk: &mut Vec<ChangeLine>, hunks: &mut Vec<Vec<ChangeLine>>| {
            while hunk
                .last()
                .is_some_and(|line| line.kind == ChangeKind::Kept)
            {
                hunk.pop();
            }
            if !hunk.is_empty() {
                hunks.push(std::mem::take(hunk));
            }
        };
        'groups: for group in diff.grouped_ops(Self::CONTEXT_LINES) {
            let mut hunk: Vec<ChangeLine> = Vec::new();
            for change in group.iter().flat_map(|op| diff.iter_changes(op)) {
                let text = change.value().trim_end_matches(['\n', '\r']);
                let (kind, index) = match change.tag() {
                    ChangeTag::Equal => (ChangeKind::Kept, change.old_index()),
                    ChangeTag::Delete => (ChangeKind::Removed, change.old_index()),
                    ChangeTag::Insert => (ChangeKind::Added, change.new_index()),
                };
                let Some(left) = budget.checked_sub(text.len() + Self::LINE_FIELDS_BYTES) else {
                    // A line around the change is there to place it, so one the room left cannot
                    // hold is let go and the hunk splits there; a changed line it cannot hold
                    // ends the record, since what follows would read as following it.
                    close(&mut hunk, &mut hunks);
                    if kind == ChangeKind::Kept {
                        continue;
                    }
                    break 'groups;
                };
                budget = left;
                hunk.push(ChangeLine {
                    kind,
                    number: index.map_or(0, |index| index + 1),
                    text: text.to_string(),
                });
            }
            if hunk.iter().any(|line| line.kind != ChangeKind::Kept) {
                hunks.push(hunk);
            }
        }
        Self {
            added,
            removed,
            hunks,
        }
    }

    /// The changed lines the change counts and its hunks do not keep, past the budget
    /// [`Self::between`] keeps to.
    pub(crate) fn lines_left_out(&self) -> usize {
        let kept = self
            .hunks
            .iter()
            .flatten()
            .filter(|line| line.kind != ChangeKind::Kept)
            .count();
        (self.added + self.removed).saturating_sub(kept)
    }
}
/// What a thinking block carries back to the wire that produced it, and which wire that is.
///
/// The wires that emit reasoning hand back different things, and the difference decides both
/// what may be replayed and what the readable half is worth. Held as one nullable `signature`,
/// wrong states are representable: a `chatgpt-subscription` session's `encrypted_content` resumed
/// under Claude is replayed verbatim as Claude's `signature`, a blob from the wrong cryptosystem
/// presented as authentication for text it does not authenticate, and the other direction opens
/// the moment the Responses encoder replays reasoning. Naming the shapes makes both a type error
/// instead of a thing to remember.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OpaqueReasoning {
    /// Anthropic. The reasoning is in the block's parts; this authenticates it, and the API wants
    /// both back together.
    Signed { signature: String },
    /// The Responses API. This *is* the reasoning, sealed, and the block's parts hold only the
    /// summary the server chose to show. Replayed under `id` when the server issued one.
    Sealed {
        encrypted_content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The origin of the endpoint that sealed it, `scheme://host[:port]`: only that endpoint
        /// can open it, so it goes back there alone. `None` where nothing records which endpoint
        /// it was, which sends it nowhere. Always written, so a block without the key is one the
        /// migration has not reached.
        #[serde(deserialize_with = "Option::deserialize")]
        origin: Option<String>,
    },
    /// Chat Completions. The reasoning is the block's parts, and nothing else goes back with it.
    Verbatim,
}
/// Why the loop wrote a nudge: the reply it answers had no visible text, or left checklist items
/// open. What a surface shows for a replayed nudge depends on it, and the wire carries it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum NudgeKind {
    VisibleReply,
    Checklist,
}

impl NudgeKind {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [NudgeKind; 2] = [Self::VisibleReply, Self::Checklist];

    /// The kind's one spelling, in the stored block, on the wire and in an archive.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::VisibleReply => "visible_reply",
            Self::Checklist => "checklist",
        }
    }
}

impl std::fmt::Display for NudgeKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for NudgeKind {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|kind| kind.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a nudge kind. Supported: {}",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

impl TryFrom<String> for NudgeKind {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<NudgeKind> for String {
    fn from(kind: NudgeKind) -> Self {
        kind.name().to_string()
    }
}

/// How an item reaches the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum InboxClass {
    /// Injected into the running turn at its next round boundary.
    Steer,
    /// Waits for the running turn to end.
    Followup,
    /// Cuts the answer the running turn is streaming and is read right after what streamed; read
    /// at the round boundary, like a steer, while tools run.
    Interrupt,
}

impl InboxClass {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [InboxClass; 3] = [Self::Steer, Self::Followup, Self::Interrupt];

    /// The one spelling of this value: the `class` column and the HTTP body.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Steer => "steer",
            Self::Followup => "followup",
            Self::Interrupt => "interrupt",
        }
    }

    fn supported() -> String {
        Self::ALL
            .iter()
            .map(|class| class.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl std::fmt::Display for InboxClass {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for InboxClass {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|class| class.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not an inbox class. Supported: {}",
                    Self::supported()
                )
            })
    }
}

impl TryFrom<String> for InboxClass {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<InboxClass> for String {
    fn from(class: InboxClass) -> Self {
        class.name().to_string()
    }
}

/// What a [`ContentBlock::Invocation`] invoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum InvocationKind {
    /// A skill, named as it is installed.
    Skill,
    /// An MCP server's prompt, named `server:prompt`.
    McpPrompt,
}

impl InvocationKind {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [InvocationKind; 2] = [Self::Skill, Self::McpPrompt];

    /// The one spelling of this value, in the stored block and on the wire.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Skill => "skill",
            Self::McpPrompt => "mcp_prompt",
        }
    }
}

impl std::fmt::Display for InvocationKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for InvocationKind {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|kind| kind.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not an invocation kind. Supported: {}",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

impl InvocationKind {
    /// The stage direction a history says where a turn opened on an invocation of `name` begins,
    /// live and on a replay alike; the words typed with it follow as the person's. The name is
    /// sanitized because an MCP prompt's is the server's.
    pub(crate) fn direction(self, name: &str) -> String {
        let name = crate::text::sanitize_to_line(name, crate::text::NAME_WIDTH);
        match self {
            Self::Skill => format!("Running skill '{name}'."),
            Self::McpPrompt => format!("Running prompt '{name}'."),
        }
    }
}

/// A user message's words with each resource an editor attached among them shown as its mention,
/// `@name`, in the words' block: an editor sends the words around a mention as blocks of their
/// own, and drawn apart they would read as separate prompts. A note Meka wrote stays a block of its
/// own, as it is drawn without resources.
pub(crate) fn mentions_inline(content: &[ContentBlock]) -> Vec<ContentBlock> {
    let mut joined: Vec<ContentBlock> = Vec::new();
    for block in content {
        let piece = match block {
            ContentBlock::Text { text, .. } => text.clone(),
            ContentBlock::Resource { name, .. } => format!("@{name}"),
            other => {
                joined.push(other.clone());
                continue;
            }
        };
        match joined.last_mut() {
            Some(ContentBlock::Text { text, .. }) => {
                text.push_str(&piece);
            }
            _ => joined.push(ContentBlock::Text {
                text: piece,
                provider_fields: None,
            }),
        }
    }
    joined
}

impl TryFrom<String> for InvocationKind {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<InvocationKind> for String {
    fn from(kind: InvocationKind) -> Self {
        kind.name().to_string()
    }
}

/// The fields a provider put on a part of a reply beyond those Meka reads, verbatim.
pub(crate) type ProviderFields = serde_json::Map<String, serde_json::Value>;

/// What a model sent as a call's arguments when Meka could not read it as a JSON object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct UnparsedArguments {
    /// The argument text exactly as the model sent it. Absent on a call recorded before the text
    /// was kept, which recorded the error alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<String>,
    /// Why the text is not arguments, in the words the call was refused with.
    pub(crate) error: String,
}

/// A call's arguments as the text the OpenAI wires send, whose spec defines them as the string the
/// model generated: what the model sent when Meka could not read it, else the arguments
/// serialized.
pub(crate) fn call_arguments_text(
    input: &serde_json::Value,
    unparsed: Option<&UnparsedArguments>,
) -> String {
    match unparsed.and_then(|unparsed| unparsed.text.as_deref()) {
        Some(text) => text.to_string(),
        None => input.to_string(),
    }
}

/// One block of a message, in the shape the session store persists and every provider maps from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ContentBlock {
    Text {
        text: String,
        /// The part's fields nothing else holds, verbatim: what Meka records of a reply as it
        /// came, and sends back where the wire's spec defines a field (a Responses message's
        /// `phase`). Every block that stands for a provider's part carries the same.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_fields: Option<ProviderFields>,
    },
    /// Words a model returned in the field its provider keeps for declining (OpenAI's `refusal`),
    /// rather than as its answer. Its own block because the provider typed it: the OpenAI wires
    /// send it back in that field, the others as text.
    Refusal {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_fields: Option<ProviderFields>,
    },
    /// What a provider returned that Meka reads no block from, verbatim and in the order it
    /// arrived: an output item of a type Meka does not know, or a reply in a shape its protocol
    /// does not define. Recorded so nothing a model returned is lost; never sent back.
    Unrecognized { parts: Vec<serde_json::Value> },
    /// Words Meka wrote into the model's own turn in the user's role: the nudge after a reply
    /// that stopped too soon. Its own block, so a nudge is known by its shape rather than its
    /// words, every provider renders it as text, every reader that shows what a person said can
    /// skip it, and an inbox item that arrives while the nudged round runs can ride behind it as a
    /// `Text` block and still be the person's. A user message carries zero or one of these, first.
    Nudge { kind: NudgeKind, text: String },
    /// Words Meka wrote into a user message in place of what it removed: an image the provider
    /// rejected, an image whose bytes disagree with their declared type, an image the request
    /// budget could not carry. `note` is the sentence alone; every wire renders it behind Meka's
    /// label, so a reader tells it from the person's words by the block, and a stand-in the replay
    /// recomputes equals the copy a log holds.
    HarnessNote { note: String },
    /// The message a compaction puts in place of the turns it summarized, as the model reads it:
    /// the summary under its heading, the messages it quotes, the request in progress, the
    /// checklist, the context after the compaction and the direction to carry on. Its own block,
    /// so the next compaction and every reader of a person's words pass it over by its type; the
    /// summary as it was written is the boundary's record, which search reads.
    Summary { text: String },
    /// What Meka injected ahead of the user's words for this turn: the permission and environment
    /// context, world state, budget, background outcomes and the resume notice. Its own
    /// block, so every reader that shows what the user typed can skip it, and every provider
    /// renders it as text ahead of the words. A user message carries zero or one of these, first.
    TurnContext { text: String },
    /// An inbox item as the model read it: `text` is the header Meka wrote with the person's
    /// words under it, `body` the words alone. Its own block, so every reader shows the words as
    /// the person's, labeled by `class` and `source` rather than read back from the header, and a
    /// replay builds the prompt the live screen showed. `item_id` and `class` are absent on an
    /// item whose id and class are not known.
    Inbox {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        item_id: Option<uuid::Uuid>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        class: Option<InboxClass>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        body: String,
        text: String,
    },
    /// A scheduled job's fire as the model read it: `text` is the header, the job's words and the
    /// gate's output when it had one, `prompt` the job's words alone. Its own block, so a replay
    /// says the stage direction the live screen said rather than showing a prompt nobody typed,
    /// and a title or a search reads the job's words.
    Scheduled {
        job_id: String,
        prompt: String,
        text: String,
    },
    /// Finished background work as the model read it, the tasks `task_ids` name. Its own block, so
    /// a reader tells it from the person's words and a replay says which tasks reported, as the
    /// live screen did.
    BackgroundReport { task_ids: Vec<String>, text: String },
    /// A skill or an MCP prompt somebody invoked by name, as the model read it: `text` is what the
    /// invocation expanded to, behind the words typed with it for a skill (an MCP prompt renders
    /// its arguments itself), and `directive` those words alone, a file an editor attached among
    /// them named in place by its mention (`@name`), blank when none were typed. Its own block, so
    /// a title, a search and a replay read what was done rather than the body it expanded to. A
    /// user message that carries one carries nothing else that was sent with it but its images.
    Invocation {
        kind: InvocationKind,
        name: String,
        directive: String,
        text: String,
    },
    /// A resource an editor attached among the person's words, as the model read it: `text` is
    /// the resource's tag, with its contents inside when the editor sent them, and `uri` and
    /// `name` are what an editor shows of it. Its own block, in the place it was attached, so a
    /// replay shows the mention rather than the contents and a title reads the words around it.
    Resource {
        uri: String,
        name: String,
        text: String,
    },
    /// Image supplied as *input* (e.g. an ACP client's @-mention or pasted screenshot). Distinct
    /// from a tool result's image, which travels inside [`ContentBlock::ToolResult`] as a
    /// [`ToolResultContent::Image`].
    Image { source: ImageSource },
    Thinking {
        /// The readable half, and only the readable half, in the parts the provider returned it
        /// in: one part of reasoning on Claude's wire and DeepSeek's, the summary and reasoning
        /// parts of a Responses item as the item held them. Read as one text through
        /// [`reasoning_text`].
        parts: Vec<ReasoningPart>,
        /// What the provider wants handed back to carry this reasoning into the next request.
        /// `None` when it gave nothing to carry, which makes the block display-only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        opaque: Option<OpaqueReasoning>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_fields: Option<ProviderFields>,
    },
    /// Encrypted reasoning the API declines to return in the clear (the `redact-thinking` beta).
    /// `data` is opaque: it cannot be read, only replayed verbatim on later turns so the model can
    /// continue its prior reasoning chain. Distinct from a [`ContentBlock::Thinking`] with empty
    /// text, which carries a `signature` instead of `data`.
    RedactedThinking {
        data: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_fields: Option<ProviderFields>,
    },
    ToolUse {
        id: String,
        name: String,
        /// The arguments, an object; `{}` when `unparsed` holds what the model sent instead.
        input: serde_json::Value,
        /// What the model sent as the arguments when it was not a JSON object, and why it could
        /// not be read. The call is refused, so it never runs on arguments it did not give.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unparsed: Option<UnparsedArguments>,
        /// Every field the provider put on the call beyond its id, name and arguments, verbatim
        /// (Gemini's `extra_content` on Chat Completions): recorded as the call came, sent back
        /// where an extension Meka supports defines it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_fields: Option<ProviderFields>,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultContent>,
        is_error: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display: Option<ResultDisplay>,
    },
}
impl ContentBlock {
    /// A thinking block when it carries replayable state: visible text or something opaque (a
    /// signature, sealed reasoning). Under `redact-thinking` or display updates the text is empty
    /// but the signature must survive to continue the chain; an empty unsigned block is nothing
    /// to replay. One rule for the streamed fold and the whole-reply parser, so a resumed
    /// transcript never differs from the live turn by a block one of them kept.
    pub(crate) fn replayable_thinking(
        parts: Vec<ReasoningPart>,
        opaque: Option<OpaqueReasoning>,
        provider_fields: Option<ProviderFields>,
    ) -> Option<Self> {
        // Every part as returned, an empty one included; a block of empty parts and nothing to
        // carry back holds nothing.
        let worded = parts.iter().any(|part| !part.text.is_empty());
        (worded || opaque.is_some()).then_some(Self::Thinking {
            parts,
            opaque,
            provider_fields,
        })
    }

    /// A thinking block of one part of reasoning, `text`.
    pub(crate) fn reasoning(text: impl Into<String>, opaque: Option<OpaqueReasoning>) -> Self {
        Self::Thinking {
            parts: vec![ReasoningPart::reasoning(text)],
            opaque,
            provider_fields: None,
        }
    }

    /// The block as one searchable string, labeled by what it is where it is not plain text, or
    /// `None` for a block that holds no words: an image, and the context block Meka writes ahead
    /// of a turn. The one definition of a block's text for search, read by the
    /// `conversation_search` tool over every block; the session search index reads the person's
    /// words alone, by [`Self::spoken_words`].
    pub(crate) fn search_text(&self) -> Option<String> {
        match self {
            ContentBlock::Text { text, .. }
            | ContentBlock::Inbox { text, .. }
            | ContentBlock::Scheduled { text, .. }
            | ContentBlock::BackgroundReport { text, .. }
            | ContentBlock::Summary { text }
            | ContentBlock::Resource { text, .. } => Some(text.clone()),
            ContentBlock::Invocation {
                kind, name, text, ..
            } => Some(format!("[{kind}: {name}] {text}")),
            ContentBlock::TurnContext { .. }
            | ContentBlock::Nudge { .. }
            | ContentBlock::HarnessNote { .. }
            | ContentBlock::Image { .. } => None,
            ContentBlock::Refusal { text, .. } => Some(format!("[refusal] {text}")),
            ContentBlock::Thinking { parts, .. } => {
                Some(format!("[thinking] {}", reasoning_text(parts)))
            }
            ContentBlock::RedactedThinking { .. } => Some("[redacted thinking]".to_string()),
            ContentBlock::Unrecognized { .. } => None,
            ContentBlock::ToolUse { name, input, .. } => {
                Some(format!("[tool call: {name}] {input}"))
            }
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                let label = if *is_error {
                    "[tool result (error)]"
                } else {
                    "[tool result]"
                };
                Some(format!(
                    "{label} {}",
                    ContentBlock::tool_result_text_content(content)
                ))
            }
        }
    }

    /// The words a person sent in this block, for a title and the search index: typed text, an
    /// inbox item's body under the header Meka wrote, a scheduled job's prompt under its fire
    /// header, the words typed with an invocation (a file attached among them by its mention, as
    /// the editor showed it) or, with none, the name it invoked. `None` for what Meka wrote, a
    /// stand-in it put where content was included, a resource block of its own, and for a block
    /// that holds no words.
    pub(crate) fn spoken_words(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            ContentBlock::Inbox { body, .. } => Some(body.as_str()),
            ContentBlock::Scheduled { prompt, .. } => Some(prompt.as_str()),
            ContentBlock::Invocation {
                name, directive, ..
            } => Some(if crate::text::is_blank(directive) {
                name.as_str()
            } else {
                directive.as_str()
            }),
            _ => None,
        }
    }

    /// The block as the text a wire sends for it, for every block a wire sends as text: the words,
    /// what Meka wrote beside them (a note behind its label), and what arrived for the model.
    /// `None` for a block that is not text on the wire: an image, reasoning, a call or a result.
    /// The one definition every encoder renders a user message's text from, so a new kind of block
    /// cannot reach one wire and miss another.
    pub(crate) fn model_text(&self) -> Option<std::borrow::Cow<'_, str>> {
        use std::borrow::Cow;
        match self {
            ContentBlock::Text { text, .. }
            | ContentBlock::TurnContext { text }
            | ContentBlock::Nudge { text, .. }
            | ContentBlock::Summary { text }
            | ContentBlock::Inbox { text, .. }
            | ContentBlock::Scheduled { text, .. }
            | ContentBlock::BackgroundReport { text, .. }
            | ContentBlock::Invocation { text, .. }
            | ContentBlock::Resource { text, .. }
            | ContentBlock::Refusal { text, .. } => Some(Cow::Borrowed(text)),
            ContentBlock::HarnessNote { note } => Some(Cow::Owned(harness_note_text(note))),
            ContentBlock::Image { .. }
            | ContentBlock::Unrecognized { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
            | ContentBlock::ToolUse { .. }
            | ContentBlock::ToolResult { .. } => None,
        }
    }

    /// A tool result as one string, for a wire that takes a result as text and for a reader that
    /// wants its words: each item as [`ToolResultContent::wire_part`] renders it, an image as
    /// `[Image]`, a note set apart from what comes before it by a blank line.
    pub(crate) fn tool_result_text_content(content: &[ToolResultContent]) -> String {
        let mut joined = String::new();
        for item in content {
            if matches!(item, ToolResultContent::HarnessNote { .. }) && !joined.is_empty() {
                joined.push_str("\n\n");
            }
            match item.wire_part() {
                ResultPart::Text(text) => joined.push_str(&text),
                ResultPart::Image(_) => joined.push_str("[Image]"),
                ResultPart::ToolReference(tool_name) => joined.push_str(tool_name),
            }
        }
        joined
    }
}

/// One part of a thinking block, as its provider returned it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ReasoningPart {
    pub(crate) kind: ReasoningKind,
    pub(crate) text: String,
}

impl ReasoningPart {
    /// A part of reasoning itself.
    pub(crate) fn reasoning(text: impl Into<String>) -> Self {
        Self {
            kind: ReasoningKind::Reasoning,
            text: text.into(),
        }
    }

    /// A part of a summary the provider wrote of reasoning it keeps.
    #[cfg(test)]
    pub(crate) fn summary(text: impl Into<String>) -> Self {
        Self {
            kind: ReasoningKind::Summary,
            text: text.into(),
        }
    }
}

/// What a part of a thinking block is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum ReasoningKind {
    /// The reasoning itself.
    Reasoning,
    /// A summary the provider wrote of reasoning it keeps.
    Summary,
}

impl ReasoningKind {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [ReasoningKind; 2] = [Self::Reasoning, Self::Summary];

    /// The value's one spelling.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::Summary => "summary",
        }
    }
}

impl std::fmt::Display for ReasoningKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for ReasoningKind {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|kind| kind.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a reasoning part. Supported: {}",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

impl TryFrom<String> for ReasoningKind {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<ReasoningKind> for String {
    fn from(kind: ReasoningKind) -> Self {
        kind.name().to_string()
    }
}

/// A thinking block's parts as one text for a reader, a blank line between parts: the one join
/// every surface that shows or searches reasoning reads it through. Nothing sends it.
pub(crate) fn reasoning_text(parts: &[ReasoningPart]) -> String {
    parts
        .iter()
        .map(|part| part.text.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The parts' text with nothing between them, as a wire that streamed them into one field sent
/// them: what goes back in that field.
pub(crate) fn reasoning_as_streamed(parts: &[ReasoningPart]) -> String {
    parts.iter().map(|part| part.text.as_str()).collect()
}

/// How Meka marks its own words to the model wherever they sit beside someone else's: in a user
/// message, where the model would otherwise take them for the person's, and in a tool result,
/// where it would take them for the tool's. Rendered ahead of a [`ContentBlock::HarnessNote`] or
/// a [`ToolResultContent::HarnessNote`] at every wire, and read back nowhere.
///
/// Names the harness in words rather than only by product, because a bare `[Meka]` is legible
/// only to a model that recalls what Meka is from the system prompt. `[system]` would read better
/// still and is the wrong choice: nothing strips markers out of a tool result, so a fetched page
/// or an MCP server's output can contain any string this does, and a marker models already obey
/// on sight is the worst one to make load-bearing.
const HARNESS_LABEL: &str = "[Meka harness]";

/// A harness note as the model reads it: the sentence behind Meka's label.
pub(crate) fn harness_note_text(note: &str) -> String {
    format!("{HARNESS_LABEL} {note}")
}

/// `text` made safe to put inside a harness note: a provider's rejection quoted in one may not
/// forge the label, since the label is what tells the model the sentence around it is Meka's.
///
/// Two steps, because neither is enough alone: [`crate::text::sanitize_text`] strips control
/// characters and bidi overrides but keeps `\n`, so a body with a newline and then the label passes
/// through it intact. This needs no hostile gateway: a provider echoing the request body back
/// reproduces every note the conversation carries.
pub(crate) fn scrub_for_harness_note(text: &str) -> String {
    crate::text::sanitize_text(text).replace(HARNESS_LABEL, "[removed]")
}

/// The note that stands where the request budget took an image out, so the body that fit is the
/// body every later request sends.
pub(crate) const BUDGET_REDACTION_NOTE: &str =
    "An image here was removed to fit the request size budget.";

/// One item of a tool result as a wire sends it. Meka's own items (a note, a caption, a large
/// output) are text here, each rendered by the one function that owns its shape.
pub(crate) enum ResultPart<'a> {
    Text(std::borrow::Cow<'a, str>),
    Image(&'a ImageSource),
    ToolReference(&'a str),
}

impl ToolResultContent {
    /// The item as a wire sends it.
    pub(crate) fn wire_part(&self) -> ResultPart<'_> {
        use std::borrow::Cow;
        match self {
            Self::Text { text } => ResultPart::Text(Cow::Borrowed(text)),
            Self::Image { source } => ResultPart::Image(source),
            Self::ToolReference { tool_name } => ResultPart::ToolReference(tool_name),
            Self::HarnessNote { note } => ResultPart::Text(Cow::Owned(harness_note_text(note))),
            Self::ImageCaption { caption } => ResultPart::Text(Cow::Owned(format!("[{caption}]"))),
            Self::LargeOutput {
                scratchpad_entry,
                size_bytes,
                preview,
            } => ResultPart::Text(Cow::Owned(large_output_text(
                scratchpad_entry,
                *size_bytes,
                preview,
            ))),
        }
    }
}

/// A [`ToolResultContent::LargeOutput`] as the model reads it: where the output went, its size,
/// how to read it, and its opening, with `...` when the opening stops short of the whole.
fn large_output_text(scratchpad_entry: &str, size_bytes: usize, preview: &str) -> String {
    let mut text = format!(
        "<large-output name=\"{scratchpad_entry}\" size=\"{size_bytes}\">\n\
         Output too large ({}). Read it with `scratchpad_read`: a read returns as much as fits in \
         the context window and says where to continue; `start` and `end` name a line range.\n\n\
         Preview (first {} bytes):\n\
         {preview}",
        crate::text::format_size(size_bytes),
        preview.len(),
    );
    if preview.len() < size_bytes {
        text.push_str("\n...");
    }
    text.push_str("\n</large-output>");
    text
}
/// One message of the conversation: who sent it and what it carries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Message {
    pub(crate) role: Role,
    pub(crate) content: Vec<ContentBlock>,
    /// The fields a provider put on the message itself that nothing else holds, verbatim: a Chat
    /// Completions message's `reasoning_details`, `annotations` or `audio`. A wire whose message
    /// is its response, or whose parts are its output, has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) provider_fields: Option<ProviderFields>,
}
impl Message {
    /// Whether this message is the envelope of a tool round's results, the user-role message that
    /// continues the turn the model is still working on. A wire encoder shapes the message by it.
    pub(crate) fn carries_tool_results(&self) -> bool {
        self.role == Role::User
            && self
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
    }

    /// Whether this message is a nudge: words Meka wrote into the model's own turn, answering a
    /// reply that stopped too soon, with no visible text or with checklist items open. Known by
    /// its shape, a [`ContentBlock::Nudge`] first, whatever else rides behind it.
    pub(crate) fn is_nudge(&self) -> bool {
        self.nudge().is_some()
    }

    /// The nudge this message carries, when it is one: why it was written and its words.
    pub(crate) fn nudge(&self) -> Option<(NudgeKind, &str)> {
        if self.role != Role::User {
            return None;
        }
        match self.content.first() {
            Some(ContentBlock::Nudge { kind, text }) => Some((*kind, text.as_str())),
            _ => None,
        }
    }

    /// The message the loop appends to send the model back to work.
    pub(crate) fn nudge_message(kind: NudgeKind, text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Nudge {
                kind,
                text: text.into(),
            }],
            provider_fields: None,
        }
    }

    /// Whether this message is shaped as a prompt to the model: user role, words rather than
    /// results. The user's turn, a scheduled job's, or a nudge. A compaction splits on it and keeps
    /// a trailing one, since the model has it to answer whoever wrote it.
    pub(crate) fn is_prompt(&self) -> bool {
        self.role == Role::User && !self.carries_tool_results()
    }

    /// Whether this message opens a turn: a prompt that is not a nudge. A nudge continues the turn
    /// it answers, so the one spelling of the rule is here: a rewind counts turns by it, a turn
    /// label starts at it, a worker listing counts by it, and a history shows it as what was
    /// asked.
    pub(crate) fn opens_turn(&self) -> bool {
        self.is_prompt() && !self.is_nudge()
    }

    /// A user message carrying only `text`.
    pub(crate) fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.into(),
                provider_fields: None,
            }],
            provider_fields: None,
        }
    }

    /// [`Self::turn_opening`] for a typed turn, as the tests build one: the context block, the
    /// words when there are any, then the images.
    #[cfg(test)]
    pub(crate) fn user_turn(
        context: impl Into<String>,
        words: impl Into<String>,
        images: Vec<ImageSource>,
    ) -> Self {
        let words = words.into();
        let spoken = (!words.is_empty()).then_some(ContentBlock::Text {
            text: words,
            provider_fields: None,
        });
        Self::turn_opening(context, spoken.into_iter().collect(), images)
    }

    /// The user message a turn appends: Meka's context block, then what follows it as blocks, a
    /// background report and the words as typed or a scheduled job's fire, each absent when the
    /// turn has none, then any input images. The words are left out when there are none, so
    /// [`Self::text_content`] stays what the user typed.
    pub(crate) fn turn_opening(
        context: impl Into<String>,
        after_context: Vec<ContentBlock>,
        images: Vec<ImageSource>,
    ) -> Self {
        let mut content = vec![ContentBlock::TurnContext {
            text: context.into(),
        }];
        content.extend(after_context);
        content.extend(
            images
                .into_iter()
                .map(|source| ContentBlock::Image { source }),
        );
        Self {
            role: Role::User,
            content,
            provider_fields: None,
        }
    }

    /// A conversation holding every kind of content Meka writes into one, for a wire's tests: a
    /// compaction's summary, a note beside a person's words, a call whose arguments were not an
    /// object, and a result carrying the tool's text, an appended note, a caption and a large
    /// output's reference.
    #[cfg(test)]
    pub(crate) fn typed_content_for_test() -> Vec<Self> {
        vec![
            Self {
                role: Role::User,
                content: vec![ContentBlock::Summary {
                    text: "[Conversation summary from session compaction]\n\nthe summary"
                        .to_string(),
                }],
                provider_fields: None,
            },
            Self {
                role: Role::User,
                content: vec![
                    ContentBlock::TurnContext {
                        text: "<context/>".to_string(),
                    },
                    ContentBlock::Text {
                        text: "the words".to_string(),
                        provider_fields: None,
                    },
                    ContentBlock::HarnessNote {
                        note: "the note".to_string(),
                    },
                ],
                provider_fields: None,
            },
            Self {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "file_read".to_string(),
                    input: serde_json::json!({}),
                    unparsed: Some(UnparsedArguments {
                        text: Some("{\"path\": ".to_string()),
                        error: "invalid JSON arguments: EOF while parsing".to_string(),
                    }),
                    provider_fields: None,
                }],
                provider_fields: None,
            },
            Self {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![
                        ToolResultContent::Text {
                            text: "the output".to_string(),
                        },
                        ToolResultContent::HarnessNote {
                            note: "the schema note".to_string(),
                        },
                        ToolResultContent::ImageCaption {
                            caption: "Image: a.png".to_string(),
                        },
                        ToolResultContent::LargeOutput {
                            scratchpad_entry: "the_entry".to_string(),
                            size_bytes: 40_000,
                            preview: "the opening".to_string(),
                        },
                    ],
                    is_error: true,
                    display: None,
                }],
                provider_fields: None,
            },
            Self::assistant_text("the answer"),
        ]
    }

    /// An opening as an editor's `/review the parser` with a file attached among words sends it:
    /// the context block, the invocation, then the words and the resource in their places.
    #[cfg(test)]
    pub(crate) fn invoked_with_a_resource_for_test() -> Self {
        Self {
            role: Role::User,
            content: vec![
                ContentBlock::TurnContext {
                    text: "<context/>".to_string(),
                },
                ContentBlock::Invocation {
                    kind: InvocationKind::Skill,
                    name: "review".to_string(),
                    directive: "the parser".to_string(),
                    text: "the parser\n\nthe skill's body".to_string(),
                },
                ContentBlock::Text {
                    text: "compare ".to_string(),
                    provider_fields: None,
                },
                ContentBlock::Resource {
                    uri: "file:///a.rs".to_string(),
                    name: "a.rs".to_string(),
                    text: "<resource uri=\"file:///a.rs\">fn a() {}</resource>".to_string(),
                },
                ContentBlock::Text {
                    text: " with the old one".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        }
    }

    /// User message carrying a text block followed by zero or more input images; `images` empty
    /// yields the same shape as [`Message::user`].
    #[cfg(test)]
    pub(crate) fn user_with_images(text: impl Into<String>, images: Vec<ImageSource>) -> Self {
        let mut content = vec![ContentBlock::Text {
            text: text.into(),
            provider_fields: None,
        }];
        content.extend(
            images
                .into_iter()
                .map(|source| ContentBlock::Image { source }),
        );
        Self {
            role: Role::User,
            content,
            provider_fields: None,
        }
    }

    /// An assistant message carrying only `text`.
    pub(crate) fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.into(),
                provider_fields: None,
            }],
            provider_fields: None,
        }
    }

    /// The words: every `Text` block joined, and nothing else. A turn's context block is not
    /// something the user wrote and is left out; [`Self::wire_text`] is the whole.
    pub(crate) fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// The message's blocks as a reader shows them: in a reply, adjacent `Text` blocks are one run
    /// of words joined with nothing, as they streamed, because a wire splits one answer into blocks
    /// for reasons of its own (a citation, an output item) and the split is no paragraph. Every
    /// other block, and every block of a message that is not a reply, is as recorded.
    pub(crate) fn shown_content(&self) -> std::borrow::Cow<'_, [ContentBlock]> {
        let is_text = |block: &ContentBlock| matches!(block, ContentBlock::Text { .. });
        if self.role != Role::Assistant
            || !self.content.windows(2).any(|pair| pair.iter().all(is_text))
        {
            return std::borrow::Cow::Borrowed(&self.content);
        }
        let mut shown: Vec<ContentBlock> = Vec::with_capacity(self.content.len());
        for block in &self.content {
            match (shown.last_mut(), block) {
                (Some(ContentBlock::Text { text: run, .. }), ContentBlock::Text { text, .. }) => {
                    run.push_str(text)
                }
                (_, ContentBlock::Text { text, .. }) => shown.push(ContentBlock::Text {
                    text: text.clone(),
                    provider_fields: None,
                }),
                (_, block) => shown.push(block.clone()),
            }
        }
        std::borrow::Cow::Owned(shown)
    }

    /// Whether the message holds words for a reader: text or a refusal that is not blank.
    pub(crate) fn has_visible_text(&self) -> bool {
        self.content.iter().any(|block| match block {
            ContentBlock::Text { text, .. } | ContentBlock::Refusal { text, .. } => {
                !crate::text::is_blank(text)
            }
            _ => false,
        })
    }

    /// Whether this is a reply that answers nothing: no words to read and no call. Its reasoning,
    /// when it has any, is not an answer.
    pub(crate) fn holds_no_answer(&self) -> bool {
        self.role == Role::Assistant
            && !self.has_visible_text()
            && !self
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
    }

    /// The words a reply that ended as a refusal declined in: the refusal it returned, else the
    /// text it wrote before the stop, the blocks a blank line apart. `None` when it wrote nothing.
    pub(crate) fn refusal_words(&self) -> Option<String> {
        let content = self.shown_content();
        let words = |refusal: bool| {
            content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Refusal { text, .. } if refusal => Some(text.trim()),
                    ContentBlock::Text { text, .. } if !refusal => Some(text.trim()),
                    _ => None,
                })
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        [words(true), words(false)]
            .into_iter()
            .find(|words| !words.is_empty())
    }

    /// The words a user message sent as the model read them, a blank line between blocks: typed
    /// text, inbox items, fires, invocations and resources, without the context block, a nudge,
    /// a report or a note Meka wrote beside them. Assistant text keeps [`Self::text_content`]'s
    /// join, since its blocks are one answer split around tool calls.
    pub(crate) fn text_paragraphs(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. }
                | ContentBlock::Inbox { text, .. }
                | ContentBlock::Scheduled { text, .. }
                | ContentBlock::Invocation { text, .. }
                | ContentBlock::Resource { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// A user message's words as a history shows them, a blank line between: what was typed with
    /// each attached resource as its mention among the words, an invocation as what was invoked
    /// and the words typed with it, an inbox item and a fire as the model read them. Never an
    /// invocation's body or a resource's contents, which are the model's to read.
    pub(crate) fn shown_words(&self) -> String {
        mentions_inline(&self.content)
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. }
                | ContentBlock::Inbox { text, .. }
                | ContentBlock::Scheduled { text, .. } => Some(text.clone()),
                ContentBlock::Invocation {
                    kind,
                    name,
                    directive,
                    ..
                } => Some(if crate::text::is_blank(directive) {
                    format!("*{}*", kind.direction(name))
                } else {
                    format!("*{}*\n\n{}", kind.direction(name), directive.trim())
                }),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Everything a provider renders as text, in order, a blank line between: the turn's context
    /// block or a nudge, then the words. What the two were joined with when they were one string,
    /// for the wires that take a user message as a single string.
    pub(crate) fn wire_text(&self) -> String {
        self.content
            .iter()
            .filter_map(ContentBlock::model_text)
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// A copy of this message with every [`ContentBlock::ToolUse`] removed. Used when persisting a
    /// turn that was interrupted before its tools ran: keeping the `tool_use` blocks would orphan
    /// them (no matching `tool_result`) and the provider would reject the next request.
    pub(crate) fn without_tool_use(&self) -> Message {
        Message {
            role: self.role,
            content: self
                .content
                .iter()
                .filter(|block| !matches!(block, ContentBlock::ToolUse { .. }))
                .cloned()
                .collect(),
            provider_fields: self.provider_fields.clone(),
        }
    }

    /// Whether this message carries thinking Claude checks when it is sent back: a block under
    /// Claude's signature, or a redacted one.
    pub(crate) fn carries_claude_thinking(&self) -> bool {
        self.content.iter().any(|block| {
            matches!(
                block,
                ContentBlock::Thinking {
                    opaque: Some(OpaqueReasoning::Signed { .. }),
                    ..
                } | ContentBlock::RedactedThinking { .. }
            )
        })
    }

    /// A copy of this message without the thinking Claude checks when it is sent back, or `None`
    /// when it carries none: a signed block keeps its readable text and loses its signature, which
    /// is what keeps it off the Claude wire, and a redacted block, all signature, goes whole. What
    /// a request rejected over a thinking block's signature is retried with.
    pub(crate) fn without_claude_thinking(&self) -> Option<Message> {
        if !self.carries_claude_thinking() {
            return None;
        }
        let content = self
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Thinking {
                    parts,
                    opaque: Some(OpaqueReasoning::Signed { .. }),
                    provider_fields,
                } => {
                    ContentBlock::replayable_thinking(parts.clone(), None, provider_fields.clone())
                }
                ContentBlock::RedactedThinking { .. } => None,
                other => Some(other.clone()),
            })
            .collect();
        Some(Message {
            role: self.role,
            content,
            provider_fields: self.provider_fields.clone(),
        })
    }

    #[cfg(test)]
    pub(crate) fn tool_uses(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|block| matches!(block, ContentBlock::ToolUse { .. }))
            .collect()
    }
}

/// What becomes of a turn's prompt when the turn ends, failed or canceled, before the model
/// produced anything.
///
/// The prompt is persisted eagerly, before the first provider call, so a crash mid-roundtrip cannot
/// lose it. That is right when losing it would be losing something, and wrong when the prompt will
/// simply be produced again. Whoever submits the turn knows which, so each host states it through
/// `TurnInput::retaining`: the scheduler per job, the HTTP API from the client's own request.
///
/// One spelling per value, [`Self::name`], is what `Display`, `FromStr` and serde all go through.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum PromptRetention {
    /// Keep it. A human typed it and can see the error, or it carries something that exists nowhere
    /// else: a background-task outcome, whose row is stamped `delivered` before the turn starts
    /// and handed out again only if the turn is refused before it opens or cannot store it, or a
    /// one-shot job's fire, whose row is retired with the turn.
    #[default]
    Keep,
    /// Withdraw it, because whoever produced it will produce it again: a recurring job on its next
    /// occurrence, whose fire says how many were missed, or a client that resends a failed turn and
    /// said so. Left in place, an outage would deposit one unanswered user message per attempt for
    /// as long as it lasted.
    Withdraw,
}

impl PromptRetention {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [PromptRetention; 2] = [Self::Keep, Self::Withdraw];

    /// The one spelling of this value: what the HTTP turn request takes.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Withdraw => "withdraw",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    fn supported() -> String {
        Self::ALL
            .iter()
            .map(|retention| retention.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl std::fmt::Display for PromptRetention {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for PromptRetention {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|retention| retention.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a prompt retention. Supported: {}",
                    Self::supported()
                )
            })
    }
}

impl TryFrom<String> for PromptRetention {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<PromptRetention> for String {
    fn from(retention: PromptRetention) -> Self {
        retention.name().to_string()
    }
}
/// The whole event log as the Markdown `meka session export` writes, compactions and repairs
/// marked in place, each event with what its provider said about the reply it appends.
pub(crate) fn format_session_as_markdown<'a>(
    session_id: uuid::Uuid,
    events: impl IntoIterator<Item = (&'a Event, Option<&'a ReplyRecord>)>,
    scratchpad_entries: &std::collections::HashMap<String, String>,
) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    writeln!(output, "# Session {session_id}\n").ok();

    // Walk the log as history so the full conversation is exported, including turns a compaction
    // later hid from the model. Each `CompactBoundary` becomes a marker; the turns it summarized
    // stay above it, and the turns it kept are said once, above it, where they were first said.
    for (event, reply) in events {
        match event {
            Event::Append(message) => {
                write_message_markdown(&mut output, message, scratchpad_entries);
                if let Some(label) = reply.and_then(ReplyRecord::unfinished_label) {
                    writeln!(output, "*{label}*\n").ok();
                }
            }
            Event::CompactBoundary {
                summary, record, ..
            } => {
                writeln!(output, "---\n").ok();
                writeln!(output, "<details>").ok();
                writeln!(
                    output,
                    "<summary>Session compaction: {} (the summary the model saw in place of the \
                     turns above)</summary>\n",
                    record.note()
                )
                .ok();
                writeln!(output, "{}\n", summary.wire_text()).ok();
                writeln!(output, "</details>\n").ok();
            }
            Event::Redact { images } => {
                writeln!(
                    output,
                    "*{} image{} above {} redacted to fit the request size budget.*\n",
                    images.len(),
                    if images.len() == 1 { "" } else { "s" },
                    if images.len() == 1 { "was" } else { "were" },
                )
                .ok();
            }
            // Same treatment as a boundary: mark what happened and render the replacement, leaving
            // the superseded messages above it. An export is the record of the session, and a
            // repair (a rewind, a degraded retry, a dropped call a crash cut, a prompt taken back)
            // is the one place where what the model saw and what actually happened diverge.
            Event::Repair {
                replaced_count,
                messages,
            } => {
                writeln!(output, "---\n").ok();
                writeln!(output, "<details>").ok();
                writeln!(
                    output,
                    "<summary>{} message(s) above replaced with {}</summary>\n",
                    replaced_count,
                    if messages.is_empty() {
                        "nothing".to_string()
                    } else {
                        format!("{} message(s)", messages.len())
                    },
                )
                .ok();
                for message in messages {
                    write_message_markdown(&mut output, message, scratchpad_entries);
                }
                writeln!(output, "</details>\n").ok();
            }
            Event::Unsign => {
                writeln!(
                    output,
                    "*The thinking above is no longer sent to the model: Claude rejected a thinking \
                     block's signature.*\n"
                )
                .ok();
            }
            Event::CompactionFailed(failure) => {
                match &failure.error {
                    Some(error) => writeln!(output, "*Compaction failed: {error}*\n").ok(),
                    None => writeln!(output, "*Compaction stopped.*\n").ok(),
                };
                if !failure.memories_written.is_empty() {
                    writeln!(
                        output,
                        "*Its checkpoint saved: {}.*\n",
                        failure.memories_written.join(", ")
                    )
                    .ok();
                }
            }
        }
    }

    output
}
/// Append one message to a Markdown export, with tool calls and results in collapsible blocks.
pub(crate) fn write_message_markdown(
    output: &mut String,
    message: &crate::conversation::Message,
    scratchpad_entries: &std::collections::HashMap<String, String>,
) {
    use std::fmt::Write;

    match message.role {
        crate::conversation::Role::User => {
            // A `User` message is a turn, a nudge Meka wrote into one, or a tool-results envelope;
            // the blocks say which. An inbox item that rode a nudge or a round's results is the
            // person's, under its own heading after what it rode; a background report is Meka's,
            // under Meka's heading wherever it sits.
            let mut reports = message.content.iter().filter_map(|block| match block {
                crate::conversation::ContentBlock::BackgroundReport { text, .. } => {
                    Some(text.as_str())
                }
                crate::conversation::ContentBlock::HarnessNote { note } => Some(note.as_str()),
                _ => None,
            });
            let opens_a_turn = message.opens_turn();
            if let Some((_, text)) = message.nudge() {
                writeln!(output, "## Meka\n").ok();
                writeln!(output, "{text}\n").ok();
            } else if message.carries_tool_results() {
                for block in &message.content {
                    if let crate::conversation::ContentBlock::ToolResult {
                        content, is_error, ..
                    } = block
                    {
                        let label = if *is_error {
                            "Tool result (error)"
                        } else {
                            "Tool result"
                        };
                        writeln!(output, "<details>").ok();
                        writeln!(output, "<summary>{label}</summary>\n").ok();
                        // A large output reads as the entry it went to, where the export has it.
                        let resolved: Vec<ToolResultContent> = content
                            .iter()
                            .map(|item| match item {
                                ToolResultContent::LargeOutput {
                                    scratchpad_entry, ..
                                } => scratchpad_entries.get(scratchpad_entry).map_or_else(
                                    || item.clone(),
                                    |whole| ToolResultContent::Text {
                                        text: whole.clone(),
                                    },
                                ),
                                other => other.clone(),
                            })
                            .collect();
                        let text = ContentBlock::tool_result_text_content(&resolved);
                        writeln!(output, "```\n{text}\n```\n").ok();
                        writeln!(output, "</details>\n").ok();
                    }
                }
            }
            let mut reported = false;
            for report in reports.by_ref() {
                writeln!(output, "## Meka\n").ok();
                writeln!(output, "{report}\n").ok();
                reported = true;
            }
            let words = message.shown_words();
            if !words.is_empty() {
                writeln!(output, "## User\n").ok();
                writeln!(output, "{words}\n").ok();
            } else if opens_a_turn && !reported {
                writeln!(output, "## User\n").ok();
                // A turn opened on an image alone carries no `Text` block, so a heading over
                // nothing is what it would otherwise leave.
                writeln!(
                    output,
                    "*Meka opened this turn with no words from the user.*\n"
                )
                .ok();
            }
        }
        crate::conversation::Role::Assistant => {
            writeln!(output, "## Assistant\n").ok();
            for block in message.shown_content().iter() {
                match block {
                    crate::conversation::ContentBlock::Text { text, .. }
                    | crate::conversation::ContentBlock::Refusal { text, .. } => {
                        writeln!(output, "{text}\n").ok();
                    }
                    crate::conversation::ContentBlock::ToolUse {
                        name,
                        unparsed: Some(unparsed),
                        ..
                    } => {
                        writeln!(output, "<details>").ok();
                        writeln!(output, "<summary>Tool call: {name}</summary>\n").ok();
                        // The arguments as sent; the call's result says why they were refused.
                        if let Some(text) = &unparsed.text {
                            writeln!(output, "```\n{text}\n```\n").ok();
                        }
                        writeln!(output, "</details>\n").ok();
                    }
                    crate::conversation::ContentBlock::ToolUse { name, input, .. } => {
                        let input_pretty = serde_json::to_string_pretty(input)
                            .unwrap_or_else(|_| input.to_string());
                        writeln!(output, "<details>").ok();
                        writeln!(output, "<summary>Tool call: {name}</summary>\n").ok();
                        writeln!(output, "```json\n{input_pretty}\n```\n").ok();
                        writeln!(output, "</details>\n").ok();
                    }
                    crate::conversation::ContentBlock::ToolResult { .. }
                    | crate::conversation::ContentBlock::TurnContext { .. }
                    | crate::conversation::ContentBlock::Nudge { .. }
                    | crate::conversation::ContentBlock::HarnessNote { .. }
                    | crate::conversation::ContentBlock::Summary { .. }
                    | crate::conversation::ContentBlock::Inbox { .. }
                    | crate::conversation::ContentBlock::Scheduled { .. }
                    | crate::conversation::ContentBlock::BackgroundReport { .. }
                    | crate::conversation::ContentBlock::Invocation { .. }
                    | crate::conversation::ContentBlock::Resource { .. }
                    | crate::conversation::ContentBlock::Thinking { .. }
                    | crate::conversation::ContentBlock::RedactedThinking { .. }
                    | crate::conversation::ContentBlock::Unrecognized { .. }
                    | crate::conversation::ContentBlock::Image { .. } => {}
                }
            }
        }
    }
}
/// Stands in for a block whose text the server withheld, under Claude's `redact-thinking` beta or
/// display updates.
///
/// Meka's own words rather than the model's, but rendered down the same path so there is one way a
/// thinking block reaches the terminal. It survives CommonMark unchanged: a bracketed run is a
/// shortcut reference link only when a matching definition exists, and none does.
pub(crate) const REDACTED_THINKING: &str = "[redacted thinking]";

#[cfg(test)]
mod tests {
    use super::*;

    /// A result records what a screen draws of it only when there is something to record, so a
    /// result with nothing more to show is stored with no record at all, and one with a record
    /// reads back whole.
    #[test]
    fn a_tool_result_records_what_it_shows_only_when_it_has_something() {
        let result = |display: Option<ResultDisplay>| ContentBlock::ToolResult {
            tool_use_id: "call-1".into(),
            content: vec![ToolResultContent::Text { text: "ok".into() }],
            is_error: false,
            display,
        };
        let bare = serde_json::to_value(result(None)).expect("json");
        assert_eq!(
            bare,
            serde_json::json!({
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": [{"type": "text", "text": "ok"}],
                "is_error": false
            })
        );
        assert_eq!(
            serde_json::from_value::<ContentBlock>(bare).expect("a result"),
            result(None)
        );
        assert_eq!(ResultDisplay::default().recorded(), None);
        let recorded = result(
            ResultDisplay {
                exit_code: Some(2),
                change: Some(Change::between(Some("a\nb\n"), Some("a\nc\n"))),
                checklist: Some(Vec::new()),
                background: None,
            }
            .recorded(),
        );
        let stored = serde_json::to_string(&recorded).expect("json");
        assert!(stored.contains("\"exit_code\":2"), "{stored}");
        assert_eq!(
            serde_json::from_str::<ContentBlock>(&stored).expect("a result"),
            recorded
        );
    }

    /// A change is numbered as a diff reads: what stayed and what went by the old text, what came
    /// by the new, with one line of what stayed on either side of each change.
    #[test]
    fn a_change_numbers_its_lines_in_the_text_each_belongs_to() {
        let change = Change::between(Some("a\nb\nc\nd\n"), Some("a\nB\nc\nd\ne\n"));
        assert_eq!((change.added, change.removed), (2, 1));
        let lines: Vec<(ChangeKind, usize, &str)> = change
            .hunks
            .iter()
            .flatten()
            .map(|line| (line.kind, line.number, line.text.as_str()))
            .collect();
        assert_eq!(lines, vec![
            (ChangeKind::Kept, 1, "a"),
            (ChangeKind::Removed, 2, "b"),
            (ChangeKind::Added, 2, "B"),
            (ChangeKind::Kept, 3, "c"),
            (ChangeKind::Kept, 4, "d"),
            (ChangeKind::Added, 5, "e"),
        ]);
        let deleted = Change::between(Some("x\n"), None);
        assert_eq!((deleted.added, deleted.removed), (0, 1));
        assert_eq!(deleted.lines_left_out(), 0);
        // A lone carriage return ends no line, as `file_read` reads it.
        let returned = Change::between(Some("x\ry\nz\n"), Some("x\ry\nZ\n"));
        let lines: Vec<(ChangeKind, usize, &str)> = returned
            .hunks
            .iter()
            .flatten()
            .map(|line| (line.kind, line.number, line.text.as_str()))
            .collect();
        assert_eq!(lines, vec![
            (ChangeKind::Kept, 1, "x\ry"),
            (ChangeKind::Removed, 2, "z"),
            (ChangeKind::Added, 2, "Z"),
        ]);
    }

    /// A change keeps the lines of its hunks up to its budget, each line costing its text and
    /// its fields, and counts the rest, so a file written whole is in the log once, however short
    /// its lines; a hunk cut short ends on its last change, and a line around a change too long
    /// for what is left is let go without the change.
    #[test]
    fn a_change_keeps_what_its_budget_holds_and_counts_the_rest() {
        let line = "x".repeat(99);
        let written: String = (0..1_000).map(|_| format!("{line}\n")).collect();
        let change = Change::between(None, Some(&written));
        assert_eq!((change.added, change.removed), (1_000, 0));
        let kept: usize = change.hunks.iter().flatten().count();
        assert_eq!(
            kept,
            MAX_INLINE_RESULT_BYTES / (99 + Change::LINE_FIELDS_BYTES)
        );
        assert_eq!(change.lines_left_out(), 1_000 - kept);
        let blank = "\n".repeat(100_000);
        let change = Change::between(None, Some(&blank));
        assert_eq!(
            change.hunks.iter().flatten().count(),
            MAX_INLINE_RESULT_BYTES / Change::LINE_FIELDS_BYTES,
            "a blank line costs its fields"
        );
        // Two hunks, the first long enough that the budget runs out just past the second's
        // opening context line: the second holds no change, and is not kept.
        let long = "x".repeat(MAX_INLINE_RESULT_BYTES - 200);
        let old = format!("a\n{long}\ny\nz\nb\n");
        let new = format!("A\n{long}\ny\nz\nB\n");
        let change = Change::between(Some(&old), Some(&new));
        assert_eq!(change.hunks.len(), 1, "{:?}", change.hunks);
        assert_eq!(change.lines_left_out(), 2);
        // A line around a change that the budget cannot hold is let go, and the change is kept.
        let huge = "x".repeat(MAX_INLINE_RESULT_BYTES);
        let change = Change::between(
            Some(&format!("{huge}\nb\n{huge}\n")),
            Some(&format!("{huge}\nc\n{huge}\n")),
        );
        let lines: Vec<(ChangeKind, usize, &str)> = change
            .hunks
            .iter()
            .flatten()
            .map(|line| (line.kind, line.number, line.text.as_str()))
            .collect();
        assert_eq!(lines, vec![
            (ChangeKind::Removed, 2, "b"),
            (ChangeKind::Added, 2, "c")
        ]);
        assert_eq!(change.lines_left_out(), 0);
    }

    /// An image counts wherever it sits, attached or inside a tool result, and a tool reference is
    /// not one.
    #[test]
    fn count_images_counts_attached_and_tool_result_images() {
        let image = || ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "QUJD".to_string(),
        };
        let messages = [
            Message::user_with_images("look at this", vec![image()]),
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![
                        ToolResultContent::ImageCaption {
                            caption: "Image: a.png".to_string(),
                        },
                        ToolResultContent::Image { source: image() },
                        ToolResultContent::ToolReference {
                            tool_name: "file_read".to_string(),
                        },
                    ],
                    is_error: false,
                    display: None,
                }],
                provider_fields: None,
            },
            Message::user("plain"),
        ];
        assert_eq!(count_images(&messages), 2);
        assert_eq!(count_images(&messages[2..]), 0);
    }

    /// The title is the user's words and nothing that rode in front of them, flattened and cut the
    /// same way for every surface that labels a session.
    #[test]
    fn a_title_is_the_first_user_words_collapsed_and_cut() {
        let mut conversation = Conversation::new();
        conversation.append(Message::user_turn(
            "<context>\n[Environment context]\n</context>",
            "find\tall\n\n  rust files",
            Vec::new(),
        ));
        conversation.append(Message::assistant_text("ok"));
        conversation.append(Message::user("a later prompt"));
        assert_eq!(conversation.title(), "find all rust files");

        let long = Conversation::from_vec(vec![Message::user("x".repeat(TITLE_CHARS + 20))]);
        let title = long.title();
        assert_eq!(title.chars().count(), TITLE_CHARS + 1, "{title:?}");
        assert!(title.ends_with('…'), "{title:?}");
    }

    /// No words, no title: a session nobody has spoken to, and one whose first turn carried no
    /// text, until a later turn brings words.
    #[test]
    fn a_title_is_empty_until_a_user_message_carries_words() {
        assert_eq!(Conversation::new().title(), "");
        let mut conversation = Conversation::from_vec(vec![Message {
            role: Role::User,
            content: vec![ContentBlock::TurnContext {
                text: "<context/>".to_string(),
            }],
            provider_fields: None,
        }]);
        assert_eq!(conversation.title(), "");
        conversation.append(Message::user("   "));
        assert_eq!(conversation.title(), "", "whitespace is not words");
        conversation.append(Message::user("now with words"));
        assert_eq!(conversation.title(), "now with words");
    }

    /// A compaction summary is not what the user said. The title is read from the log, where the
    /// summary is a boundary rather than a user `Append`, so reopening a compacted session labels
    /// it with the first user's words and not with the summary that replaced them in the view.
    #[test]
    fn a_title_after_a_compaction_is_still_the_first_users_words() {
        let conversation = Conversation::from_events(vec![
            Event::Append(Message::user_turn(
                "<context/>",
                "find rust files",
                Vec::new(),
            )),
            Event::Append(Message::assistant_text("ok")),
            Event::CompactBoundary {
                summary: Message::user("[Conversation summary from session compaction] found them"),
                replaced_count: 2,
                record: crate::conversation::CompactionRecord::for_test(),
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: Default::default(),
            },
            Event::Append(Message::user_turn(
                "<context/>",
                "and python ones",
                Vec::new(),
            )),
        ]);
        assert_eq!(conversation.title(), "find rust files");
    }

    /// A session opened on an invocation is named by the words typed with it, or by what it
    /// invoked when none were, never by the body it expanded to; a resource an editor attached is
    /// not words, and the words after it name the session.
    #[test]
    fn a_title_reads_an_invocation_and_not_its_body() {
        let invoked = |directive: &str| {
            Conversation::from_vec(vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::TurnContext {
                        text: "<context/>".to_string(),
                    },
                    ContentBlock::Invocation {
                        kind: InvocationKind::Skill,
                        name: "review".to_string(),
                        directive: directive.to_string(),
                        text: format!(
                            "{directive}\n\nBase directory for this skill: /skills/review"
                        ),
                    },
                ],
                provider_fields: None,
            }])
        };
        assert_eq!(invoked("the parser").title(), "the parser");
        assert_eq!(invoked(" ").title(), "review");
        let attached = Conversation::from_vec(vec![Message {
            role: Role::User,
            content: vec![
                ContentBlock::Resource {
                    uri: "file:///a.rs".to_string(),
                    name: "a.rs".to_string(),
                    text: "<resource uri=\"file:///a.rs\">fn main() {}</resource>".to_string(),
                },
                ContentBlock::Text {
                    text: "explain it".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        }]);
        assert_eq!(attached.title(), "explain it");
    }

    /// A boundary writes the tools loaded at it in one order, by name, however they were
    /// gathered, so the same compaction is the same row and an archive of it the same bytes.
    #[test]
    fn a_boundary_lists_its_loaded_tools_by_name() {
        let names = [
            "web_fetch",
            "mcp__exa__search",
            "scratchpad_read",
            "agent_spawn",
            "mcp__notion__fetch",
            "file_find",
            "skill_read",
            "memory_search",
        ];
        let boundary = Event::CompactBoundary {
            summary: Message::user("summary"),
            replaced_count: 1,
            loaded_tools_snapshot: names.iter().map(|name| name.to_string()).collect(),
            checklist_snapshot: Default::default(),
            record: CompactionRecord::for_test(),
        };
        let written = serde_json::to_value(&boundary).expect("serializes");
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            written["CompactBoundary"]["loaded_tools_snapshot"],
            serde_json::json!(sorted)
        );
    }

    /// A reply that did not end as an answer or a round of calls is exported with how it stopped,
    /// and one cut short with what cut it, as `conversation_read` reads them.
    #[test]
    fn the_export_says_how_a_reply_stopped_when_it_did_not_end_as_an_answer() {
        let record = |stop: Option<StopReason>, cut: Option<ReplyCut>| ReplyRecord {
            stop: stop.map(Stop::of),
            cut,
            model: None,
            profile: None,
            usage: None,
            provider_fields: None,
        };
        let events = [
            (
                Event::Append(Message::assistant_text("half")),
                record(Some(StopReason::MaxTokens), None),
            ),
            (
                Event::Append(Message::assistant_text("so far")),
                record(None, Some(ReplyCut::Interrupted)),
            ),
            (
                Event::Append(Message::assistant_text("done")),
                record(Some(StopReason::EndTurn), None),
            ),
        ];
        let markdown = format_session_as_markdown(
            uuid::Uuid::nil(),
            events.iter().map(|(event, record)| (event, Some(record))),
            &std::collections::HashMap::new(),
        );
        assert!(
            markdown.contains("half\n\n*[stop: max_tokens]*"),
            "{markdown}"
        );
        assert!(
            markdown.contains("so far\n\n*[cut: interrupted]*"),
            "{markdown}"
        );
        assert!(markdown.trim_end().ends_with("done"), "{markdown}");
    }

    /// The Markdown export says what was done, as every history does: an invocation as what was
    /// invoked and the words typed with it, an attached file as its mention, never the skill's body
    /// or the file's contents.
    #[test]
    fn the_export_shows_an_invocation_and_a_file_as_what_was_done() {
        let event = Event::Append(Message::invoked_with_a_resource_for_test());
        let markdown = format_session_as_markdown(
            uuid::Uuid::nil(),
            [(&event, None)],
            &std::collections::HashMap::new(),
        );
        assert!(markdown.contains("*Running skill 'review'.*"), "{markdown}");
        assert!(markdown.contains("the parser"), "{markdown}");
        assert!(
            markdown.contains("compare @a.rs with the old one"),
            "{markdown}"
        );
        assert!(!markdown.contains("the skill's body"), "{markdown}");
        assert!(!markdown.contains("fn a() {}"), "{markdown}");
    }

    /// A session opened on an inbox item is named by the message, not by the header Meka wrote
    /// above it.
    #[test]
    fn a_title_skips_the_header_above_an_inbox_item() {
        let item = |source: Option<&str>, body: &str| Message {
            role: Role::User,
            content: vec![ContentBlock::Inbox {
                item_id: None,
                class: Some(InboxClass::Followup),
                source: source.map(str::to_string),
                body: body.to_string(),
                text: match source {
                    Some(source) => {
                        format!("[Message from {source}, arrived 2026-09-16 05:41 +00:00]\n{body}")
                    }
                    None => {
                        format!("[Message arrived 2026-09-16 05:41 +00:00]\n{body}")
                    }
                },
            }],
            provider_fields: None,
        };
        let mut log = Conversation::new();
        log.append(item(Some("telegram"), "What is 5+5?"));
        assert_eq!(log.title(), "What is 5+5?");
        let mut unnamed = Conversation::new();
        unnamed.append(item(None, "What is 6+6?"));
        assert_eq!(unnamed.title(), "What is 6+6?");
    }

    /// What Meka put in place of an attachment is a note, not words, and is not a title: a first
    /// turn that carried only an image has none, whether the image was redacted by the budget or
    /// replaced by another note, until a user says something.
    #[test]
    fn a_title_skips_what_meka_put_in_place_of_an_attachment() {
        let image = ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "QUJD".to_string(),
        };
        let mut conversation = Conversation::from_events(vec![
            Event::Append(Message::user_turn("<context/>", "", vec![image])),
            Event::Redact {
                images: vec![RedactedImage {
                    from_end: 1,
                    block: 1,
                    item: None,
                }],
            },
        ]);
        assert_eq!(conversation.title(), "", "a redacted image is not words");
        // The shapes a compaction re-appends verbatim from a view that carried the stand-ins.
        conversation.append(Message {
            role: Role::User,
            content: vec![
                ContentBlock::TurnContext {
                    text: "<context/>".to_string(),
                },
                ContentBlock::HarnessNote {
                    note: BUDGET_REDACTION_NOTE.to_string(),
                },
            ],
            provider_fields: None,
        });
        conversation.append(Message {
            role: Role::User,
            content: vec![ContentBlock::HarnessNote {
                note: "An image here was removed.".to_string(),
            }],
            provider_fields: None,
        });
        assert_eq!(conversation.title(), "", "nor is a note");
        conversation.append(Message::user("what is in the picture?"));
        assert_eq!(conversation.title(), "what is in the picture?");
    }

    /// Words a person typed are words, whatever they look like: a prompt that opens with Meka's
    /// label, or spells a note's sentence, titles its session like any other.
    #[test]
    fn a_prompt_that_looks_like_a_note_is_a_title() {
        let mut conversation = Conversation::new();
        conversation.append(Message::user(harness_note_text(
            "what does this label mean?",
        )));
        assert_eq!(
            conversation.title(),
            "[Meka harness] what does this label mean?"
        );
        let mut placeholder = Conversation::new();
        placeholder.append(Message::user(BUDGET_REDACTION_NOTE));
        assert_eq!(placeholder.title(), BUDGET_REDACTION_NOTE);
    }

    /// A note recorded beside the user's words is its own paragraph on a single-string wire, behind
    /// Meka's label, and Meka's in the export, not a suffix of what they typed.
    #[test]
    fn a_users_words_and_a_note_are_separate_paragraphs() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::TurnContext {
                    text: "<context/>".to_string(),
                },
                ContentBlock::Text {
                    text: "look at this".to_string(),
                    provider_fields: None,
                },
                ContentBlock::HarnessNote {
                    note: BUDGET_REDACTION_NOTE.to_string(),
                },
            ],
            provider_fields: None,
        };
        assert_eq!(
            message.wire_text(),
            format!("<context/>\n\nlook at this\n\n[Meka harness] {BUDGET_REDACTION_NOTE}")
        );
        let mut output = String::new();
        write_message_markdown(&mut output, &message, &std::collections::HashMap::new());
        assert!(
            output.contains(&format!("## Meka\n\n{BUDGET_REDACTION_NOTE}\n"))
                && output.contains("## User\n\nlook at this\n"),
            "{output}"
        );
        assert_eq!(
            Message::assistant_text("one").text_content(),
            "one",
            "assistant text keeps its join"
        );
    }

    /// What rode a round's results or a nudge is the person's, under its own heading after what
    /// it rode: the results the model read first, then the words it read with them; the nudge
    /// as Meka's, then the words.
    #[test]
    fn an_inbox_item_that_rode_a_round_or_a_nudge_exports_after_what_it_rode() {
        let mut envelope = user_with_tool_result("call_1");
        envelope.content.push(ContentBlock::Text {
            text: "[Message from a client, arrived now]\nand name the file".to_string(),
            provider_fields: None,
        });
        let mut output = String::new();
        write_message_markdown(&mut output, &envelope, &std::collections::HashMap::new());
        let results_at = output
            .find("<summary>Tool result</summary>")
            .expect("the results");
        let words_at = output
            .find("## User\n\n[Message from a client")
            .expect("the words");
        assert!(results_at < words_at, "{output}");

        let mut nudge = Message::nudge_message(NudgeKind::Checklist, "[Checklist: open]");
        nudge.content.push(ContentBlock::Text {
            text: "[Message from a client, arrived now]\nstop".to_string(),
            provider_fields: None,
        });
        let mut output = String::new();
        write_message_markdown(&mut output, &nudge, &std::collections::HashMap::new());
        let nudge_at = output
            .find("## Meka\n\n[Checklist: open]")
            .expect("Meka's words");
        let words_at = output
            .find("## User\n\n[Message from a client")
            .expect("the words");
        assert!(nudge_at < words_at, "{output}");
    }

    /// A turn opened with no words exports as a marked line rather than a heading over nothing.
    #[test]
    fn a_turn_with_no_words_exports_as_a_marked_line() {
        let mut output = String::new();
        write_message_markdown(
            &mut output,
            &Message::user_turn("<context/>", "", Vec::new()),
            &std::collections::HashMap::new(),
        );
        assert!(
            output.contains("## User\n\n*Meka opened this turn with no words"),
            "{output}"
        );
    }

    fn assistant_with_tool_use(use_id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: use_id.to_string(),
                name: "file_read".to_string(),
                input: serde_json::json!({"path": "/tmp/x"}),
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        }
    }

    fn user_with_tool_result(use_id: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: use_id.to_string(),
                content: vec![crate::conversation::ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
                display: None,
            }],
            provider_fields: None,
        }
    }

    fn load_tool_use(id: &str, target: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_string(),
                name: crate::tools::LOAD_TOOL_NAME.to_string(),
                input: serde_json::json!({"name": target}),
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        }
    }

    fn load_tool_result(use_id: &str, is_error: bool) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: use_id.to_string(),
                content: vec![crate::conversation::ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error,
                display: None,
            }],
            provider_fields: None,
        }
    }

    /// A plain appended prompt is withdrawable; anything appended after it is not.
    #[test]
    fn a_turn_opening_is_recognized_only_while_it_is_the_tail() {
        let mut conversation = Conversation::new();
        assert!(
            !conversation.ends_on_a_turn_opening(),
            "an empty conversation has no prompt to withdraw"
        );

        conversation.append(Message::user("check the news"));
        assert!(conversation.ends_on_a_turn_opening());

        conversation.append(assistant_with_tool_use("call_1"));
        assert!(!conversation.ends_on_a_turn_opening(), "the model replied");

        conversation.append(user_with_tool_result("call_1"));
        assert!(
            !conversation.ends_on_a_turn_opening(),
            "a tool result is a User message, but it did not open the turn"
        );
    }

    /// The trap this guard exists for. A compaction summary is itself a plain `User` message, so a
    /// compaction that keeps no tail leaves a materialized view whose last entry looks exactly like
    /// a turn-opening prompt. Withdrawing it would delete the summary standing in for the entire
    /// conversation, which is unrecoverable -- the events it replaced are below the view's logical
    /// start. Only the shape of the event log tells the two apart.
    #[test]
    fn a_compaction_summary_is_not_mistaken_for_a_withdrawable_prompt() {
        let mut conversation = Conversation::new();
        conversation.append(Message::user("first"));
        conversation.append(Message::assistant_text("reply"));
        conversation.replace_for_compaction(
            Message::user("[summary of everything above]"),
            Vec::new(),
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );

        assert!(
            conversation
                .last()
                .is_some_and(|message| message.role == Role::User),
            "the summary really does look like a prompt from the outside"
        );
        assert!(
            !conversation.ends_on_a_turn_opening(),
            "but it is carried by a boundary, not appended, so it is not withdrawable"
        );

        // The next real turn's prompt, appended on top of the summary, is withdrawable again.
        conversation.append(Message::user("check the news"));
        assert!(conversation.ends_on_a_turn_opening());
    }

    /// A repair that rewrites the whole view, as a provider's rejection of its content does, puts
    /// a copy of the summary back at the head; the boundary and the list it recorded stand with
    /// it. Only a repair that puts nothing back, the rewind of everything, takes them away.
    #[test]
    fn a_repair_that_rewrites_the_summary_keeps_the_boundarys_checklist() {
        let mut snapshot = crate::checklist::ChecklistState::default();
        snapshot.add(vec![crate::checklist::NewItem {
            text: "from before".to_string(),
            status: crate::checklist::ChecklistStatus::Pending,
        }]);
        let mut conversation = Conversation::new();
        conversation.append(Message::user("first"));
        conversation.append(Message::assistant_text("reply"));
        conversation.replace_for_compaction(
            Message::user("[summary of everything above]"),
            vec![Message::user("kept"), Message::assistant_text("kept reply")],
            Default::default(),
            snapshot.clone(),
            CompactionRecord::for_test(),
        );
        let (baseline, after) = conversation.checklist_baseline();
        assert_eq!(baseline, snapshot);
        assert_eq!(after.len(), 2);

        let rewritten: Vec<Message> = conversation.as_slice().to_vec();
        conversation.replace_tail(rewritten.len(), rewritten);
        let (baseline, after) = conversation.checklist_baseline();
        assert_eq!(
            baseline, snapshot,
            "the rewritten summary still heads the boundary"
        );
        assert_eq!(after.len(), 2);

        conversation.replace_tail(conversation.len(), Vec::new());
        let (baseline, after) = conversation.checklist_baseline();
        assert!(baseline.items.is_empty() && after.is_empty());
    }

    /// The interaction that makes `run_turn`'s withdrawal guard need *both* of its conditions.
    ///
    /// A compaction can legitimately keep the running turn's prompt as its tail, and then the tail
    /// is once again an appended, turn-opening `User` message, indistinguishable from an untouched
    /// prompt by inspection. Only the event count records that a whole compaction happened in
    /// between, which is why `run_turn` compares it against the value taken at the append rather
    /// than trusting the shape of the tail alone.
    #[test]
    fn a_compaction_that_keeps_the_prompt_still_moves_the_event_count() {
        let mut conversation = Conversation::new();
        conversation.append(Message::user("first"));
        conversation.append(Message::assistant_text("reply"));
        let prompt = Message::user("the turn's prompt");
        conversation.append(prompt.clone());
        let at_the_prompt = conversation.events_len();
        assert!(conversation.ends_on_a_turn_opening());

        conversation.replace_for_compaction(
            Message::user("[summary of everything above]"),
            vec![prompt],
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );

        assert!(
            conversation.ends_on_a_turn_opening(),
            "the tail looks exactly as it did before the compaction"
        );
        assert_ne!(
            conversation.events_len(),
            at_the_prompt,
            "but the log remembers, which is what stops the withdrawal"
        );
    }

    /// The flag tracks "came off disk with something in it", which is the only condition under
    /// which the model can be holding a belief the restart invalidated.
    #[test]
    fn resumed_notice_is_set_only_by_hydration_and_taken_once() {
        let mut hydrated = Conversation::from_events(vec![Event::Append(Message::user("earlier"))]);
        assert!(hydrated.take_resumed_notice());
        assert!(
            !hydrated.take_resumed_notice(),
            "saying it twice would make it scenery"
        );

        // Withdrawn by a turn that failed and popped its user message, and offered again after.
        hydrated.restore_resumed_notice();
        assert!(hydrated.take_resumed_notice());

        // A session with no turns behind it has nothing to be stale about, and a log built up in
        // this process was never restored at all.
        assert!(!Conversation::from_events(Vec::new()).take_resumed_notice());
        let mut fresh = Conversation::new();
        fresh.append(Message::user("first"));
        assert!(!fresh.take_resumed_notice());
    }

    #[test]
    fn message_log_append_and_read() {
        let mut log = Conversation::new();
        log.append(Message::user("first"));
        log.append(Message::assistant_text("second"));
        log.append(Message::user("third"));

        assert_eq!(log.len(), 3);
        assert!(!log.is_empty());
        assert_eq!(log.as_slice().len(), 3);
        assert_eq!(log.as_slice()[0].text_content(), "first");
        assert_eq!(log.last().unwrap().text_content(), "third");
        let collected: Vec<&Message> = log.iter().collect();
        assert_eq!(collected.len(), 3);
    }

    /// A reply a wire split into text blocks reads as one run of words wherever it is shown, as it
    /// streamed, while words on either side of another block stay apart and a person's blocks are
    /// left as they are.
    #[test]
    fn a_reply_s_adjacent_text_blocks_read_as_one_run() {
        let text = |words: &str| ContentBlock::Text {
            text: words.to_string(),
            provider_fields: None,
        };
        let reply = Message {
            role: Role::Assistant,
            content: vec![
                text("The answer is "),
                text("42"),
                text(", as cited."),
                ContentBlock::reasoning("then", None),
                text("After."),
            ],
            provider_fields: None,
        };
        assert_eq!(reply.shown_content().into_owned(), vec![
            text("The answer is 42, as cited."),
            ContentBlock::reasoning("then", None),
            text("After."),
        ]);
        let mut log = Conversation::new();
        log.append(Message::user("q"));
        log.append(reply.clone());
        assert_eq!(
            log.last_assistant_text().as_deref(),
            Some("The answer is 42, as cited.\n\nAfter.")
        );
        let mut markdown = String::new();
        write_message_markdown(&mut markdown, &reply, &std::collections::HashMap::new());
        assert!(
            markdown.contains("The answer is 42, as cited.\n"),
            "{markdown}"
        );
        let asked = Message {
            role: Role::User,
            content: vec![text("one"), text("two")],
            provider_fields: None,
        };
        assert!(matches!(
            asked.shown_content(),
            std::borrow::Cow::Borrowed(blocks) if blocks == asked.content.as_slice()
        ));
    }

    /// A call whose arguments Meka could not read is exported as the model sent them, not as the
    /// empty arguments it was refused with; its result says why.
    #[test]
    fn an_unread_call_is_exported_as_the_model_sent_it() {
        let reply = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "file_read".to_string(),
                input: serde_json::json!({}),
                unparsed: Some(UnparsedArguments {
                    text: Some("{\"path\": ".to_string()),
                    error: "EOF while parsing".to_string(),
                }),
                provider_fields: None,
            }],
            provider_fields: None,
        };
        let mut markdown = String::new();
        write_message_markdown(&mut markdown, &reply, &std::collections::HashMap::new());
        assert!(markdown.contains("```\n{\"path\": \n```"), "{markdown}");
        assert!(!markdown.contains("```json"), "{markdown}");
    }

    #[test]
    fn last_assistant_text_walks_past_tool_results() {
        // Sub-agent turn shape after a tool-use round: assistant emits a tool_use, then the loop
        // appends the matching tool_result as a Role::User trailer. `last()` would return that
        // trailer, not the assistant's text; the helper has to walk backward.
        let mut log = Conversation::new();
        log.append(Message::user("kick off"));
        log.append(Message::assistant_text("final assistant answer"));
        log.append(user_with_tool_result("call_id"));

        assert_eq!(
            log.last_assistant_text().as_deref(),
            Some("final assistant answer")
        );
    }

    /// A reply's parts are joined when its words are read, a blank line apart, so text the provider
    /// returned in two parts is not glued into one sentence; a reply with no words reads as none.
    #[test]
    fn the_last_reply_reads_its_parts_apart_and_a_blank_one_as_none() {
        let mut log = Conversation::new();
        log.append(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "First part.".to_string(),
                    provider_fields: None,
                },
                ContentBlock::Thinking {
                    parts: vec![crate::conversation::ReasoningPart::reasoning("late")],
                    opaque: None,
                    provider_fields: None,
                },
                ContentBlock::Text {
                    text: "Second part.".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        });
        assert_eq!(
            log.last_assistant_text().as_deref(),
            Some("First part.\n\nSecond part.")
        );
        log.append(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Thinking {
                parts: vec![crate::conversation::ReasoningPart::reasoning(
                    "nothing to say",
                )],
                opaque: None,
                provider_fields: None,
            }],
            provider_fields: None,
        });
        assert_eq!(log.last_assistant_text(), None);
    }

    #[test]
    fn last_assistant_text_none_on_empty() {
        let log = Conversation::new();
        assert_eq!(log.last_assistant_text(), None);
    }

    #[test]
    fn last_assistant_text_none_when_no_assistant_message() {
        let mut log = Conversation::new();
        log.append(Message::user("only user message"));
        assert_eq!(log.last_assistant_text(), None);
    }

    #[test]
    fn message_log_replace_for_compaction_replaces_all() {
        let mut log = Conversation::new();
        log.append(Message::user("m1"));
        log.append(Message::assistant_text("m2"));
        log.append(Message::user("m3"));

        let summary = Message::user("[summary]");
        let tail = vec![Message::assistant_text("kept-1"), Message::user("kept-2")];
        log.replace_for_compaction(
            summary,
            tail,
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );

        let view = log.as_slice();
        assert_eq!(view.len(), 3);
        assert_eq!(view[0].text_content(), "[summary]");
        assert_eq!(view[1].text_content(), "kept-1");
        assert_eq!(view[2].text_content(), "kept-2");
    }

    #[test]
    fn message_log_replace_for_compaction_empty_tail() {
        let mut log = Conversation::new();
        log.append(Message::user("m1"));
        log.replace_for_compaction(
            Message::user("[summary]"),
            Vec::new(),
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "[summary]");
    }

    #[test]
    fn message_log_pop_unsaved() {
        let mut log = Conversation::new();
        log.append(Message::user("staying"));
        log.append(Message::user("rolling-back"));

        let popped = log.pop_unsaved();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().text_content(), "rolling-back");
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "staying");
    }

    #[test]
    fn message_log_pop_unsaved_on_empty() {
        let mut log = Conversation::new();
        assert!(log.pop_unsaved().is_none());
    }

    /// Taking the Claude thinking out of a message leaves the reasoning readable and nothing
    /// Claude would check: a signed block loses its signature and keeps its text, one with no text
    /// goes, a redacted block goes, and another wire's reasoning stays as it was.
    #[test]
    fn a_message_without_claude_thinking_keeps_the_readable_reasoning() {
        let signed = |thinking: &str| ContentBlock::Thinking {
            parts: vec![crate::conversation::ReasoningPart::reasoning(
                thinking.to_string(),
            )],
            opaque: Some(OpaqueReasoning::Signed {
                signature: "sig".to_string(),
            }),
            provider_fields: None,
        };
        let sealed = ContentBlock::Thinking {
            parts: vec![crate::conversation::ReasoningPart::reasoning("summary")],
            opaque: Some(OpaqueReasoning::Sealed {
                encrypted_content: "sealed".to_string(),
                id: None,
                origin: Some("https://chatgpt.com".to_string()),
            }),
            provider_fields: None,
        };
        let message = Message {
            role: Role::Assistant,
            content: vec![
                signed("the reasoning"),
                signed(""),
                ContentBlock::RedactedThinking {
                    data: "redacted".to_string(),
                    provider_fields: None,
                },
                sealed.clone(),
                ContentBlock::Text {
                    text: "the answer".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let without = message.without_claude_thinking().expect("it carried some");
        assert_eq!(
            serde_json::to_value(&without.content).expect("serializes"),
            serde_json::to_value(vec![
                ContentBlock::Thinking {
                    parts: vec![crate::conversation::ReasoningPart::reasoning(
                        "the reasoning"
                    )],
                    opaque: None,
                    provider_fields: None,
                },
                sealed.clone(),
                ContentBlock::Text {
                    text: "the answer".to_string(),
                    provider_fields: None,
                },
            ])
            .expect("serializes")
        );
        assert!(without.without_claude_thinking().is_none());
        assert!(
            Message::assistant_text("plain")
                .without_claude_thinking()
                .is_none()
        );
        assert!(
            Message {
                role: Role::Assistant,
                content: vec![sealed],
                provider_fields: None,
            }
            .without_claude_thinking()
            .is_none(),
            "another wire's reasoning is not Claude's to check"
        );
    }

    /// An unsign takes the Claude thinking out of the messages ahead of it and leaves one appended
    /// after it signed, the same on a reload; it is undone like a repair, and an export marks
    /// where it happened while keeping each message once.
    #[test]
    fn an_unsign_strips_the_thinking_ahead_of_it_and_no_further() {
        let signed = |thinking: &str| Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    parts: vec![crate::conversation::ReasoningPart::reasoning(
                        thinking.to_string(),
                    )],
                    opaque: Some(OpaqueReasoning::Signed {
                        signature: "sig".to_string(),
                    }),
                    provider_fields: None,
                },
                ContentBlock::Text {
                    text: "an answer".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let mut log = Conversation::new();
        log.append(Message::user("first question"));
        log.append(signed("first reasoning"));
        log.append(Message::user("second question"));
        assert!(matches!(log.unsign_thinking(), Some(Event::Unsign)));
        log.append(signed("second reasoning"));

        for view in [
            log.as_slice().to_vec(),
            Conversation::from_events(log.events().to_vec())
                .as_slice()
                .to_vec(),
        ] {
            assert_eq!(view.len(), 4);
            assert!(!view[1].carries_claude_thinking());
            assert!(matches!(
                &view[1].content[0],
                ContentBlock::Thinking { parts, opaque: None, .. } if reasoning_text(parts) == "first reasoning"
            ));
            assert!(view[3].carries_claude_thinking(), "appended after it");
        }

        let markdown = format_session_as_markdown(
            uuid::Uuid::nil(),
            log.events().iter().map(|event| (event, None)),
            &std::collections::HashMap::new(),
        );
        assert!(
            markdown.contains("no longer sent to the model"),
            "{markdown}"
        );
        assert_eq!(markdown.matches("first question").count(), 1, "{markdown}");

        let mut undone = Conversation::new();
        undone.append(Message::user("a question"));
        undone.append(signed("reasoning"));
        assert!(undone.unsign_thinking().is_some());
        assert!(undone.pop_repair());
        assert!(undone.as_slice()[1].carries_claude_thinking(), "put back");
        let mut plain = Conversation::new();
        plain.append(Message::user("a question"));
        assert!(plain.unsign_thinking().is_none(), "nothing to take");
        assert_eq!(plain.events().len(), 1);
    }

    #[test]
    fn replace_tail_swaps_the_trailing_messages() {
        let mut log = Conversation::new();
        log.append(Message::user("kept"));
        log.append(assistant_with_tool_use("call_1"));
        log.append(user_with_tool_result("call_1"));

        log.replace_tail(2, vec![
            Message::assistant_text("degraded assistant"),
            Message::user("degraded result"),
        ]);

        let view = log.as_slice();
        assert_eq!(view.len(), 3);
        assert_eq!(view[0].text_content(), "kept");
        assert_eq!(view[1].text_content(), "degraded assistant");
        assert_eq!(view[2].text_content(), "degraded result");
    }

    /// A failed repair has to leave nothing behind, or a misdiagnosed rejection would permanently
    /// cost a good tool result.
    #[test]
    fn pop_repair_restores_the_originals_exactly() {
        let mut log = Conversation::new();
        log.append(Message::user("kept"));
        log.append(assistant_with_tool_use("call_1"));
        log.append(user_with_tool_result("call_1"));
        let before: Vec<String> = log.iter().map(|m| format!("{m:?}")).collect();

        log.replace_tail(2, vec![Message::assistant_text("degraded")]);
        assert_eq!(log.len(), 2);

        assert!(log.pop_repair());
        let after: Vec<String> = log.iter().map(|m| format!("{m:?}")).collect();
        assert_eq!(before, after);
        // And the event log is clean, not carrying a repair that cancels another repair.
        assert_eq!(log.events().len(), 3);
    }

    /// The rollback compaction takes when its write fails. Getting this wrong is silent: the model
    /// would keep reasoning from a summary the database never accepted.
    #[test]
    fn pop_compaction_restores_the_pre_compaction_view() {
        let mut log = Conversation::new();
        log.append(Message::user("one"));
        log.append(Message::assistant_text("two"));
        log.append(Message::user("three"));
        let before: Vec<String> = log.iter().map(|m| format!("{m:?}")).collect();
        let events_before = log.events().len();

        log.replace_for_compaction(
            Message::user("summary"),
            vec![Message::user("three")],
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );
        assert_ne!(log.len(), before.len(), "compaction must have rewritten it");

        assert!(log.pop_compaction());
        assert_eq!(
            before,
            log.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>()
        );
        assert_eq!(
            log.events().len(),
            events_before,
            "the boundary and its re-appended tail must both be gone"
        );
    }

    /// The second compaction of a session must roll back to the *first* boundary's view, not past
    /// it. `rposition` picking the wrong boundary would resurrect turns the earlier compaction
    /// legitimately retired.
    #[test]
    fn pop_compaction_on_an_already_compacted_session() {
        let mut log = Conversation::new();
        log.append(Message::user("old"));
        log.append(Message::assistant_text("older"));
        log.replace_for_compaction(
            Message::user("summary one"),
            Vec::new(),
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );
        log.append(Message::user("after the first summary"));
        let before: Vec<String> = log.iter().map(|m| format!("{m:?}")).collect();

        log.replace_for_compaction(
            Message::user("summary two"),
            vec![Message::user("kept")],
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );
        assert!(log.pop_compaction());
        assert_eq!(
            before,
            log.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>(),
            "rollback must land on the first boundary's view"
        );
    }

    #[test]
    fn pop_compaction_without_a_boundary_is_a_no_op() {
        let mut log = Conversation::new();
        log.append(Message::user("only"));
        assert!(!log.pop_compaction());
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn pop_repair_ignores_a_non_repair_tail() {
        let mut log = Conversation::new();
        log.append(Message::user("only"));
        assert!(!log.pop_repair());
        assert_eq!(log.len(), 1);
    }

    /// A repair is position-relative, so removing an earlier event must not retarget it.
    #[test]
    fn repair_survives_orphan_sanitization_of_an_earlier_event() {
        let mut log = Conversation::new();
        log.append(Message::user("first"));
        // Orphaned: no tool_result follows.
        log.append(assistant_with_tool_use("orphan"));
        log.append(Message::user("second"));
        log.append(Message::assistant_text("rejected"));
        log.replace_tail(1, vec![Message::assistant_text("degraded")]);

        let dropped = log.sanitize_orphans();
        assert_eq!(dropped.len(), 1);
        assert_eq!(log.len(), 3);
        assert_eq!(log.as_slice()[0].text_content(), "first");
        assert_eq!(log.as_slice()[1].text_content(), "second");
        assert_eq!(log.as_slice()[2].text_content(), "degraded");
    }

    fn base64_of(format: image::ImageFormat) -> String {
        use base64::Engine as _;
        let mut bytes = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);
        if format == image::ImageFormat::Jpeg {
            image::RgbImage::from_pixel(4, 4, image::Rgb([1, 2, 3]))
                .write_to(&mut cursor, format)
                .expect("encode");
        } else {
            image::RgbaImage::from_pixel(4, 4, image::Rgba([1, 2, 3, 255]))
                .write_to(&mut cursor, format)
                .expect("encode");
        }
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    }

    fn image_block(data: String, media_type: &str) -> ContentBlock {
        ContentBlock::Image {
            source: crate::image::ImageSource::Base64 {
                media_type: media_type.to_string(),
                data,
            },
        }
    }

    #[test]
    fn materialization_replaces_a_mislabeled_image() {
        let mut log = Conversation::new();
        log.append(Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "look".to_string(),
                    provider_fields: None,
                },
                image_block(base64_of(image::ImageFormat::Jpeg), "image/png"),
            ],
            provider_fields: None,
        });

        assert_eq!(log.invalid_images_replaced(), 1);
        let content = &log.as_slice()[0].content;
        assert!(
            matches!(content[0], ContentBlock::Text { .. }),
            "text is kept"
        );
        match &content[1] {
            ContentBlock::HarnessNote { note } => assert!(note.contains("image/png"), "{note}"),
            other => panic!("expected the image to become a note, got {other:?}"),
        }
    }

    #[test]
    fn materialization_marks_a_repaired_tool_result_as_an_error() {
        let mut log = Conversation::new();
        log.append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: vec![crate::conversation::ToolResultContent::Image {
                    source: crate::image::ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: base64_of(image::ImageFormat::Jpeg),
                    },
                }],
                is_error: false,
                display: None,
            }],
            provider_fields: None,
        });

        assert_eq!(log.invalid_images_replaced(), 1);
        match &log.as_slice()[0].content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "call_1");
                assert!(is_error);
                assert!(matches!(
                    content[0],
                    crate::conversation::ToolResultContent::HarnessNote { .. }
                ));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    /// A correctly-labeled image, and one whose bytes this build simply can't identify, both have
    /// to survive: the second may be a format the provider accepts and we don't decode.
    #[test]
    fn materialization_leaves_valid_and_unidentifiable_images_alone() {
        let mut log = Conversation::new();
        log.append(Message {
            role: Role::User,
            content: vec![
                image_block(base64_of(image::ImageFormat::Png), "image/png"),
                image_block("BASE64DATA".to_string(), "image/png"),
            ],
            provider_fields: None,
        });

        assert_eq!(log.invalid_images_replaced(), 0);
        assert!(
            log.as_slice()[0]
                .content
                .iter()
                .all(|block| matches!(block, ContentBlock::Image { .. }))
        );
    }

    /// The repair lives in materialization, not in a one-shot pass, precisely so a later rebuild
    /// can't quietly put the refused bytes back.
    #[test]
    fn materialization_repair_survives_a_later_rebuild() {
        let mut log = Conversation::new();
        log.append(Message::user("turn one"));
        log.append(Message::assistant_text("answer one"));
        log.append(Message {
            role: Role::User,
            content: vec![image_block(
                base64_of(image::ImageFormat::Jpeg),
                "image/png",
            )],
            provider_fields: None,
        });
        log.append(Message::assistant_text("answer two"));

        // Any operation that re-derives the view from the event log.
        assert!(log.rewind(1).is_some());

        assert!(
            log.iter()
                .flat_map(|message| message.content.iter())
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "the mislabeled image must not come back"
        );
    }

    #[test]
    fn rewind_drops_whole_turns_and_snaps_to_a_user_boundary() {
        let mut log = Conversation::new();
        log.append(Message::user("turn one"));
        log.append(Message::assistant_text("answer one"));
        log.append(Message::user("turn two"));
        log.append(assistant_with_tool_use("call_1"));
        log.append(user_with_tool_result("call_1"));
        log.append(Message::assistant_text("answer two"));

        assert!(log.rewind(1).is_some());

        let view = log.as_slice();
        assert_eq!(
            view.len(),
            2,
            "the whole second turn goes, results included"
        );
        assert_eq!(view[0].text_content(), "turn one");
        assert_eq!(view[1].text_content(), "answer one");
        assert!(
            orphan_positions(&replay(log.events().iter()).placed).is_empty(),
            "the cut must not separate a tool_use from its tool_result"
        );
    }

    /// A nudge is a user-role message the loop writes inside the model's turn, so it must not be
    /// where a rewind cuts, a turn label starts, or a worker's turn count steps; a tool round's
    /// envelope never was. Both stay prompts to the model, which is what a compaction splits on.
    #[test]
    fn a_nudge_continues_the_turn_it_answers() {
        let checklist_nudge = Message::nudge_message(
            NudgeKind::Checklist,
            "[Checklist: items are still open, so this turn is not over.]\n\nOpen:\n",
        );
        let thinking_nudge = Message::nudge_message(NudgeKind::VisibleReply, THINKING_ONLY_NUDGE);
        for nudge in [&checklist_nudge, &thinking_nudge] {
            assert!(nudge.is_nudge());
            assert!(nudge.is_prompt(), "the model has it to answer");
            assert!(!nudge.opens_turn());
        }
        let envelope = user_with_tool_result("call_1");
        assert!(envelope.carries_tool_results() && !envelope.is_prompt());
        assert!(!envelope.opens_turn() && !envelope.is_nudge());
        let typed = Message::user("[Checklist: the user's own words, were they ever first]");
        assert!(
            !typed.is_nudge(),
            "words alone are never a nudge, whatever they open with"
        );
        let mut ridden = checklist_nudge.clone();
        ridden.content.push(ContentBlock::Text {
            text: "[Message from a client, arrived now]\nstop".to_string(),
            provider_fields: None,
        });
        assert!(
            ridden.is_nudge(),
            "an item riding behind the block leaves it a nudge"
        );

        let mut log = Conversation::new();
        log.append(Message::user("turn one"));
        log.append(Message::assistant_text("answer one"));
        log.append(Message::user("plan and do it"));
        log.append(Message::assistant_text("I have planned it."));
        log.append(checklist_nudge);
        log.append(assistant_with_tool_use("call_1"));
        log.append(user_with_tool_result("call_1"));
        log.append(Message::assistant_text("Done."));
        assert!(
            !log.ends_on_a_turn_opening(),
            "the turn was answered before the nudge"
        );

        assert!(log.rewind(1).is_some());
        let view = log.as_slice();
        assert_eq!(view.len(), 2, "the whole nudged turn goes: {view:?}");
        assert_eq!(view[1].text_content(), "answer one");
    }

    /// A summary heading the view is no turn and is never taken: a rewind past every turn after
    /// it answers nothing to drop rather than emptying the context. The rounds a compaction kept
    /// from inside a turn are the rest of that turn and count as one.
    #[test]
    fn a_rewind_never_takes_the_summary_heading_the_view() {
        let compacted = |after: Vec<Message>| {
            Conversation::from_events(
                [
                    Event::Append(Message::user("old turn")),
                    Event::CompactBoundary {
                        summary: Message::user("summary"),
                        replaced_count: 1,
                        record: CompactionRecord::for_test(),
                        loaded_tools_snapshot: Default::default(),
                        checklist_snapshot: Default::default(),
                    },
                ]
                .into_iter()
                .chain(after.into_iter().map(Event::Append))
                .collect(),
            )
        };
        let texts = |log: &Conversation| -> Vec<String> {
            log.as_slice().iter().map(Message::text_content).collect()
        };

        assert!(compacted(Vec::new()).rewind(1).is_none());

        let mut log = compacted(vec![
            Message::user("new turn"),
            Message::assistant_text("new answer"),
        ]);
        assert!(log.rewind(2).is_none(), "one turn follows the summary");
        assert!(log.rewind(1).is_some());
        assert_eq!(texts(&log), ["summary"]);
        assert!(log.rewind(1).is_none());

        let mut log = compacted(vec![
            assistant_with_tool_use("call_1"),
            user_with_tool_result("call_1"),
            Message::assistant_text("kept answer"),
            Message::user("new turn"),
            Message::assistant_text("new answer"),
        ]);
        assert!(log.rewind(3).is_none());
        assert!(log.rewind(2).is_some(), "the kept rounds count as a turn");
        assert_eq!(texts(&log), ["summary"]);
    }

    /// A repair that puts nothing back spans the events from the earliest one a dropped message
    /// stood for to its own. A repair's replacement stands for the message it replaced, position
    /// by position, so a cut at the second of them leaves the first's event out. A summary stands
    /// for everything it replaced, so a repair that takes it takes the turns before it too, and
    /// counts it as a turn. A prompt taken back and appended again with a message joined to it is
    /// marked as such, and stands for the prompt it took the place of; a summary taken back so
    /// spans the repair alone, the turns before it staying where they are.
    #[test]
    fn a_rewind_spans_the_events_its_dropped_messages_stood_for() {
        let stamped = |log: &Conversation| -> Vec<(usize, Event)> {
            log.events().iter().cloned().enumerate().collect()
        };
        let rewind = |first, at, turns, rejoined| Rewind {
            first,
            at,
            turns,
            rejoined,
        };
        let mut log = Conversation::new();
        log.append(Message::user("turn one"));
        log.append(Message::assistant_text("answer one"));
        log.append(Message::user("turn two"));
        log.append(assistant_with_tool_use("call_1"));
        log.append(user_with_tool_result("call_1"));
        log.replace_tail(4, vec![
            Message::assistant_text("answer one"),
            Message::user("turn two"),
            assistant_with_tool_use("call_1"),
            user_with_tool_result("call_1"),
        ]);
        log.append(Message::assistant_text("answer two"));
        assert!(log.rewind(1).is_some());
        assert_eq!(rewinds(&stamped(&log)), vec![rewind(2, 7, 1, false)]);

        let mut compacted = Conversation::from_events(vec![
            Event::Append(Message::user("old turn")),
            Event::CompactBoundary {
                summary: Message::user("summary"),
                replaced_count: 1,
                record: CompactionRecord::for_test(),
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: Default::default(),
            },
            Event::Append(Message::user("new turn")),
            Event::Append(Message::assistant_text("new answer")),
        ]);
        compacted.replace_tail(3, Vec::new());
        assert_eq!(rewinds(&stamped(&compacted)), vec![rewind(0, 4, 2, false)]);

        let mut joined = Conversation::new();
        joined.append(Message::user("turn one"));
        joined.append(Message::assistant_text("answer one"));
        joined.append(Message::user("turn two"));
        joined.replace_tail(1, Vec::new());
        let mut both = Message::user("turn two");
        both.content.push(ContentBlock::Text {
            text: "and this".to_string(),
            provider_fields: None,
        });
        joined.append(both);
        joined.append(Message::assistant_text("answer two"));
        assert!(joined.rewind(1).is_some());
        assert_eq!(rewinds(&stamped(&joined)), vec![
            rewind(2, 3, 1, true),
            rewind(2, 6, 1, false),
        ]);

        let mut steered = Message::user("summary");
        steered.content.push(ContentBlock::Text {
            text: "and this".to_string(),
            provider_fields: None,
        });
        let summary_joined = Conversation::from_events(vec![
            Event::Append(Message::user("old turn")),
            Event::CompactBoundary {
                summary: Message::user("summary"),
                replaced_count: 1,
                record: CompactionRecord::for_test(),
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: Default::default(),
            },
            Event::Repair {
                replaced_count: 1,
                messages: Vec::new(),
            },
            Event::Append(steered),
        ]);
        assert_eq!(rewinds(&stamped(&summary_joined)), vec![rewind(
            2, 2, 1, true
        )]);
    }

    /// Items an interrupt joins onto a lone summary leave the summary standing for its boundary:
    /// the checklist the boundary recorded is still the baseline, and the log's reader still sees
    /// the compaction.
    #[test]
    fn a_summary_something_joined_still_stands_for_its_boundary() {
        let mut checklist = crate::checklist::ChecklistState::default();
        checklist.add(vec![crate::checklist::NewItem {
            text: "keep going".to_string(),
            status: crate::checklist::ChecklistStatus::Pending,
        }]);
        let mut steered = Message::user("summary");
        steered.content.push(ContentBlock::Text {
            text: "and this".to_string(),
            provider_fields: None,
        });
        let events = vec![
            Event::Append(Message::user("old turn")),
            Event::CompactBoundary {
                summary: Message::user("summary"),
                replaced_count: 1,
                record: CompactionRecord::for_test(),
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: checklist.clone(),
            },
            Event::Repair {
                replaced_count: 1,
                messages: Vec::new(),
            },
            Event::Append(steered),
        ];
        let conversation = Conversation::from_events(events.clone());
        assert_eq!(conversation.checklist_baseline().0, checklist);
        let stamped: Vec<(usize, Event)> = events.into_iter().enumerate().collect();
        let view = materialize_annotated(&stamped);
        assert!(
            view.markers[0].is_some(),
            "the compaction left the log's view"
        );
    }

    #[test]
    fn rewind_past_the_start_returns_none() {
        let mut log = Conversation::new();
        log.append(Message::user("only turn"));
        log.append(Message::assistant_text("answer"));

        assert!(log.rewind(2).is_none(), "only one turn exists");
        assert!(log.rewind(0).is_none());
        assert_eq!(log.len(), 2, "a refused rewind leaves the log alone");
    }

    #[test]
    fn rewind_all_turns_empties_the_view() {
        let mut log = Conversation::new();
        log.append(Message::user("turn one"));
        log.append(Message::assistant_text("answer one"));
        log.append(Message::user("turn two"));
        log.append(Message::assistant_text("answer two"));

        assert!(log.rewind(2).is_some());
        assert!(log.is_empty());
    }

    /// A redaction names one image, tail-relative, and the replay puts Meka's note exactly there:
    /// an input image by block, a tool result's image by item, and nothing else.
    #[test]
    fn a_redaction_replaces_the_image_it_names_and_nothing_else() {
        let image = crate::image::ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "aGk=".to_string(),
        };
        let mut log = Conversation::new();
        log.append(Message::user_with_images("look", vec![image.clone()]));
        log.append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".to_string(),
                content: vec![
                    ToolResultContent::ImageCaption {
                        caption: "Image: x.png".to_string(),
                    },
                    ToolResultContent::Image {
                        source: image.clone(),
                    },
                ],
                is_error: false,
                display: None,
            }],
            provider_fields: None,
        });
        log.append(Message::assistant_text("seen"));

        let event = log.redact_images(vec![
            RedactedImage {
                from_end: 3,
                block: 1,
                item: None,
            },
            RedactedImage {
                from_end: 2,
                block: 0,
                item: Some(1),
            },
            // A position naming text, which a stale address could: left alone.
            RedactedImage {
                from_end: 3,
                block: 0,
                item: None,
            },
        ]);
        assert!(matches!(event, Event::Redact { .. }));

        let check = |view: &[Message]| {
            assert!(
                matches!(&view[0].content[0], ContentBlock::Text { text, .. } if text == "look")
            );
            assert!(
                matches!(&view[0].content[1], ContentBlock::HarnessNote { note } if note == BUDGET_REDACTION_NOTE),
                "{:?}",
                view[0].content
            );
            match &view[1].content[0] {
                ContentBlock::ToolResult { content, .. } => {
                    assert!(
                        matches!(&content[0], ToolResultContent::ImageCaption { caption } if caption == "Image: x.png")
                    );
                    assert!(
                        matches!(&content[1], ToolResultContent::HarnessNote { note } if note == BUDGET_REDACTION_NOTE)
                    );
                }
                other => panic!("expected the tool result, got {other:?}"),
            }
            assert_eq!(view[2].text_content(), "seen");
        };
        check(log.as_slice());
        // And the same from the store's replay of the same events.
        check(Conversation::from_events(log.events().to_vec()).as_slice());
    }

    /// A compaction after a resume that dropped an orphan replaces the whole view when the store
    /// replays it, not the whole view minus one.
    ///
    /// `sanitize_orphans` shortens the log in memory only, so the `replaced_count` a compaction
    /// then records is one short of what the store holds. Truncating by it left the session's
    /// first message standing above the summary on every later turn, and each compaction after a
    /// dropped orphan widened the gap by one.
    #[test]
    fn a_compaction_after_an_orphan_dropping_resume_replaces_the_whole_view() {
        // What the store holds: a crash left a tool call without its result.
        let stored = vec![
            Event::Append(Message::user("first")),
            Event::Append(Message::assistant_text("second")),
            Event::Append(assistant_with_tool_use("u1")),
        ];
        // The resume drops the orphan in memory, then the session compacts.
        let mut resumed = Conversation::from_events(stored.clone());
        assert_eq!(resumed.sanitize_orphans().len(), 1);
        resumed.replace_for_compaction(
            Message::assistant_text("summary"),
            Vec::new(),
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );
        let boundary = resumed
            .events()
            .last()
            .cloned()
            .expect("the boundary was recorded");
        // The next resume replays the store, which still has the orphan.
        let mut replayed = stored;
        replayed.push(boundary);
        let view = Conversation::from_events(replayed);
        assert_eq!(
            view.as_slice()
                .iter()
                .map(|message| message.text_content())
                .collect::<Vec<_>>(),
            vec!["summary".to_string()],
            "nothing from before the boundary may survive it"
        );
    }

    #[test]
    fn message_log_sanitize_orphans_drops_unmatched_tool_use() {
        let mut log = Conversation::new();
        log.append(Message::user("hello"));
        log.append(assistant_with_tool_use("u1"));
        // No matching tool_result follows; the assistant message is orphaned.

        let dropped = log.sanitize_orphans();
        assert_eq!(dropped.len(), 1);
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "hello");
    }

    #[test]
    fn message_log_sanitize_orphans_drops_truncated_multi_tool_use_tail() {
        // Reproduces the real corruption shape: a model response truncated at `max_tokens` while
        // emitting tools, persisted as a trailing assistant message with leading text plus several
        // `tool_use` blocks and no following `tool_result`. Anthropic rejects this on the next turn
        // ("tool_use ids were found without tool_result blocks"); sanitize must drop the whole
        // message regardless of the leading text or the number of tool_use blocks.
        let mut log = Conversation::new();
        log.append(Message::user("read the diff and explain it"));
        log.append(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "I'll check the uncommitted changes.".to_string(),
                    provider_fields: None,
                },
                ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "shell_execute".to_string(),
                    input: serde_json::json!({"command": "git diff"}),
                    unparsed: None,
                    provider_fields: None,
                },
                ContentBlock::ToolUse {
                    id: "u2".to_string(),
                    name: "scratchpad_read".to_string(),
                    input: serde_json::json!({"name": "tool_1_output"}),
                    unparsed: None,
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        });

        let dropped = log.sanitize_orphans();
        assert_eq!(dropped.len(), 1);
        assert_eq!(log.len(), 1);
        assert_eq!(
            log.as_slice()[0].text_content(),
            "read the diff and explain it"
        );
        // Idempotent: a second pass on the now-clean log is a no-op.
        assert!(log.sanitize_orphans().is_empty());
    }

    #[test]
    fn message_log_sanitize_orphans_preserves_matched_tool_use() {
        let mut log = Conversation::new();
        log.append(Message::user("ask"));
        log.append(assistant_with_tool_use("u1"));
        log.append(user_with_tool_result("u1"));

        let dropped = log.sanitize_orphans();
        assert!(dropped.is_empty());
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn message_log_clone_independent() {
        let mut log = Conversation::new();
        log.append(Message::user("original"));
        let mut cloned = log.clone();
        cloned.append(Message::user("only-in-clone"));

        assert_eq!(log.len(), 1);
        assert_eq!(cloned.len(), 2);
    }

    #[test]
    fn message_log_into_iter_for_ref() {
        let mut log = Conversation::new();
        log.append(Message::user("a"));
        log.append(Message::user("b"));
        let texts: Vec<String> = (&log).into_iter().map(|m| m.text_content()).collect();
        assert_eq!(texts, vec!["a", "b"]);
    }

    #[test]
    fn events_are_append_only_after_compaction() {
        // After replace_for_compaction, the prior Append events MUST still be present in the events
        // log, even though the materialized view has truncated them. This is the structural
        // invariant: events in the log only ever grow.
        let mut log = Conversation::new();
        log.append(Message::user("m1"));
        log.append(Message::assistant_text("m2"));
        log.append(Message::user("m3"));
        let pre_event_count = log.events().len();

        log.replace_for_compaction(
            Message::user("[summary]"),
            vec![Message::user("tail")],
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );

        let post_event_count = log.events().len();
        // pre + 1 boundary + 1 tail Append = pre + 2.
        assert_eq!(post_event_count, pre_event_count + 2);
        // The original three Append events are still there.
        let append_count = log
            .events()
            .iter()
            .filter(|e| matches!(e, Event::Append(_)))
            .count();
        assert_eq!(append_count, pre_event_count + 1); // 3 + 1 tail
    }

    #[test]
    fn materialize_with_compact_boundary() {
        let mut log = Conversation::new();
        for i in 1..=5 {
            log.append(Message::user(format!("m{i}")));
        }
        log.replace_for_compaction(
            Message::user("[summary]"),
            vec![Message::assistant_text("kept-1"), Message::user("kept-2")],
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );

        let view = log.as_slice();
        assert_eq!(view.len(), 3);
        assert_eq!(view[0].text_content(), "[summary]");
        assert_eq!(view[1].text_content(), "kept-1");
        assert_eq!(view[2].text_content(), "kept-2");
    }

    #[test]
    fn extract_loaded_tool_names_pure_appends() {
        let log = Conversation::from_vec(vec![
            load_tool_use("u1", "scratchpad_read"),
            load_tool_result("u1", false),
        ]);
        let loaded = crate::tools::load_tool::extract_loaded_tool_names_from_events(
            log.events(),
            crate::provider::DeferredToolDelivery::Appended,
        );
        assert!(loaded.iter().any(|name| name == "scratchpad_read"));
    }

    /// Load order, not registry order, is what makes the tools array append-only. A failed load
    /// contributes nothing, and a repeat load must not move a name to the back.
    #[test]
    fn extract_loaded_tool_names_preserves_load_order() {
        let log = Conversation::from_vec(vec![
            load_tool_use("u1", "omega"),
            load_tool_result("u1", false),
            load_tool_use("u2", "alpha"),
            load_tool_result("u2", false),
            load_tool_use("u3", "broken"),
            load_tool_result("u3", true),
            load_tool_use("u4", "omega"),
            load_tool_result("u4", false),
        ]);
        assert_eq!(
            crate::tools::load_tool::extract_loaded_tool_names_from_events(
                log.events(),
                crate::provider::DeferredToolDelivery::Appended
            ),
            vec!["omega".to_string(), "alpha".to_string()]
        );
    }

    #[test]
    fn extract_loaded_tool_names_recovers_snapshot_across_boundary() {
        // Pre-boundary: tool_load(scratchpad_read) succeeds. After the boundary swallows it, the
        // snapshot must restore scratchpad_read in the active set.
        let mut log = Conversation::new();
        log.append(load_tool_use("u1", "scratchpad_read"));
        log.append(load_tool_result("u1", false));

        let snapshot: std::collections::BTreeSet<String> =
            ["scratchpad_read".to_string()].into_iter().collect();
        log.replace_for_compaction(
            Message::user("[summary]"),
            Vec::new(),
            snapshot,
            Default::default(),
            CompactionRecord::for_test(),
        );

        let loaded = crate::tools::load_tool::extract_loaded_tool_names_from_events(
            log.events(),
            crate::provider::DeferredToolDelivery::Appended,
        );
        assert!(loaded.iter().any(|name| name == "scratchpad_read"));
    }

    #[test]
    fn prune_compacted_events_drops_pre_boundary_log() {
        let mut log = Conversation::new();
        log.append(load_tool_use("u1", "scratchpad_read"));
        log.append(load_tool_result("u1", false));
        log.append(Message::user("m1"));

        let snapshot: std::collections::BTreeSet<String> =
            ["scratchpad_read".to_string()].into_iter().collect();
        log.replace_for_compaction(
            Message::user("[summary-1]"),
            vec![Message::user("tail-1")],
            snapshot.clone(),
            Default::default(),
            CompactionRecord::for_test(),
        );
        log.append(Message::assistant_text("m2"));
        log.replace_for_compaction(
            Message::user("[summary-2]"),
            vec![Message::user("tail-2")],
            snapshot,
            Default::default(),
            CompactionRecord::for_test(),
        );

        let view_before: Vec<String> = log.as_slice().iter().map(|m| m.text_content()).collect();
        let loaded_before = crate::tools::load_tool::extract_loaded_tool_names_from_events(
            log.events(),
            crate::provider::DeferredToolDelivery::Appended,
        );

        log.prune_compacted_events();

        // Materialized view and recovered tool set are unchanged.
        let view_after: Vec<String> = log.as_slice().iter().map(|m| m.text_content()).collect();
        assert_eq!(view_before, view_after);
        assert_eq!(
            loaded_before,
            crate::tools::load_tool::extract_loaded_tool_names_from_events(
                log.events(),
                crate::provider::DeferredToolDelivery::Appended
            )
        );
        assert!(
            crate::tools::load_tool::extract_loaded_tool_names_from_events(
                log.events(),
                crate::provider::DeferredToolDelivery::Appended
            )
            .iter()
            .any(|name| name == "scratchpad_read"),
            "deferred tool must survive the prune"
        );

        // The log now starts at the last boundary; nothing precedes it.
        assert!(matches!(
            log.events().first(),
            Some(Event::CompactBoundary { .. })
        ));
        let boundary_count = log
            .events()
            .iter()
            .filter(|e| matches!(e, Event::CompactBoundary { .. }))
            .count();
        assert_eq!(boundary_count, 1, "only the last boundary should remain");
    }

    /// A reference in a result is a load whichever tool answered, which is how `tool_search` loads
    /// on a declaring wire. Across a boundary the deliveries part: an appending wire's array
    /// carries the snapshot forward, a declaring wire's references went with the window, so only
    /// the loads after it stand.
    #[test]
    fn extract_loaded_tool_names_reads_references_and_a_declaring_wire_forgets_them_at_a_boundary()
    {
        use crate::provider::DeferredToolDelivery;
        let mut log = Conversation::new();
        log.append(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "s1".to_string(),
                name: "tool_search".to_string(),
                input: serde_json::json!({"names": ["mcp__kv__alpha"]}),
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        });
        log.append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "s1".to_string(),
                content: vec![ToolResultContent::ToolReference {
                    tool_name: "mcp__kv__alpha".to_string(),
                }],
                is_error: false,
                display: None,
            }],
            provider_fields: None,
        });
        let before = extract(&log, DeferredToolDelivery::Appended);
        assert_eq!(before, vec!["mcp__kv__alpha"]);
        assert_eq!(extract(&log, DeferredToolDelivery::Searched), before);

        log.replace_for_compaction(
            Message::user("[summary]"),
            Vec::new(),
            ["mcp__kv__alpha", "mcp__kv__beta"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            Default::default(),
            CompactionRecord::for_test(),
        );
        log.append(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "l1".to_string(),
                name: "tool_load".to_string(),
                input: serde_json::json!({"name": "mcp__kv__gamma"}),
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        });
        log.append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "l1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "loaded".to_string(),
                }],
                is_error: false,
                display: None,
            }],
            provider_fields: None,
        });
        assert_eq!(extract(&log, DeferredToolDelivery::Appended), vec![
            "mcp__kv__alpha",
            "mcp__kv__beta",
            "mcp__kv__gamma"
        ]);
        assert_eq!(extract(&log, DeferredToolDelivery::Referenced), vec![
            "mcp__kv__gamma"
        ]);
        assert_eq!(extract(&log, DeferredToolDelivery::Searched), vec![
            "mcp__kv__gamma"
        ]);

        fn extract(log: &Conversation, delivery: DeferredToolDelivery) -> Vec<String> {
            crate::tools::load_tool::extract_loaded_tool_names_from_events(log.events(), delivery)
        }
    }

    #[test]
    fn extract_loaded_tool_names_pending_use_wiped_at_boundary() {
        // tool_load tool_use lives on one side of the boundary, its tool_result on the other; both
        // vanish from the materialized view, so the scanner must NOT count the pending pair across
        // the boundary.
        let mut log = Conversation::new();
        log.append(load_tool_use("u1", "scratchpad_read"));
        // No tool_result yet.
        log.replace_for_compaction(
            Message::user("[summary]"),
            vec![load_tool_result("u1", false)],
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );

        let loaded = crate::tools::load_tool::extract_loaded_tool_names_from_events(
            log.events(),
            crate::provider::DeferredToolDelivery::Appended,
        );
        assert!(!loaded.iter().any(|name| name == "scratchpad_read"));
    }

    #[test]
    fn pop_unsaved_only_removes_trailing_append() {
        // After a CompactBoundary, the next legal call is `append`. A failed-save rollback after
        // that should remove the failed append, not the boundary.
        let mut log = Conversation::new();
        log.append(Message::user("pre"));
        log.replace_for_compaction(
            Message::user("[summary]"),
            Vec::new(),
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );
        log.append(Message::user("post-comp"));

        let popped = log.pop_unsaved();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().text_content(), "post-comp");

        // Boundary's summary survives.
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "[summary]");

        // Calling pop_unsaved again must NOT eat the boundary.
        assert!(log.pop_unsaved().is_none());
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn from_vec_produces_append_events() {
        let log = Conversation::from_vec(vec![Message::user("a"), Message::assistant_text("b")]);
        assert_eq!(log.events().len(), 2);
        assert!(log.events().iter().all(|e| matches!(e, Event::Append(_))));
    }

    fn record(
        trigger: CompactionTrigger,
        source: CompactionSource,
        memories: &[&str],
        kept_recent: bool,
    ) -> CompactionRecord {
        CompactionRecord {
            trigger,
            source,
            memories_written: memories.iter().map(|name| name.to_string()).collect(),
            kept_recent,
            written_summary: String::new(),
            checkpoint: Vec::new(),
            summarizer: Vec::new(),
        }
    }

    /// A turn that ran a compaction alone ends as its last reply stopped, the summarizer's ahead of
    /// the checkpoint's, except that a round that stopped for its calls ended on nothing left to
    /// run, since the compaction ran them; a reply that recorded no stop ended as an answer.
    #[test]
    fn a_compaction_turn_ends_as_its_last_reply_stopped() {
        let replied = |reason: Option<StopReason>| RecordedMessage {
            message: Message::assistant_text("words"),
            reply: Some(ReplyRecord {
                stop: reason.map(Stop::of),
                cut: None,
                model: None,
                profile: None,
                usage: None,
                provider_fields: None,
            }),
        };
        let mut compaction = record(
            CompactionTrigger::Manual,
            CompactionSource::Checkpoint,
            &[],
            true,
        );
        assert_eq!(compaction.last_stop(), StopReason::EndTurn);
        compaction.checkpoint = vec![
            replied(Some(StopReason::MaxTokens)),
            replied(Some(StopReason::ToolUse)),
        ];
        assert_eq!(compaction.last_stop(), StopReason::EndTurn);
        compaction.summarizer = vec![replied(Some(StopReason::Refusal))];
        assert_eq!(compaction.last_stop(), StopReason::Refusal);
        compaction.summarizer = vec![replied(None)];
        assert_eq!(compaction.last_stop(), StopReason::EndTurn);
    }

    /// The note names what the checkpoint wrote, because a memory is durable and instance-scoped:
    /// notes accumulating unmentioned under an event called a compaction is the surprise this
    /// reporting exists to prevent.
    #[test]
    fn compaction_note_names_memories_written() {
        let line = record(
            CompactionTrigger::Manual,
            CompactionSource::Checkpoint,
            &["deploy-quirks", "rate-limits"],
            true,
        )
        .note();
        assert_eq!(
            line,
            "Compacted the conversation on user request. Wrote 2 memories: deploy-quirks, \
             rate-limits."
        );
    }

    /// Each trigger has its own sentence, and the ordinary case adds nothing to it.
    #[test]
    fn compaction_note_says_what_set_the_compaction_going() {
        let note = |trigger| record(trigger, CompactionSource::Checkpoint, &[], true).note();
        assert_eq!(
            note(CompactionTrigger::Reactive),
            "Autocompacted the conversation after context usage crossed the ceiling."
        );
        assert_eq!(
            note(CompactionTrigger::Proactive),
            "Autocompacted the conversation ahead of a turn that would have crossed the ceiling."
        );
        assert_eq!(
            note(CompactionTrigger::Requested),
            "The agent compacted the conversation."
        );
        assert_eq!(
            note(CompactionTrigger::Emergency),
            "Compacted the conversation after the provider refused the request as too large."
        );
        assert_eq!(
            note(CompactionTrigger::Retention),
            "Autocompacted the conversation to summarize turns older than the retention window."
        );
    }

    /// Both departures from the ordinary case are named. A user comparing one compaction against
    /// another needs to know the summary was not the one the agent chose to write.
    #[test]
    fn compaction_note_reports_a_fallback_and_a_discarded_tail() {
        let line = record(
            CompactionTrigger::Emergency,
            CompactionSource::Summarizer,
            &[],
            false,
        )
        .note();
        assert_eq!(
            line,
            "Compacted the conversation after the provider refused the request as too large. No \
             recent turns were kept. Written by the standalone summarizer."
        );
    }

    /// The record is four keys beside the boundary's own, which is the shape the migration module
    /// writes onto an older row, and each spelling is the one the value enums name.
    #[test]
    fn a_boundary_carries_its_record_as_flat_keys() {
        let event = Event::CompactBoundary {
            summary: Message::user("summary"),
            replaced_count: 2,
            loaded_tools_snapshot: Default::default(),
            checklist_snapshot: Default::default(),
            record: CompactionRecord {
                trigger: CompactionTrigger::Requested,
                source: CompactionSource::Summarizer,
                memories_written: vec!["notes".to_string()],
                kept_recent: false,
                written_summary: "what was done".to_string(),
                checkpoint: Vec::new(),
                summarizer: Vec::new(),
            },
        };
        let encoded = serde_json::to_value(&event).expect("encodes");
        let boundary = &encoded["CompactBoundary"];
        assert_eq!(boundary["trigger"], "requested");
        assert_eq!(boundary["source"], "summarizer");
        assert_eq!(boundary["memories_written"], serde_json::json!(["notes"]));
        assert_eq!(boundary["kept_recent"], false);
        assert_eq!(boundary["written_summary"], "what was done");
        assert_eq!(
            boundary["summary"]["role"], "user",
            "the message keeps its own key"
        );
        let decoded: Event = serde_json::from_value(encoded).expect("decodes");
        let Event::CompactBoundary { record, .. } = decoded else {
            panic!("a boundary decodes as one");
        };
        assert_eq!(record.trigger, CompactionTrigger::Requested);
        assert_eq!(record.source, CompactionSource::Summarizer);
        assert!(
            serde_json::from_str::<Event>(
                r#"{"CompactBoundary":{"summary":{"role":"user","content":[]},"replaced_count":1,"loaded_tools_snapshot":[],"checklist_snapshot":{"next_id":1,"items":[]}}}"#
            )
            .is_err(),
            "a boundary without its record is not one this build reads; the migration writes it"
        );
    }

    #[test]
    fn event_serializes_round_trip() {
        // Serialize one of each event variant and round-trip through JSON.
        let append = Event::Append(Message::user("hi"));
        let json = serde_json::to_string(&append).expect("serialize append");
        let back: Event = serde_json::from_str(&json).expect("deserialize append");
        match back {
            Event::Append(m) => assert_eq!(m.text_content(), "hi"),
            _ => panic!("wrong variant"),
        }

        let snapshot: std::collections::BTreeSet<String> =
            ["mcp__notion__fetch".to_string()].into_iter().collect();
        let boundary = Event::CompactBoundary {
            summary: Message::user("[summary]"),
            replaced_count: 5,
            record: crate::conversation::CompactionRecord::for_test(),
            loaded_tools_snapshot: snapshot,
            checklist_snapshot: Default::default(),
        };
        let json = serde_json::to_string(&boundary).expect("serialize boundary");
        let back: Event = serde_json::from_str(&json).expect("deserialize boundary");
        match back {
            Event::CompactBoundary {
                replaced_count,
                loaded_tools_snapshot,
                ..
            } => {
                assert_eq!(replaced_count, 5);
                assert!(loaded_tools_snapshot.contains("mcp__notion__fetch"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn sanitize_orphans_does_not_touch_compact_boundary() {
        let mut log = Conversation::new();
        log.append(Message::user("u1"));
        log.append(Message::assistant_text("a1"));
        log.replace_for_compaction(
            Message::user("[summary]"),
            Vec::new(),
            Default::default(),
            Default::default(),
            CompactionRecord::for_test(),
        );
        // Synthetic summary is a plain user message; sanitize must leave it.
        log.sanitize_orphans();
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "[summary]");
    }

    /// One spelling per retention on every surface: `name()` round-trips through `FromStr`,
    /// `Display` and serde, and anything else is refused naming what would have been accepted.
    #[test]
    fn every_retention_round_trips_through_its_one_spelling_and_no_other() {
        for retention in PromptRetention::ALL {
            assert_eq!(retention.name().parse::<PromptRetention>(), Ok(retention));
            assert_eq!(retention.to_string(), retention.name());
            let json = serde_json::to_string(&retention).expect("serialize");
            assert_eq!(json, format!("\"{}\"", retention.name()));
            assert_eq!(
                serde_json::from_str::<PromptRetention>(&json).expect("deserialize"),
                retention
            );
            assert!(
                retention
                    .name()
                    .to_uppercase()
                    .parse::<PromptRetention>()
                    .is_err(),
                "{} must be the only spelling of {retention}",
                retention.name()
            );
        }
        let refusal = "drop"
            .parse::<PromptRetention>()
            .expect_err("not a retention");
        assert!(
            refusal.contains("keep, withdraw"),
            "the refusal lists what would have been accepted: {refusal}"
        );
    }

    /// One spelling per change line's kind, per nudge kind and per role, the one the store and
    /// archives already hold, and no other.
    #[test]
    fn every_change_line_kind_nudge_kind_and_role_has_its_one_spelling() {
        crate::text::assert_one_spelling(
            &ChangeKind::ALL,
            &["kept", "removed", "added"],
            "changed",
            "kept, removed, added",
        );
        crate::text::assert_one_spelling(
            &NudgeKind::ALL,
            &["visible_reply", "checklist"],
            "visibleReply",
            "visible_reply, checklist",
        );
        crate::text::assert_one_spelling(
            &Role::ALL,
            &["user", "assistant"],
            "system",
            "user, assistant",
        );
    }

    fn user_msg(text: &str) -> crate::conversation::Message {
        crate::conversation::Message::user(text)
    }

    fn assistant_text(text: &str) -> crate::conversation::Message {
        crate::conversation::Message::assistant_text(text)
    }

    fn assistant_tool_use(id: &str, name: &str) -> crate::conversation::Message {
        crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![crate::conversation::ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({}),
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        }
    }

    fn tool_result(tool_use_id: &str) -> crate::conversation::Message {
        crate::conversation::Message {
            role: crate::conversation::Role::User,
            content: vec![crate::conversation::ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![crate::conversation::ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
                display: None,
            }],
            provider_fields: None,
        }
    }

    fn build_log(messages: Vec<crate::conversation::Message>) -> crate::conversation::Conversation {
        crate::conversation::Conversation::from_vec(messages)
    }

    #[test]
    fn validate_valid_chain() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "file_read"),
            tool_result("c1"),
            assistant_text("done"),
        ]);
        let dropped = log.sanitize_orphans();
        assert!(dropped.is_empty());
        assert_eq!(log.len(), 4);
    }

    #[test]
    fn validate_orphaned_tool_use_dropped() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "file_read"),
            // Missing tool_result for c1
            assistant_text("done"),
        ]);
        let dropped = log.sanitize_orphans();
        assert_eq!(dropped.len(), 1);
        assert_eq!(log.len(), 2);
        let view = log.as_slice();
        assert_eq!(view[0].role, crate::conversation::Role::User);
        assert_eq!(view[1].role, crate::conversation::Role::Assistant);
        assert_eq!(view[1].text_content(), "done");
    }

    #[test]
    fn validate_orphaned_at_end() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "file_read"),
        ]);
        log.sanitize_orphans();
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "hello");
    }

    #[test]
    fn validate_mismatched_ids() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "file_read"),
            tool_result("c2"), // Wrong ID
        ]);
        log.sanitize_orphans();
        // The assistant message is dropped because c1 has no matching result.
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn validate_text_only_preserved() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_text("hi"),
            user_msg("bye"),
        ]);
        log.sanitize_orphans();
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn validate_multiple_chains() {
        let mut log = build_log(vec![
            user_msg("start"),
            assistant_tool_use("c1", "file_read"),
            tool_result("c1"),
            assistant_tool_use("c2", "file_write"),
            // Missing tool_result for c2
            assistant_text("done"),
        ]);
        log.sanitize_orphans();
        // c2 should be dropped, rest preserved.
        assert_eq!(log.len(), 4);
        assert_eq!(log.as_slice()[3].text_content(), "done");
    }

    #[test]
    fn export_without_compaction_renders_plain_turns() {
        let mut log = crate::conversation::Conversation::new();
        log.append(user_msg("hello"));
        log.append(assistant_text("hi there"));
        let markdown = crate::conversation::format_session_as_markdown(
            uuid::Uuid::nil(),
            log.events().iter().map(|event| (event, None)),
            &std::collections::HashMap::new(),
        );
        assert!(markdown.contains("## User") && markdown.contains("hello"));
        assert!(markdown.contains("## Assistant") && markdown.contains("hi there"));
        // No compaction happened, so no boundary marker.
        assert!(!markdown.contains("Session compaction"));
    }
}
