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

use std::collections::HashSet;

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
    message.role == Role::User
        && !message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
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
    /// can recover them after the boundary (otherwise compaction would silently un-load them).
    CompactBoundary {
        summary: Message,
        replaced_count: usize,
        loaded_tools_snapshot: HashSet<String>,
    },
    /// Replaces the last `replaced_count` materialized messages with `messages`. An empty
    /// `messages` therefore means "drop them", which is what [`Conversation::rewind`] emits.
    ///
    /// Position-relative rather than index-addressed, deliberately: like
    /// [`Self::CompactBoundary`] it replays correctly no matter what precedes it, so
    /// [`Conversation::sanitize_orphans`] removing an earlier event can't silently retarget it.
    /// The corollary is an invariant on producers: emit it only while the messages it replaces are
    /// still the trailing materialized entries.
    Repair {
        replaced_count: usize,
        messages: Vec<Message>,
    },
    /// Replaces the images at `images` with [`IMAGE_REDACTION_PLACEHOLDER`], once, so the body
    /// that fit the request budget is the body every later request sends and the cache prefix
    /// ahead of the newest turn stops moving. Redacting per request instead would move it on every
    /// send, since each request re-derives the set from scratch.
    ///
    /// Tail-relative like [`Self::Repair`], and for the same reason: the producer,
    /// `Agent::run_turn`, records it before appending the round's own messages, so the view it
    /// addresses is the view the request was built from.
    Redact { images: Vec<RedactedImage> },
}

pub(crate) use crate::image::RedactedImage;

/// Placeholder text that replaces an image payload when the request body would otherwise exceed
/// the ceiling.
pub(crate) const IMAGE_REDACTION_PLACEHOLDER: &str = "[image redacted to fit request size budget]";

/// Characters a session title keeps before it is cut.
pub(crate) const TITLE_CHARS: usize = 80;

/// Append-only conversation: an event log, plus the materialized message view derived from it.
#[derive(Debug, Default, Clone)]
pub(crate) struct Conversation {
    events: Vec<Event>,
    /// Materialized view kept in lockstep with `events`. Rebuilt by `rebuild_materialized` after
    /// every mutation; reads are zero-cost.
    materialized: Vec<Message>,
    /// Images replaced since the last full rebuild, for
    /// [`Conversation::invalid_images_replaced`].
    invalid_images_replaced: usize,
    /// Whether this log came off disk and the model has not been told yet.
    ///
    /// Set by [`Self::from_events`] and consumed once by [`Self::take_resumed_notice`]. Lives here
    /// rather than on the agent because the conversation is the thing that was hydrated, and
    /// hydration has four entry points (REPL resume, two ACP paths, serve reattach) that all reach
    /// `from_events`. A flag set at those call sites instead would be a flag the fifth one
    /// forgets.
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
    /// nothing. The words are the first non-blank `Text` block of a user `Append` that is not a
    /// stand-in meka wrote ([`is_harness_stand_in`]); a turn's context block is not a `Text`
    /// block. The store's `title_of_first_user_row` selects the same row in SQL. A log pruned by
    /// [`Self::prune_compacted_events`] starts at its last boundary and no longer holds the first
    /// turn; every surface that labels a stored session reads the store's rows, which do.
    pub(crate) fn title(&self) -> String {
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::Append(message) if message.role == Role::User => Some(message),
                _ => None,
            })
            .flat_map(|message| &message.content)
            .find_map(|block| match block {
                ContentBlock::Text { text } if !is_harness_stand_in(text) => {
                    let words = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    (!words.is_empty()).then_some(words)
                }
                _ => None,
            })
            .map(|words| {
                if words.chars().count() <= TITLE_CHARS {
                    words
                } else {
                    let kept: String = words.chars().take(TITLE_CHARS).collect();
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
    /// (permission, cwd, todos, and the tool catalog are all restated every turn regardless).
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
    pub(crate) fn ends_on_a_turn_opening(&self) -> bool {
        matches!(self.events.last(), Some(Event::Append(_)))
            && self.materialized.last().is_some_and(opens_turn)
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, Message> {
        self.materialized.iter()
    }

    /// Text content of the most recent `Role::Assistant` message, or `None` when no assistant
    /// message exists. Walks backward, which is necessary because a turn that ended via tool-use
    /// leaves a `Role::User` tool-result trailer in the conversation, hiding the assistant's
    /// final text from a look at the last message alone.
    pub(crate) fn last_assistant_text(&self) -> Option<String> {
        self.materialized
            .iter()
            .rev()
            .find(|message| matches!(message.role, crate::conversation::Role::Assistant))
            .map(|message| message.text_content())
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
        loaded_tools_snapshot: HashSet<String>,
    ) {
        let replaced_count = self.materialized.len();
        self.events.push(Event::CompactBoundary {
            summary,
            replaced_count,
            loaded_tools_snapshot,
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
    /// leave the model reasoning from a summary the database has never heard of, while
    /// `GET /messages` still serves the full history with `revision` unmoved. The caller was told
    /// the compaction failed, so the two must agree that it did.
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

    /// Record that a request budget redacted `images`, replacing each with the placeholder in the
    /// view. Returns the event so the caller can persist it; see [`Event::Redact`] for why it is
    /// recorded rather than redone per request.
    pub(crate) fn redact_images(&mut self, images: Vec<RedactedImage>) -> Event {
        let event = Event::Redact { images };
        self.events.push(event.clone());
        self.rebuild_materialized();
        event
    }

    /// Undo the most recent [`Self::replace_tail`], restoring the messages it replaced.
    ///
    /// The inverse of [`Self::pop_unsaved`] for repairs: `run_turn` degrades content, retries, and
    /// calls this when the retry fails too, so a misdiagnosed rejection leaves the conversation
    /// byte-identical instead of permanently losing a good tool result. Returns whether anything
    /// was undone; a trailing event that isn't a `Repair` is left alone.
    pub(crate) fn pop_repair(&mut self) -> bool {
        if !matches!(self.events.last(), Some(Event::Repair { .. })) {
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
    /// so a `tool_use` is never separated from its `tool_result`; the rounds a compaction keeps
    /// from inside a turn open no turn, so a rewind past them takes the summary before them
    /// too. A compaction summary is a plain `User` message and so counts as one such
    /// boundary: rewinding far enough past a compaction discards the summary too, which is right
    /// (it stands in for the turns before it) but means a big `turns` can empty a compacted session
    /// faster than the turn count suggests.
    pub(crate) fn rewind(&mut self, turns: usize) -> Option<Event> {
        if turns == 0 {
            return None;
        }
        let cut = self
            .materialized
            .iter()
            .enumerate()
            .filter(|(_, message)| opens_turn(message))
            .map(|(index, _)| index)
            .nth_back(turns - 1)?;
        let replaced_count = self.materialized.len() - cut;
        Some(self.replace_tail(replaced_count, Vec::new()))
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
    /// so the on-disk log stays complete.
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
    /// Removes the corresponding `Event::Append` entries from the event log so future
    /// re-materializations stay clean. `Event::CompactBoundary` events are never touched (their
    /// synthetic summary is a plain user message that can't be orphaned).
    pub(crate) fn sanitize_orphans(&mut self) -> Vec<Message> {
        let dropped_indices = orphan_event_indices(&self.events);
        if dropped_indices.is_empty() {
            return Vec::new();
        }

        let mut dropped = Vec::with_capacity(dropped_indices.len());
        // Highest index first, so each removal leaves the earlier indices valid; `remove` keeps
        // the log's order, and the reverse below restores the dropped messages' own.
        let mut to_remove = dropped_indices;
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for index in to_remove {
            if let Event::Append(message) = self.events.remove(index) {
                dropped.push(message);
            }
        }
        dropped.reverse();
        self.rebuild_materialized();
        dropped
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
        let (placed, _) = replay(self.events.iter());
        self.materialized = placed.into_iter().map(|placed| placed.message).collect();
        self.invalid_images_replaced = repair_invalid_images(&mut self.materialized);
    }
}

/// The event a materialized message came from.
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
    marker: Option<CompactionMarker>,
}

/// Which compaction a summary stands for, as a view of the log reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompactionMarker {
    /// How many materialized messages the boundary recorded replacing.
    pub(crate) replaced_count: usize,
    /// Which compaction produced it, counting from 1.
    pub(crate) generation: u64,
}

/// Replay `events` into the messages the model sees, with each one's origin, and the count of
/// rewrites (boundaries and repairs) the log has been through.
///
/// The one statement of the rules. A boundary replaces everything before it: its producer,
/// [`Conversation::replace_for_compaction`], records the whole view as `replaced_count`, but that
/// number was measured against the view in memory, which a resume that dropped orphans has made
/// shorter than what the store replays. Truncating by the count would leave the difference
/// standing above the summary, and every later compaction would widen it. A repair is
/// position-relative by design, since it replaces a trailing run its producer just appended.
fn replay<'a>(events: impl Iterator<Item = &'a Event>) -> (Vec<Placed>, u64) {
    let mut placed: Vec<Placed> = Vec::new();
    let mut generation: u64 = 0;
    let mut revision: u64 = 0;
    for (index, event) in events.enumerate() {
        match event {
            Event::Append(message) => placed.push(Placed {
                message: message.clone(),
                source: Source::Append(index),
                marker: None,
            }),
            Event::CompactBoundary {
                summary,
                replaced_count,
                ..
            } => {
                revision = revision.saturating_add(1);
                generation = generation.saturating_add(1);
                placed.clear();
                placed.push(Placed {
                    message: summary.clone(),
                    source: Source::Boundary(index),
                    marker: Some(CompactionMarker {
                        replaced_count: *replaced_count,
                        generation,
                    }),
                });
            }
            Event::Repair {
                replaced_count,
                messages,
            } => {
                revision = revision.saturating_add(1);
                let truncate_to = placed.len().saturating_sub(*replaced_count);
                placed.truncate(truncate_to);
                placed.extend(messages.iter().map(|message| Placed {
                    message: message.clone(),
                    source: Source::Repair(index),
                    marker: None,
                }));
            }
            Event::Redact { images } => {
                revision = revision.saturating_add(1);
                for image in images {
                    let Some(at) = placed.len().checked_sub(image.from_end) else {
                        continue;
                    };
                    if let Some(placed) = placed.get_mut(at) {
                        redact_image(&mut placed.message, image);
                    }
                }
            }
        }
    }
    (placed, revision)
}

/// Replace the image `image` names in `message` with the placeholder. A position that names
/// something else is left alone: the event was recorded against a view this one may not be, and
/// removing text on a stale address would be worse than keeping an image.
fn redact_image(message: &mut Message, image: &RedactedImage) {
    let placeholder = || IMAGE_REDACTION_PLACEHOLDER.to_string();
    match (message.content.get_mut(image.block), image.item) {
        (Some(block @ ContentBlock::Image { .. }), None) => {
            *block = ContentBlock::Text {
                text: placeholder(),
            };
        }
        (Some(ContentBlock::ToolResult { content, .. }), Some(item)) => {
            if let Some(entry @ ToolResultContent::Image { .. }) = content.get_mut(item) {
                *entry = ToolResultContent::Text {
                    text: placeholder(),
                };
            }
        }
        _ => {}
    }
}

/// The replayed log with, for each message, when it was written and whether it is a compaction
/// summary. Three vectors of one length, which every arm of the replay maintains.
pub(crate) struct AnnotatedView {
    pub(crate) messages: Vec<Message>,
    pub(crate) timestamps: Vec<String>,
    pub(crate) markers: Vec<Option<CompactionMarker>>,
    /// How many times the log was rewritten: every boundary and every repair.
    pub(crate) revision: u64,
}

/// What `GET /v1/sessions/{id}/messages` serves: the same view the model gets, by the same rules,
/// with each message stamped from the row that produced it. A summary takes its boundary's time
/// and a repair's replacements take the repair's, because that is when the content came to be.
pub(crate) fn materialize_annotated(events: &[(String, Event)]) -> AnnotatedView {
    let (placed, revision) = replay(events.iter().map(|(_, event)| event));
    let mut messages = Vec::with_capacity(placed.len());
    let mut timestamps = Vec::with_capacity(placed.len());
    let mut markers = Vec::with_capacity(placed.len());
    for placed in placed {
        let index = match placed.source {
            Source::Append(index) | Source::Boundary(index) | Source::Repair(index) => index,
        };
        timestamps.push(events[index].0.clone());
        markers.push(placed.marker);
        messages.push(placed.message);
    }
    repair_invalid_images(&mut messages);
    AnnotatedView {
        messages,
        timestamps,
        markers,
        revision,
    }
}

/// Prefix on every note meka writes *into the conversation itself*, as opposed to onto the screen.
///
/// It answers one question the surrounding blocks cannot: who wrote this. A note replacing a tool
/// result sits where the tool's own output would be, and one replacing an attachment sits in the
/// user's message, so without a marker the model has to guess whether the tool said it, the user
/// said it, or something between them did.
///
/// Names the harness in words rather than only by product, because a bare `[meka]` is legible only
/// to a model that recalls what meka is from the system prompt and infers the rest. `[system]`
/// would read better still and is the wrong choice: nothing strips markers out of a tool result, so
/// a fetched page or an MCP server's output can contain any string this does, and a marker models
/// already obey on sight is the worst one to make load-bearing.
pub(crate) const HARNESS_NOTE: &str = "[meka harness]";

/// Whether `text` is something meka put into a user message in place of the user's own content:
/// the placeholder a redaction leaves or a harness note. Both are `Text` blocks, because that is
/// the one shape every provider renders, so a reader after what the user typed has to ask.
pub(crate) fn is_harness_stand_in(text: &str) -> bool {
    text == IMAGE_REDACTION_PLACEHOLDER || text.starts_with(HARNESS_NOTE)
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
            "{HARNESS_NOTE} An image here was removed: it is declared {} but the bytes are \
             something else, which the provider refuses.",
            source.media_type()
        )
    };

    let mut replaced = 0usize;
    for message in messages {
        for block in &mut message.content {
            match block {
                ContentBlock::Image { source } if mismatched(source) => {
                    replaced += 1;
                    *block = ContentBlock::Text { text: note(source) };
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
                            *item =
                                crate::conversation::ToolResultContent::Text { text: note(source) };
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

/// Walk the event log and return the indices of `Event::Append` entries that carry orphaned
/// assistant `tool_use` blocks (i.e. no matching `tool_result` in the next materialized message).
/// The check uses the *materialized* view so a `CompactBoundary` between an orphan and its would-be
/// result correctly counts as orphaned.
fn orphan_event_indices(events: &[Event]) -> Vec<usize> {
    // `(event_index, &Message)` pairs in materialization order, so the adjacency scan can report
    // event indices rather than materialized ones. An `Append` a `CompactBoundary` truncated
    // away is not visited, since the materialized view never sees that orphan.
    //
    // The index is `None` for messages that don't come from an `Append` event (a repair's
    // replacement). They still take part in the adjacency scan, since a `tool_result` inside one
    // answers the `tool_use` before it, but they can't be removed.
    let (placed, _) = replay(events.iter());
    let pairs: Vec<(Option<usize>, &Message)> = placed
        .iter()
        .map(|placed| {
            (
                match placed.source {
                    Source::Append(index) => Some(index),
                    Source::Boundary(_) | Source::Repair(_) => None,
                },
                &placed.message,
            )
        })
        .collect();

    let mut orphan = Vec::new();
    for window_index in 0..pairs.len() {
        let (Some(event_index), message) = pairs[window_index] else {
            continue;
        };
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

        let next = pairs.get(window_index + 1).map(|(_, m)| *m);
        let has_results = next.is_some_and(|next_message| {
            next_message.role == Role::User
                && tool_use_ids.iter().all(|id| {
                    next_message.content.iter().any(|block| {
                        matches!(block, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == *id)
                    })
                })
        });

        if !has_results {
            orphan.push(event_index);
        }
    }
    orphan
}

/// Who a message is from, as every provider's wire spells it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Role {
    User,
    Assistant,
}
/// One item of a tool result: text, or an image the tool produced.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolResultContent {
    Text { text: String },
    Image { source: ImageSource },
}
/// The half of a thinking block meka cannot read, and which provider it belongs to.
///
/// The two backends that emit reasoning hand back different things, and the difference decides both
/// what may be replayed and what the readable half is worth. Held as one nullable `signature`,
/// wrong states are representable: a `chatgpt-subscription` session's `encrypted_content` resumed
/// under Claude is replayed verbatim as Claude's `signature`, a blob from the wrong cryptosystem
/// presented as authentication for text it does not authenticate, and the other direction opens
/// the moment the Responses encoder replays reasoning. Naming the two shapes makes both a type
/// error instead of a thing to remember.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OpaqueReasoning {
    /// Anthropic. The reasoning is in the block's `thinking` text; this authenticates it, and the
    /// API wants both back together.
    Signed { signature: String },
    /// The Responses API. This *is* the reasoning, sealed, and the block's `thinking` holds only
    /// the summary the server chose to show. Replayed under `id` when the server issued one.
    Sealed {
        encrypted_content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
}
/// One block of a message, in the shape the session store persists and every provider maps from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ContentBlock {
    Text {
        text: String,
    },
    /// What meka injected ahead of the user's words for this turn: the permission and environment
    /// context, todos, world state, budget, background outcomes and the resume notice. Its own
    /// block, so every reader that shows what the user typed can skip it, and every provider
    /// renders it as text ahead of the words. A user message carries zero or one of these, first.
    TurnContext {
        text: String,
    },
    /// Image supplied as *input* (e.g. an ACP client's @-mention or pasted screenshot). Distinct
    /// from a tool result's image, which travels inside [`ContentBlock::ToolResult`] as a
    /// [`ToolResultContent::Image`].
    Image {
        source: ImageSource,
    },
    Thinking {
        /// The readable half, and only the readable half. How much of the reasoning it is depends
        /// on `opaque`: under [`OpaqueReasoning::Signed`] this is the reasoning, under
        /// [`OpaqueReasoning::Sealed`] it is a summary of reasoning kept elsewhere.
        thinking: String,
        /// What the provider wants handed back to carry this reasoning into the next request.
        /// `None` when it gave nothing to carry, which makes the block display-only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        opaque: Option<OpaqueReasoning>,
    },
    /// Encrypted reasoning the API declines to return in the clear (the `redact-thinking` beta).
    /// `data` is opaque: it cannot be read, only replayed verbatim on later turns so the model can
    /// continue its prior reasoning chain. Distinct from a [`ContentBlock::Thinking`] with empty
    /// text, which carries a `signature` instead of `data`.
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultContent>,
        is_error: bool,
    },
}
impl ContentBlock {
    /// Extract the text content of a ToolResult (for display/logging).
    pub(crate) fn tool_result_text_content(content: &[ToolResultContent]) -> String {
        content
            .iter()
            .map(|block| match block {
                ToolResultContent::Text { text } => text.as_str(),
                ToolResultContent::Image { .. } => "[Image]",
            })
            .collect::<Vec<_>>()
            .join("")
    }
}
/// One message of the conversation: who sent it and what it carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Message {
    pub(crate) role: Role,
    pub(crate) content: Vec<ContentBlock>,
}
impl Message {
    /// A user message carrying only `text`.
    pub(crate) fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// The user message a turn appends: meka's context block, the words as typed, then any input
    /// images. The words are left out when there are none (a turn that only delivers background
    /// outcomes), so [`Self::text_content`] stays what the user typed.
    pub(crate) fn user_turn(
        context: impl Into<String>,
        words: impl Into<String>,
        images: Vec<ImageSource>,
    ) -> Self {
        let words = words.into();
        let mut content = vec![ContentBlock::TurnContext {
            text: context.into(),
        }];
        if !words.is_empty() {
            content.push(ContentBlock::Text { text: words });
        }
        content.extend(
            images
                .into_iter()
                .map(|source| ContentBlock::Image { source }),
        );
        Self {
            role: Role::User,
            content,
        }
    }

    /// User message carrying a text block followed by zero or more input images; `images` empty
    /// yields the same shape as [`Message::user`].
    #[cfg(test)]
    pub(crate) fn user_with_images(text: impl Into<String>, images: Vec<ImageSource>) -> Self {
        let mut content = vec![ContentBlock::Text { text: text.into() }];
        content.extend(
            images
                .into_iter()
                .map(|source| ContentBlock::Image { source }),
        );
        Self {
            role: Role::User,
            content,
        }
    }

    /// An assistant message carrying only `text`.
    pub(crate) fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// The words: every `Text` block joined, and nothing else. A turn's context block is not
    /// something the user wrote and is left out; [`Self::wire_text`] is the whole.
    pub(crate) fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// The `Text` blocks as paragraphs, a blank line between them. A user message's second `Text`
    /// block is a stand-in meka put where an attachment was, and glued to the words it would read
    /// as part of them. For the wires that take a user message as one string and for the Markdown
    /// export; assistant text keeps [`Self::text_content`]'s join, since its blocks are one answer
    /// split around tool calls.
    pub(crate) fn text_paragraphs(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Everything a provider renders as text, in order: the turn's context block, a blank line,
    /// then the words. What the two were joined with when they were one string, for the wires that
    /// take a user message as a single string.
    pub(crate) fn wire_text(&self) -> String {
        let context = self.content.iter().find_map(|block| match block {
            ContentBlock::TurnContext { text } => Some(text.as_str()),
            _ => None,
        });
        let words = self.text_paragraphs();
        match context {
            Some(context) if words.is_empty() => context.to_string(),
            Some(context) => format!("{context}\n\n{words}"),
            None => words,
        }
    }

    /// A copy of this message with every [`ContentBlock::ToolUse`] removed. Used when persisting a
    /// turn that was interrupted before its tools ran: keeping the `tool_use` blocks would orphan
    /// them (no matching `tool_result`) and the provider would reject the next request.
    pub(crate) fn without_tool_use(&self) -> Message {
        Message {
            role: self.role.clone(),
            content: self
                .content
                .iter()
                .filter(|block| !matches!(block, ContentBlock::ToolUse { .. }))
                .cloned()
                .collect(),
        }
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
    /// and is never handed out again, or a one-shot job's fire, whose row is retired with the turn.
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
/// marked in place.
pub(crate) fn format_session_as_markdown(
    session_id: uuid::Uuid,
    events: &[Event],
    tool_outputs: &std::collections::HashMap<String, String>,
) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    writeln!(output, "# Session {session_id}\n").ok();

    // Walk the raw event log so the full conversation is exported, including turns a compaction
    // later hid from the model. Each `CompactBoundary` becomes a marker; the turns it summarized
    // stay above it (the kept tail is re-appended after it, so the recent turns appear on both
    // sides of the marker, as stored).
    for event in events {
        match event {
            Event::Append(message) => {
                write_message_markdown(&mut output, message, tool_outputs);
            }
            Event::CompactBoundary { summary, .. } => {
                writeln!(output, "---\n").ok();
                writeln!(output, "<details>").ok();
                writeln!(
                    output,
                    "<summary>Session compaction (summary the model saw in place of the turns above)</summary>\n"
                )
                .ok();
                writeln!(output, "{}\n", summary.text_content()).ok();
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
            // repair (or a rewind, which is a repair with nothing to put back) is the one place
            // where what the model saw and what actually happened diverge.
            Event::Repair {
                replaced_count,
                messages,
            } => {
                writeln!(output, "---\n").ok();
                writeln!(output, "<details>").ok();
                writeln!(
                    output,
                    "<summary>{} message(s) above replaced with {} (rejected by the provider, or rewound)</summary>\n",
                    replaced_count,
                    if messages.is_empty() {
                        "nothing".to_string()
                    } else {
                        format!("{} message(s)", messages.len())
                    },
                )
                .ok();
                for message in messages {
                    write_message_markdown(&mut output, message, tool_outputs);
                }
                writeln!(output, "</details>\n").ok();
            }
        }
    }

    output
}
/// Append one message to a Markdown export, with tool calls and results in collapsible blocks.
pub(crate) fn write_message_markdown(
    output: &mut String,
    message: &crate::conversation::Message,
    tool_outputs: &std::collections::HashMap<String, String>,
) {
    use std::fmt::Write;

    match message.role {
        crate::conversation::Role::User => {
            // A `User` message is either a turn or a tool-results envelope; the blocks say which.
            let has_tool_results = message
                .content
                .iter()
                .any(|block| matches!(block, crate::conversation::ContentBlock::ToolResult { .. }));
            if has_tool_results {
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
                        let text =
                            crate::conversation::ContentBlock::tool_result_text_content(content);
                        let text = resolve_large_output_tags(&text, tool_outputs);
                        writeln!(output, "```\n{text}\n```\n").ok();
                        writeln!(output, "</details>\n").ok();
                    }
                }
            } else {
                writeln!(output, "## User\n").ok();
                let words = message.text_paragraphs();
                if words.is_empty() {
                    // `Message::user_turn` leaves the `Text` block out when there were no words,
                    // so a heading over nothing is what a turn fired for background outcomes
                    // would otherwise leave.
                    writeln!(
                        output,
                        "*meka opened this turn with no words from the user; its context block \
                         carried background outcomes or a resume notice.*\n"
                    )
                    .ok();
                } else {
                    writeln!(output, "{words}\n").ok();
                }
            }
        }
        crate::conversation::Role::Assistant => {
            writeln!(output, "## Assistant\n").ok();
            for block in &message.content {
                match block {
                    crate::conversation::ContentBlock::Text { text } => {
                        writeln!(output, "{text}\n").ok();
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
                    | crate::conversation::ContentBlock::Thinking { .. }
                    | crate::conversation::ContentBlock::RedactedThinking { .. }
                    | crate::conversation::ContentBlock::Image { .. } => {}
                }
            }
        }
    }
}
/// Replace each `<large-output>` tag with the scratchpad entry it names, where one exists.
pub(crate) fn resolve_large_output_tags(
    text: &str,
    tool_outputs: &std::collections::HashMap<String, String>,
) -> String {
    let re = match regex::Regex::new(r#"<large-output name="([^"]+)"[^>]*>[\s\S]*?</large-output>"#)
    {
        Ok(re) => re,
        Err(_) => return text.to_string(),
    };

    re.replace_all(text, |caps: &regex::Captures| {
        let name = &caps[1];
        match tool_outputs.get(name) {
            Some(content) => content.clone(),
            None => caps[0].to_string(),
        }
    })
    .into_owned()
}
/// Stands in for a block whose text the server withheld, under Claude's `redact-thinking` beta or
/// display updates.
///
/// meka's own words rather than the model's, but rendered down the same path so there is one way a
/// thinking block reaches the terminal. It survives CommonMark unchanged: a bracketed run is a
/// shortcut reference link only when a matching definition exists, and none does.
pub(crate) const REDACTED_THINKING: &str = "[redacted thinking]";

#[cfg(test)]
mod tests {
    use super::*;

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
                loaded_tools_snapshot: HashSet::new(),
            },
            Event::Append(Message::user_turn(
                "<context/>",
                "and python ones",
                Vec::new(),
            )),
        ]);
        assert_eq!(conversation.title(), "find rust files");
    }

    /// What meka put in place of an attachment is a `Text` block like the words, and is not a
    /// title: a first turn that carried only an image has none, whether the image was redacted by
    /// the budget or replaced by a harness note, until a user says something.
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
                ContentBlock::Text {
                    text: IMAGE_REDACTION_PLACEHOLDER.to_string(),
                },
            ],
        });
        conversation.append(Message::user(format!(
            "{HARNESS_NOTE} An image here was removed."
        )));
        assert_eq!(
            conversation.title(),
            "",
            "nor is a placeholder or a harness note"
        );
        conversation.append(Message::user("what is in the picture?"));
        assert_eq!(conversation.title(), "what is in the picture?");
    }

    /// A placeholder recorded beside the user's words is its own paragraph on a single-string wire
    /// and in the export, not a suffix of what they typed.
    #[test]
    fn a_users_words_and_a_placeholder_are_separate_paragraphs() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::TurnContext {
                    text: "<context/>".to_string(),
                },
                ContentBlock::Text {
                    text: "look at this".to_string(),
                },
                ContentBlock::Text {
                    text: IMAGE_REDACTION_PLACEHOLDER.to_string(),
                },
            ],
        };
        assert_eq!(
            message.wire_text(),
            format!("<context/>\n\nlook at this\n\n{IMAGE_REDACTION_PLACEHOLDER}")
        );
        let mut output = String::new();
        write_message_markdown(&mut output, &message, &std::collections::HashMap::new());
        assert!(
            output.contains(&format!("look at this\n\n{IMAGE_REDACTION_PLACEHOLDER}\n")),
            "{output}"
        );
        assert_eq!(
            Message::assistant_text("one").text_content(),
            "one",
            "assistant text keeps its join"
        );
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
            output.contains("## User\n\n*meka opened this turn with no words"),
            "{output}"
        );
    }

    fn assistant_with_tool_use(use_id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: use_id.to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "/tmp/x"}),
            }],
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
            }],
        }
    }

    fn load_tool_use(id: &str, target: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_string(),
                name: crate::tools::LOAD_TOOL_NAME.to_string(),
                input: serde_json::json!({"name": target}),
            }],
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
            }],
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
            HashSet::new(),
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
            HashSet::new(),
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
        log.replace_for_compaction(summary, tail, HashSet::new());

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
        log.replace_for_compaction(Message::user("[summary]"), Vec::new(), HashSet::new());
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
            HashSet::new(),
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
        log.replace_for_compaction(Message::user("summary one"), Vec::new(), HashSet::new());
        log.append(Message::user("after the first summary"));
        let before: Vec<String> = log.iter().map(|m| format!("{m:?}")).collect();

        log.replace_for_compaction(
            Message::user("summary two"),
            vec![Message::user("kept")],
            HashSet::new(),
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
                },
                image_block(base64_of(image::ImageFormat::Jpeg), "image/png"),
            ],
        });

        assert_eq!(log.invalid_images_replaced(), 1);
        let content = &log.as_slice()[0].content;
        assert!(
            matches!(content[0], ContentBlock::Text { .. }),
            "text is kept"
        );
        match &content[1] {
            ContentBlock::Text { text } => assert!(text.contains("image/png"), "{text}"),
            other => panic!("expected the image to become text, got {other:?}"),
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
            }],
        });

        assert_eq!(log.invalid_images_replaced(), 1);
        match &log.as_slice()[0].content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "call_1");
                assert!(is_error);
                assert!(matches!(
                    content[0],
                    crate::conversation::ToolResultContent::Text { .. }
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
            orphan_event_indices(log.events()).is_empty(),
            "the cut must not separate a tool_use from its tool_result"
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

    /// A redaction names one image, tail-relative, and the replay puts the placeholder exactly
    /// there: an input image by block, a tool result's image by item, and nothing else.
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
                    ToolResultContent::Text {
                        text: "[Image: x.png]".to_string(),
                    },
                    ToolResultContent::Image {
                        source: image.clone(),
                    },
                ],
                is_error: false,
            }],
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
            assert!(matches!(&view[0].content[0], ContentBlock::Text { text } if text == "look"));
            assert!(
                matches!(&view[0].content[1], ContentBlock::Text { text } if text == IMAGE_REDACTION_PLACEHOLDER),
                "{:?}",
                view[0].content
            );
            match &view[1].content[0] {
                ContentBlock::ToolResult { content, .. } => {
                    assert!(
                        matches!(&content[0], ToolResultContent::Text { text } if text == "[Image: x.png]")
                    );
                    assert!(
                        matches!(&content[1], ToolResultContent::Text { text } if text == IMAGE_REDACTION_PLACEHOLDER)
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
            HashSet::new(),
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
                },
                ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "execute_command".to_string(),
                    input: serde_json::json!({"command": "git diff"}),
                },
                ContentBlock::ToolUse {
                    id: "u2".to_string(),
                    name: "scratchpad_read".to_string(),
                    input: serde_json::json!({"name": "tool_1_output"}),
                },
            ],
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
            HashSet::new(),
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
            HashSet::new(),
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
        let loaded = crate::tools::load_tool::extract_loaded_tool_names_from_events(log.events());
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
            crate::tools::load_tool::extract_loaded_tool_names_from_events(log.events()),
            vec!["omega".to_string(), "alpha".to_string()]
        );
    }

    #[test]
    fn extract_loaded_tool_names_recovers_snapshot_across_boundary() {
        // Pre-boundary: load_tool(scratchpad_read) succeeds. After the boundary swallows it, the
        // snapshot must restore scratchpad_read in the active set.
        let mut log = Conversation::new();
        log.append(load_tool_use("u1", "scratchpad_read"));
        log.append(load_tool_result("u1", false));

        let snapshot: HashSet<String> = ["scratchpad_read".to_string()].into_iter().collect();
        log.replace_for_compaction(Message::user("[summary]"), Vec::new(), snapshot);

        let loaded = crate::tools::load_tool::extract_loaded_tool_names_from_events(log.events());
        assert!(loaded.iter().any(|name| name == "scratchpad_read"));
    }

    #[test]
    fn prune_compacted_events_drops_pre_boundary_log() {
        let mut log = Conversation::new();
        log.append(load_tool_use("u1", "scratchpad_read"));
        log.append(load_tool_result("u1", false));
        log.append(Message::user("m1"));

        let snapshot: HashSet<String> = ["scratchpad_read".to_string()].into_iter().collect();
        log.replace_for_compaction(
            Message::user("[summary-1]"),
            vec![Message::user("tail-1")],
            snapshot.clone(),
        );
        log.append(Message::assistant_text("m2"));
        log.replace_for_compaction(
            Message::user("[summary-2]"),
            vec![Message::user("tail-2")],
            snapshot,
        );

        let view_before: Vec<String> = log.as_slice().iter().map(|m| m.text_content()).collect();
        let loaded_before =
            crate::tools::load_tool::extract_loaded_tool_names_from_events(log.events());

        log.prune_compacted_events();

        // Materialized view and recovered tool set are unchanged.
        let view_after: Vec<String> = log.as_slice().iter().map(|m| m.text_content()).collect();
        assert_eq!(view_before, view_after);
        assert_eq!(
            loaded_before,
            crate::tools::load_tool::extract_loaded_tool_names_from_events(log.events())
        );
        assert!(
            crate::tools::load_tool::extract_loaded_tool_names_from_events(log.events())
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

    #[test]
    fn extract_loaded_tool_names_pending_use_wiped_at_boundary() {
        // load_tool tool_use lives on one side of the boundary, its tool_result on the other; both
        // vanish from the materialized view, so the scanner must NOT count the pending pair across
        // the boundary.
        let mut log = Conversation::new();
        log.append(load_tool_use("u1", "scratchpad_read"));
        // No tool_result yet.
        log.replace_for_compaction(
            Message::user("[summary]"),
            vec![load_tool_result("u1", false)],
            HashSet::new(),
        );

        let loaded = crate::tools::load_tool::extract_loaded_tool_names_from_events(log.events());
        assert!(!loaded.iter().any(|name| name == "scratchpad_read"));
    }

    #[test]
    fn pop_unsaved_only_removes_trailing_append() {
        // After a CompactBoundary, the next legal call is `append`. A failed-save rollback after
        // that should remove the failed append, not the boundary.
        let mut log = Conversation::new();
        log.append(Message::user("pre"));
        log.replace_for_compaction(Message::user("[summary]"), Vec::new(), HashSet::new());
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

        let snapshot: HashSet<String> = ["mcp__notion__fetch".to_string()].into_iter().collect();
        let boundary = Event::CompactBoundary {
            summary: Message::user("[summary]"),
            replaced_count: 5,
            loaded_tools_snapshot: snapshot,
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
        log.replace_for_compaction(Message::user("[summary]"), Vec::new(), HashSet::new());
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
}
