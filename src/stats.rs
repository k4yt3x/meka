//! Per-session counters surfaced by `/status`, recorded by the agent: tokens and turn count from
//! the provider's usage, image redactions from the [`Redaction`] a Claude provider reports.
//!
//! All fields are lock-free atomics so any task can update without contention; readers take a
//! [`SessionStatsSnapshot`] for display.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// What one image-redaction pass removed from a request body.
///
/// Reported by the provider that did it, on the notice it sends, and counted by the agent: the
/// provider is cached per profile and serves every session on it, so it cannot know whose
/// statistics the pass belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Redaction {
    pub(crate) images: u64,
    pub(crate) bytes: u64,
    /// Where in the request's messages the images sat, for the agent to record on the
    /// conversation so the next request does not redact them again.
    pub(crate) positions: Vec<crate::image::RedactedImage>,
}

#[derive(Debug, Default)]
pub(crate) struct SessionStats {
    turns: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    cache_creation_input_tokens: AtomicU64,
    cache_read_input_tokens: AtomicU64,
    redactions: AtomicU64,
    redacted_images: AtomicU64,
    redacted_bytes: AtomicU64,
}

impl SessionStats {
    /// Rebuild the counters from a persisted snapshot, so a resumed session continues its lifetime
    /// totals instead of restarting at zero.
    pub(crate) fn from_snapshot(snapshot: &SessionStatsSnapshot) -> Self {
        Self {
            turns: AtomicU64::new(snapshot.turns),
            input_tokens: AtomicU64::new(snapshot.input_tokens),
            output_tokens: AtomicU64::new(snapshot.output_tokens),
            cache_creation_input_tokens: AtomicU64::new(snapshot.cache_creation_input_tokens),
            cache_read_input_tokens: AtomicU64::new(snapshot.cache_read_input_tokens),
            redactions: AtomicU64::new(snapshot.redactions),
            redacted_images: AtomicU64::new(snapshot.redacted_images),
            redacted_bytes: AtomicU64::new(snapshot.redacted_bytes),
        }
    }

    /// Roll a successful turn's usage into the running totals.
    pub(crate) fn record_turn(&self, usage: &TokenUsage) {
        self.turns.fetch_add(1, Relaxed);
        self.input_tokens.fetch_add(usage.input_tokens, Relaxed);
        self.output_tokens.fetch_add(usage.output_tokens, Relaxed);
        self.cache_creation_input_tokens
            .fetch_add(usage.cache_creation_input_tokens, Relaxed);
        self.cache_read_input_tokens
            .fetch_add(usage.cache_read_input_tokens, Relaxed);
    }

    /// Record tokens spent outside any turn: the compaction calls, both the checkpoint turn and
    /// the standalone summarizer.
    ///
    /// Separate from [`Self::record_turn`] only because that also increments the turn counter, and
    /// a compaction is not a turn the user asked for. The tokens are real spend and belong in the
    /// totals regardless: compaction is the most expensive thing meka does without being asked,
    /// and leaving it out made `/status` disagree with the provider's bill by exactly the amount
    /// the user would most want explained.
    pub(crate) fn record_untracked_tokens(&self, usage: &TokenUsage) {
        self.input_tokens.fetch_add(usage.input_tokens, Relaxed);
        self.output_tokens.fetch_add(usage.output_tokens, Relaxed);
        self.cache_creation_input_tokens
            .fetch_add(usage.cache_creation_input_tokens, Relaxed);
        self.cache_read_input_tokens
            .fetch_add(usage.cache_read_input_tokens, Relaxed);
    }

    /// Record one body-redaction pass a provider reported.
    pub(crate) fn record_redaction(&self, redaction: &Redaction) {
        self.redactions.fetch_add(1, Relaxed);
        self.redacted_images.fetch_add(redaction.images, Relaxed);
        self.redacted_bytes.fetch_add(redaction.bytes, Relaxed);
    }

    pub(crate) fn snapshot(&self) -> SessionStatsSnapshot {
        SessionStatsSnapshot {
            turns: self.turns.load(Relaxed),
            input_tokens: self.input_tokens.load(Relaxed),
            output_tokens: self.output_tokens.load(Relaxed),
            cache_creation_input_tokens: self.cache_creation_input_tokens.load(Relaxed),
            cache_read_input_tokens: self.cache_read_input_tokens.load(Relaxed),
            redactions: self.redactions.load(Relaxed),
            redacted_images: self.redacted_images.load(Relaxed),
            redacted_bytes: self.redacted_bytes.load(Relaxed),
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct SessionStatsSnapshot {
    pub(crate) turns: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cache_creation_input_tokens: u64,
    pub(crate) cache_read_input_tokens: u64,
    pub(crate) redactions: u64,
    pub(crate) redacted_images: u64,
    pub(crate) redacted_bytes: u64,
}

impl SessionStatsSnapshot {
    /// Sum of all three input-token tiers (live, cache-write, cache-read). Matches what Anthropic
    /// bills against "input".
    pub(crate) fn total_input_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.cache_creation_input_tokens)
            .saturating_add(self.cache_read_input_tokens)
    }

    /// Cache-hit ratio as an integer percent (0–100). Returns 0 when no input tokens have been
    /// recorded yet.
    pub(crate) fn cache_hit_pct(&self) -> u64 {
        let total = self.total_input_tokens();
        if total == 0 {
            0
        } else {
            ((self.cache_read_input_tokens as f64) / (total as f64) * 100.0).round() as u64
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub(crate) struct TokenUsage {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    /// Tokens billed at the cache-write tier (content newly cached this turn). Anthropic-only;
    /// OpenAI providers leave this at 0.
    pub(crate) cache_creation_input_tokens: u64,
    /// Tokens served from the prompt cache (cache-read tier). Anthropic returns this in
    /// `usage.cache_read_input_tokens`; OpenAI providers leave it at 0 today.
    pub(crate) cache_read_input_tokens: u64,
}
impl TokenUsage {
    /// Fold a streamed usage update into the running per-round total, taking each field from
    /// `update` only when it is non-zero. Providers split usage across events: Anthropic reports
    /// the input/cache tiers on `message_start` and the final `output_tokens` on
    /// `message_delta` (the other fields absent, i.e. parsed as 0), while OpenAI/Codex send a
    /// single usage event. The non-zero rule keeps the `message_start` input/cache values
    /// instead of letting a later event that omits them clobber the count back to 0.
    pub(crate) fn merge_stream(&mut self, update: &TokenUsage) {
        if update.input_tokens > 0 {
            self.input_tokens = update.input_tokens;
        }
        if update.output_tokens > 0 {
            self.output_tokens = update.output_tokens;
        }
        if update.cache_creation_input_tokens > 0 {
            self.cache_creation_input_tokens = update.cache_creation_input_tokens;
        }
        if update.cache_read_input_tokens > 0 {
            self.cache_read_input_tokens = update.cache_read_input_tokens;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Compaction spend has to reach the totals without pretending to be a turn, so `/status`
    /// agrees with the provider's bill while still reporting how many turns the user actually ran.
    #[test]
    fn untracked_tokens_add_spend_without_adding_a_turn() {
        let stats = SessionStats::default();
        stats.record_turn(&TokenUsage {
            input_tokens: 100,
            output_tokens: 10,
            ..Default::default()
        });
        stats.record_untracked_tokens(&TokenUsage {
            input_tokens: 900,
            output_tokens: 90,
            ..Default::default()
        });

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.turns, 1, "the compaction must not count as a turn");
        assert_eq!(snapshot.input_tokens, 1_000);
        assert_eq!(snapshot.output_tokens, 100);
    }

    #[test]
    fn recorded_turns_accumulate_token_usage() {
        let stats = SessionStats::default();
        stats.record_turn(&TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 20,
            cache_read_input_tokens: 30,
        });
        stats.record_turn(&TokenUsage {
            input_tokens: 5,
            output_tokens: 7,
            cache_creation_input_tokens: 1,
            cache_read_input_tokens: 2,
        });
        let snap = stats.snapshot();
        assert_eq!(snap.turns, 2);
        assert_eq!(snap.input_tokens, 105);
        assert_eq!(snap.output_tokens, 57);
        assert_eq!(snap.cache_creation_input_tokens, 21);
        assert_eq!(snap.cache_read_input_tokens, 32);
        assert_eq!(snap.total_input_tokens(), 105 + 21 + 32);
    }

    #[test]
    fn recorded_redactions_accumulate_images_and_bytes() {
        let stats = SessionStats::default();
        stats.record_redaction(&Redaction {
            images: 2,
            bytes: 4_000_000,
            positions: Vec::new(),
        });
        stats.record_redaction(&Redaction {
            images: 1,
            bytes: 2_000_000,
            positions: Vec::new(),
        });
        let snap = stats.snapshot();
        assert_eq!(snap.redactions, 2);
        assert_eq!(snap.redacted_images, 3);
        assert_eq!(snap.redacted_bytes, 6_000_000);
    }

    #[test]
    fn from_snapshot_seeds_all_fields() {
        // Resume rebuilds the live counters from the persisted snapshot.
        let snapshot = SessionStatsSnapshot {
            turns: 3,
            input_tokens: 10,
            output_tokens: 20,
            cache_creation_input_tokens: 30,
            cache_read_input_tokens: 40,
            redactions: 1,
            redacted_images: 2,
            redacted_bytes: 1024,
        };
        let round = SessionStats::from_snapshot(&snapshot).snapshot();
        assert_eq!(round.turns, 3);
        assert_eq!(round.input_tokens, 10);
        assert_eq!(round.output_tokens, 20);
        assert_eq!(round.cache_creation_input_tokens, 30);
        assert_eq!(round.cache_read_input_tokens, 40);
        assert_eq!(round.redactions, 1);
        assert_eq!(round.redacted_images, 2);
        assert_eq!(round.redacted_bytes, 1024);
        // A turn recorded after seeding accumulates on top of the seeded totals.
        let seeded = SessionStats::from_snapshot(&snapshot);
        seeded.record_turn(&TokenUsage {
            input_tokens: 5,
            output_tokens: 5,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        });
        assert_eq!(seeded.snapshot().turns, 4);
        assert_eq!(seeded.snapshot().input_tokens, 15);
    }

    #[test]
    fn cache_hit_pct_zero_when_no_input() {
        let snap = SessionStats::default().snapshot();
        assert_eq!(snap.cache_hit_pct(), 0);
    }

    #[test]
    fn cache_hit_pct_rounds() {
        let stats = SessionStats::default();
        stats.record_turn(&TokenUsage {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: 5,
            cache_read_input_tokens: 85,
        });
        // total = 100, hit = 85
        assert_eq!(stats.snapshot().cache_hit_pct(), 85);
    }
}
