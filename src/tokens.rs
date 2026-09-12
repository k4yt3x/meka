//! Cheap, dependency-free token estimation for the context gauge's fallback paths: the
//! post-compaction interim, the pre-send projection, and a resume of a row that recorded no
//! measurement, where there is no fresh provider `usage` to read. The authoritative figure is
//! always the provider's reported usage (see `Agent::record_context_tokens`); this only fills the
//! gap until the next real response corrects it. [`bound_text`] is the other kind of number this
//! module gives: not an estimate but a bound, for the one caller that acts before any correction.
//!
//! We estimate on UTF-8 **byte** length, not `char` count. BPE tokenizers merge roughly four bytes
//! per token for ASCII, and byte length tracks token count far better than code-point count for
//! multibyte scripts: a CJK character is one `char` but ~3 UTF-8 bytes and ~1 token, so
//! `chars().count() / 4` would undercount CJK by ~3x while `len() / 4` stays in the right ballpark.
//! This mirrors Codex's "4 bytes per token" heuristic. It is deliberately approximate; precision is
//! not needed for a transient interim value.

use crate::conversation::{ContentBlock, Message, ToolResultContent};

/// UTF-8 bytes of text per estimated token.
const BYTES_PER_TOKEN: u64 = 4;

/// Flat estimate for an image block (images don't tokenize by text length). Mirrors Claude Code's
/// ~2000 and pi's 1200; a middle value is plenty for a fallback.
const IMAGE_TOKENS: u64 = 1500;

/// Small per-message overhead for the role/turn framing the wire format wraps around each message.
const MESSAGE_OVERHEAD_TOKENS: u64 = 4;

/// Estimate the tokens a string contributes, from its UTF-8 byte length. Rounds up so non-empty
/// text never estimates to zero.
pub(crate) fn estimate_text(text: &str) -> u64 {
    (text.len() as u64).div_ceil(BYTES_PER_TOKEN)
}

/// A *bound* on the tokens a string will cost, for text about to be put into context by a tool
/// that sizes its own result. The estimate above is corrected by the next measurement; a bound is
/// acted on before any measurement can correct it, and its wrong answers are not symmetric: too
/// low sends a request the window cannot hold, too high cuts a read the model can continue. So it
/// counts the way BPE tokenizers split at their finest, not on average: a run of letters merges
/// about five to a token, but every digit, symbol and non-ASCII character is counted as one, which
/// is where four bytes per token under-reads by two to three times (measured: 736 KB of digits
/// reported as 460k tokens). Whitespace merges into its neighbor, except that a run of it, the
/// indentation of code, costs one.
#[derive(Default)]
struct TokenBound {
    total: u64,
    letters: u32,
    whitespace: u32,
}

impl TokenBound {
    /// Letters per token in a run, the rate ordinary English words merge at.
    const LETTERS_PER_TOKEN: u32 = 5;

    fn push(&mut self, character: char) {
        if character.is_ascii_alphabetic() {
            self.whitespace = 0;
            self.letters += 1;
            if self.letters % Self::LETTERS_PER_TOKEN == 1 {
                self.total += 1;
            }
            return;
        }
        self.letters = 0;
        if character.is_ascii_whitespace() {
            self.whitespace += 1;
            if self.whitespace == 2 {
                self.total += 1;
            }
            return;
        }
        self.whitespace = 0;
        self.total += 1;
    }
}

/// The most tokens `text` is expected to cost; see [`TokenBound`].
pub(crate) fn bound_text(text: &str) -> u64 {
    let mut bound = TokenBound::default();
    for character in text.chars() {
        bound.push(character);
    }
    bound.total
}

/// The longest prefix of `text`, in bytes and on a character boundary, whose [`bound_text`] is
/// within `tokens`.
pub(crate) fn prefix_within(text: &str, tokens: u64) -> usize {
    let mut bound = TokenBound::default();
    for (index, character) in text.char_indices() {
        bound.push(character);
        if bound.total > tokens {
            return index;
        }
    }
    text.len()
}

/// Estimate the tokens one message contributes to the context.
pub(crate) fn estimate_message(message: &Message) -> u64 {
    let mut total = MESSAGE_OVERHEAD_TOKENS;
    for block in &message.content {
        let block_tokens = match block {
            ContentBlock::Text { text } => estimate_text(text),
            // Sent as text ahead of the words, so it costs what text costs.
            ContentBlock::TurnContext { text } => estimate_text(text),
            ContentBlock::Image { .. } => IMAGE_TOKENS,
            // The readable half only, which is the whole cost under `Signed` and an under-read
            // under `Sealed`: there the text is a summary and the reasoning is the sealed blob
            // beside it, replayed on every later request and billed as the reasoning tokens it
            // decrypts to, a figure its base64 length does not give (measured: 2572 bytes
            // decrypting to ~491 tokens, where this estimator's bytes-per-token would say 643).
            // Counting it would be guesswork about the provider's side, so the gap is deliberate,
            // in the same direction as the tool schemas this already omits. Three of the four
            // callers correct against the provider's reported usage on the next response;
            // `compute_compaction_split` does not, so a reasoning-heavy tail is kept larger than
            // its budget rather than smaller: the safe direction, but not a free one.
            ContentBlock::Thinking { thinking, .. } => estimate_text(thinking),
            // Same shape, same reasoning: opaque encrypted reasoning whose on-wire token cost isn't
            // derivable from `data` length.
            ContentBlock::RedactedThinking { .. } => 0,
            // Tool-call args are serialized JSON on the wire; count the name plus the compact JSON.
            ContentBlock::ToolUse { name, input, .. } => {
                estimate_text(name).saturating_add(estimate_text(&input.to_string()))
            }
            ContentBlock::ToolResult { content, .. } => content
                .iter()
                .map(|item| match item {
                    ToolResultContent::Text { text } => estimate_text(text),
                    ToolResultContent::Image { .. } => IMAGE_TOKENS,
                })
                .fold(0u64, u64::saturating_add),
        };
        total = total.saturating_add(block_tokens);
    }
    total
}

/// Estimate the tokens a set of messages contributes. Used to seed the context gauge when there is
/// no fresh provider reading (post-compaction, or on resume before the first turn). It omits the
/// fixed system-prompt + tool-schema overhead, so it under-reads the true input until the next real
/// response corrects it; acceptable for the transient interim value.
pub(crate) fn estimate_messages(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(estimate_message)
        .fold(0u64, u64::saturating_add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        conversation::{ContentBlock, Message, Role, ToolResultContent},
        image::ImageSource,
    };

    #[test]
    fn estimate_text_rounds_up_on_bytes_not_chars() {
        assert_eq!(estimate_text(""), 0);
        // 8 ASCII bytes -> 2 tokens.
        assert_eq!(estimate_text("abcdefgh"), 2);
        // Rounds up: 5 bytes -> 2.
        assert_eq!(estimate_text("hello"), 2);
    }

    /// Each class at the rate the bound gives it, on text whose true cost is known to sit under
    /// that rate: prose merges, digits and symbols and CJK do not, single spaces ride along.
    #[test]
    fn the_bound_counts_letters_in_runs_and_everything_else_singly() {
        assert_eq!(bound_text(""), 0);
        assert_eq!(bound_text("the"), 1);
        assert_eq!(bound_text("conversation"), 3);
        assert_eq!(bound_text("12345"), 5);
        assert_eq!(bound_text("a, b"), 3);
        assert_eq!(bound_text("    x"), 2);
        assert_eq!(bound_text("a b\nc"), 3);
        assert_eq!(bound_text("字字"), 2);
        // Nine digits, ten words of which `number` spans two tokens, single spaces free.
        assert_eq!(
            bound_text("01573 the quick brown fox jumps over the lazy dog number 1573\n"),
            20
        );
    }

    /// The prefix stops at the character that would break the budget and never inside one.
    #[test]
    fn the_prefix_within_a_budget_ends_on_a_character_boundary() {
        assert_eq!(prefix_within("abcdefghij", 1), 5);
        assert_eq!(prefix_within("abcdefghij", 2), 10);
        assert_eq!(prefix_within("12345", 10), 5);
        assert_eq!(prefix_within("12345", 3), 3);
        assert_eq!(prefix_within("12345", 0), 0);
        assert_eq!(prefix_within("é1", 1), 2);
        assert_eq!(prefix_within("", 5), 0);
    }

    #[test]
    fn estimate_text_counts_cjk_by_bytes_not_codepoints() {
        // 10 CJK chars = 30 UTF-8 bytes. Byte-based: 30/4 -> 8 tokens (right ballpark, ~1
        // tok/char). A char-count/4 estimate would give 10/4 = 3, undercounting ~3x, which is the
        // bug this avoids.
        let cjk = "字".repeat(10);
        assert_eq!(cjk.chars().count(), 10);
        assert_eq!(cjk.len(), 30);
        assert_eq!(estimate_text(&cjk), 8);
        assert!(estimate_text(&cjk) > (cjk.chars().count() as u64) / 4);
    }

    #[test]
    fn estimate_text_handles_multibyte_emoji_without_panicking() {
        // A ZWJ family emoji is several code points / many bytes; just assert it's charged
        // proportional to its byte length and doesn't panic on a char boundary.
        let emoji = "👨‍👩‍👧‍👦";
        assert_eq!(estimate_text(emoji), (emoji.len() as u64).div_ceil(4));
        assert!(estimate_text(emoji) > 0);
    }

    #[test]
    fn estimate_message_sums_blocks_with_overhead() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "abcdefgh".to_string(), // 2
                },
                ContentBlock::ToolUse {
                    id: "1".to_string(),
                    name: "read".to_string(), // 1 (4 bytes -> 1)
                    input: serde_json::json!({"path": "a"}), // compact JSON bytes/4
                },
            ],
        };
        let json_tokens = estimate_text(&serde_json::json!({"path": "a"}).to_string());
        let expected = MESSAGE_OVERHEAD_TOKENS + 2 + 1 + json_tokens;
        assert_eq!(estimate_message(&message), expected);
    }

    #[test]
    fn estimate_message_charges_flat_for_input_images() {
        let message = Message::user_with_images("look", vec![ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "x".repeat(100_000), // payload length must NOT inflate the estimate
        }]);
        // overhead + text("look" = 4 bytes -> 1) + flat image cost.
        assert_eq!(
            estimate_message(&message),
            MESSAGE_OVERHEAD_TOKENS + 1 + IMAGE_TOKENS
        );
    }

    #[test]
    fn estimate_message_charges_flat_for_images() {
        let message = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "1".to_string(),
                content: vec![ToolResultContent::Image {
                    source: ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "x".repeat(100_000), // huge payload must NOT be counted by length
                    },
                }],
                is_error: false,
            }],
        };
        assert_eq!(
            estimate_message(&message),
            MESSAGE_OVERHEAD_TOKENS + IMAGE_TOKENS
        );
    }

    #[test]
    fn estimate_messages_sums_and_saturates() {
        let messages = vec![
            Message::user("abcd"),               // overhead + 1
            Message::assistant_text("abcdefgh"), // overhead + 2
        ];
        assert_eq!(
            estimate_messages(&messages),
            (MESSAGE_OVERHEAD_TOKENS + 1) + (MESSAGE_OVERHEAD_TOKENS + 2)
        );
    }
}
