//! The OpenAI backends.
//!
//! Three live here, across two protocols. OpenAI serves both, which is why the backend names say
//! which one rather than naming the vendor twice:
//!
//! - [`chat_completions`]: Chat Completions, `POST {base}/chat/completions`, against
//!   `api.openai.com/v1` or any endpoint implementing that format (Ollama, vLLM, OpenRouter,
//!   Synthetic, …). Bearer-token auth. Note this is *not* the legacy `/v1/completions`, a different
//!   protocol several of those same servers also expose.
//! - [`responses`]: the Responses API, `POST {base}/responses`, likewise against OpenAI or any
//!   endpoint serving it (Ollama v0.13.3+, vLLM, LM Studio, OpenRouter). Bearer-token auth.
//! - [`subscription`]: the Responses API against `chatgpt.com/backend-api/codex`, authenticated by
//!   ChatGPT subscription OAuth (Plus / Pro / Team / Business / Enterprise) and shaped like
//!   OpenAI's own Codex CLI.
//!
//! The two Responses backends share the wire format through [`responses_wire`]; Chat Completions
//! is a different protocol and shares nothing but [`data_url`].

pub(crate) mod chat_completions;
pub(crate) mod responses;
pub(crate) mod responses_wire;
pub(crate) mod subscription;

pub(crate) use chat_completions::OpenAiChatCompletionsProvider;
pub(crate) use responses::OpenAiResponsesProvider;
pub(crate) use subscription::ChatGptSubscriptionProvider;

/// A `usage` object of either OpenAI wire as the agent counts it.
///
/// Both wires fold the cached tokens into the total (`input_tokens` on Responses, `prompt_tokens`
/// on Chat Completions) and name the cached subset under a `*_details` object, while the agent
/// counts each tier once, so the cached count comes out of the total here. The endpoint fills its
/// cache on its own and bills nothing for it, so the cache-write tier stays at zero.
pub(super) fn parse_usage(
    usage: &serde_json::Value,
    input_key: &str,
    input_details_key: &str,
    output_key: &str,
) -> crate::stats::TokenUsage {
    let field = |key: &str| usage.get(key).and_then(|value| value.as_u64()).unwrap_or(0);
    let input = field(input_key);
    let cached = usage
        .get(input_details_key)
        .and_then(|details| details.get("cached_tokens"))
        .and_then(|value| value.as_u64())
        .unwrap_or(0)
        .min(input);
    crate::stats::TokenUsage {
        input_tokens: input - cached,
        output_tokens: field(output_key),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cached,
    }
}

/// A `data:` URL for an image, the one piece of image wire-format both protocols share (Chat
/// Completions `image_url.url` and the Responses API `input_image.image_url`). `None` for a
/// reference whose bytes were never loaded, which the caller sends as a sentence instead.
fn data_url(source: &crate::image::ImageSource) -> Option<String> {
    Some(format!(
        "data:{};base64,{}",
        source.media_type(),
        source.base64_data()?
    ))
}

#[cfg(test)]
mod tests {
    use super::parse_usage;

    /// A Responses `usage` counts its cached tokens inside `input_tokens`; the agent counts them
    /// once, as reads, so `/status` reports a hit rate instead of a flat zero.
    #[test]
    fn cached_tokens_come_out_of_the_responses_input_total() {
        let usage = parse_usage(
            &serde_json::json!({
                "input_tokens": 23465,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 20864},
                "output_tokens": 287,
                "output_tokens_details": {"reasoning_tokens": 192},
            }),
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
        );
        assert_eq!(usage.input_tokens, 23465 - 20864);
        assert_eq!(usage.cache_read_input_tokens, 20864);
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.output_tokens, 287);
    }

    /// Chat Completions names the same three things differently and is read the same way.
    #[test]
    fn cached_tokens_come_out_of_the_chat_completions_prompt_total() {
        let usage = parse_usage(
            &serde_json::json!({
                "prompt_tokens": 1200,
                "prompt_tokens_details": {"cached_tokens": 1024},
                "completion_tokens": 40,
            }),
            "prompt_tokens",
            "prompt_tokens_details",
            "completion_tokens",
        );
        assert_eq!(usage.input_tokens, 176);
        assert_eq!(usage.cache_read_input_tokens, 1024);
        assert_eq!(usage.output_tokens, 40);
    }

    /// An endpoint that reports no details, or a cached count past the total, leaves the input
    /// whole rather than negative.
    #[test]
    fn a_usage_without_details_is_all_live_input() {
        let plain = parse_usage(
            &serde_json::json!({"input_tokens": 42, "output_tokens": 7}),
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
        );
        assert_eq!(plain.input_tokens, 42);
        assert_eq!(plain.cache_read_input_tokens, 0);

        let overstated = parse_usage(
            &serde_json::json!({"input_tokens": 10, "input_tokens_details": {"cached_tokens": 50}}),
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
        );
        assert_eq!(overstated.input_tokens, 0);
        assert_eq!(overstated.cache_read_input_tokens, 10);
    }
}
