//! Backoff policy for [`crate::error::MekaError::RetryableProvider`]. Pure, provider-agnostic: no
//! knowledge of streaming vs. non-streaming, no I/O. Callers (`agent.rs`) own the retry loop and
//! the decision of *whether* a given failure is safe to retry (e.g. no user-visible content shown
//! yet); this module only answers *how long to wait* before the next attempt.

use std::time::Duration;

/// Maximum number of retries after the initial attempt (so `MAX_PROVIDER_RETRIES + 1` total
/// attempts). Hardcoded, not config-exposed — matches the project's convention for turn-level retry
/// knobs (see `MAX_OVERFLOW_RETRIES` in `agent.rs`).
pub(crate) const MAX_PROVIDER_RETRIES: u32 = 3;

/// Delay cap for the computed exponential backoff (no `Retry-After` header present).
const BACKOFF_CAP: Duration = Duration::from_secs(8);

/// Delay cap for a provider-supplied `Retry-After` value. Protects an interactive turn from a
/// broken or unexpectedly large header value blocking the CLI for an excessive amount of time;
/// Ctrl-C still works during the wait regardless (the caller sleeps via `tokio::select!` against
/// the turn's cancellation token).
const RETRY_AFTER_CAP: Duration = Duration::from_secs(15);

/// How long to wait before retry attempt number `attempt` (1-indexed: the first retry is
/// `attempt == 1`). Honors the provider's `Retry-After` hint when present (capped); otherwise
/// exponential backoff `1s, 2s, 4s, ...` capped at [`BACKOFF_CAP`], mirroring the shape of the
/// existing MCP reconnect backoff (`src/mcp.rs`) but tuned tighter for interactive turn latency.
pub(crate) fn backoff_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    match retry_after {
        Some(delay) => delay.min(RETRY_AFTER_CAP),
        None => {
            let computed = Duration::from_secs(1u64 << attempt.saturating_sub(1));
            computed.min(BACKOFF_CAP)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff_delay_exponential_without_retry_after() {
        assert_eq!(backoff_delay(1, None), Duration::from_secs(1));
        assert_eq!(backoff_delay(2, None), Duration::from_secs(2));
        assert_eq!(backoff_delay(3, None), Duration::from_secs(4));
        // Capped at BACKOFF_CAP (8s) even though 2^(4-1) = 8s exactly; confirm attempt 5 (16s
        // uncapped) is clamped too.
        assert_eq!(backoff_delay(4, None), Duration::from_secs(8));
        assert_eq!(backoff_delay(5, None), Duration::from_secs(8));
    }

    #[test]
    fn test_backoff_delay_zero_attempt_does_not_panic() {
        // `attempt.saturating_sub(1)` guards against underflow if ever called with 0.
        assert_eq!(backoff_delay(0, None), Duration::from_secs(1));
    }

    #[test]
    fn test_backoff_delay_uses_retry_after_when_present() {
        assert_eq!(
            backoff_delay(1, Some(Duration::from_secs(3))),
            Duration::from_secs(3)
        );
        // retry_after takes priority over the computed exponential value even at a later attempt.
        assert_eq!(
            backoff_delay(3, Some(Duration::from_secs(5))),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn test_backoff_delay_caps_large_retry_after() {
        assert_eq!(
            backoff_delay(1, Some(Duration::from_secs(120))),
            RETRY_AFTER_CAP
        );
    }
}
