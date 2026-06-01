pub mod api;
pub mod oauth;
mod shared;

pub use api::ClaudeApiProvider;
pub use oauth::ClaudeOAuthProvider;
/// Re-exported so config-layer validation can tell whether a model uses adaptive thinking (no
/// explicit `budget_tokens`) vs the budgeted path (where `max_output_tokens` must exceed the
/// budget). Keeps the model-name logic in one place.
pub(crate) use shared::model_supports_adaptive_thinking;
