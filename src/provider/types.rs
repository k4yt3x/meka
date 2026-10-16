//! The wire-neutral vocabulary every backend speaks: messages and their content blocks, tool
//! definitions, stream events, usage and thinking settings.

use super::*;

/// Normalized account rate-limit usage, as returned by a subscription provider's usage endpoint
/// (Claude OAuth's `/api/oauth/usage`, Codex's `/wham/usage`). Provider-agnostic so one renderer
/// serves every backend; providers map their native shapes into [`UsageWindow`]s.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct AccountUsage {
    /// Rolling rate-limit windows (e.g. the 5-hour session window and the weekly window), in the
    /// order the provider reported them.
    pub(crate) windows: Vec<UsageWindow>,
    /// Pay-as-you-go / extra-usage (overage credit) state, when the provider reports it.
    pub(crate) extra_usage: Option<ExtraUsage>,
    /// Optional one-line addendum (e.g. the plan name) shown beneath the windows. `None` when the
    /// provider offered nothing extra.
    pub(crate) note: Option<String>,
}
/// The shared reasoning-effort policy, applied identically by every provider that exposes an effort
/// knob (Claude `output_config.effort`, OpenAI `reasoning.effort`).
///
/// Effort is a request parameter the *provider* owns: leaving the field off is not a degraded
/// setting, it is how you ask for the provider's own default. meka therefore sends it only when the
/// profile asks for one. A configured value is passed through verbatim (trimmed + lowercased) and
/// is **absolute**: never clamped, never dropped, whatever model it is aimed at; the user owns
/// correctness for their model and endpoint. A blank value (empty or whitespace-only) reads as
/// unset.
///
/// meka deliberately picks no default of its own. It cannot know what tiers a given endpoint
/// implements (`anthropic-messages` and `openai-chat-completions` reach any compatible server,
/// including local ones serving weights that never had an effort knob), and a tier the backend
/// does not implement is a rejected request rather than a graceful ignore.
pub(crate) fn resolve_effort_level(configured: Option<&str>) -> Option<String> {
    configured
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
}
/// Pay-as-you-go / extra-usage (overage credits) state. Normalized from Anthropic's `extra_usage` +
/// `spend` blocks and Codex's `credits` + `spend_control` blocks; every numeric field is optional
/// because the two providers report different subsets.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExtraUsage {
    /// Whether extra usage / pay-as-you-go is enabled on the account.
    pub(crate) enabled: bool,
    /// Percent of the extra-usage / spend limit consumed (`0.0..=100.0`), if reported.
    pub(crate) utilization: Option<f64>,
    /// Amount spent this period, in `currency`, if reported.
    pub(crate) used: Option<f64>,
    /// Remaining credit balance, in `currency`, if reported.
    pub(crate) balance: Option<f64>,
    /// Currency code (e.g. `"USD"`); `None` when the provider didn't say.
    pub(crate) currency: Option<String>,
}
/// A single rolling usage window.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct UsageWindow {
    /// Human label, e.g. `"5-hour (session)"` or `"Weekly"`.
    pub(crate) label: String,
    /// Percentage of the window consumed, `0.0..=100.0`.
    pub(crate) used_percent: f64,
    /// When the window resets, as a Unix timestamp in seconds. `None` if the provider didn't say.
    pub(crate) resets_at: Option<i64>,
}
/// Normalized account identity, from a subscription provider's profile endpoint (Claude OAuth's
/// `/api/oauth/profile` + `/api/oauth/claude_cli/roles`, Codex's `plan_type`). Every field is
/// optional so each backend fills what it can. Serialized as the `identity` block of `meka account
/// whoami --format json`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct AccountIdentity {
    pub(crate) display_name: Option<String>,
    pub(crate) email: Option<String>,
    /// Plan label, e.g. `"claude_max"`, `"pro"`, `"plus"`.
    pub(crate) plan: Option<String>,
    /// Rate-limit tier, e.g. `"default_claude_max_20x"`.
    pub(crate) tier: Option<String>,
    pub(crate) subscription_status: Option<String>,
    pub(crate) organization: Option<String>,
    /// Organization role, e.g. `"admin"`.
    pub(crate) role: Option<String>,
}
/// Normalized historical usage, from a provider's stats endpoint (Codex's `/wham/profiles/me`,
/// Claude's `/api/organization/claude_code_first_token_date`). Fields are optional because the
/// providers report very different amounts: Codex is rich (lifetime/daily/streaks), Claude offers
/// only a first-used date.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct UsageHistory {
    pub(crate) lifetime_tokens: Option<i64>,
    pub(crate) peak_daily_tokens: Option<i64>,
    pub(crate) current_streak_days: Option<i64>,
    pub(crate) longest_streak_days: Option<i64>,
    /// When the account first used the tool (RFC 3339 or `YYYY-MM-DD`), if known.
    pub(crate) first_used: Option<String>,
    /// Per-day token counts, in the order the provider returned them.
    pub(crate) daily: Vec<DailyUsage>,
}
/// One day's token count in [`UsageHistory::daily`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct DailyUsage {
    pub(crate) date: String,
    pub(crate) tokens: i64,
}
/// Name of the tool that loads a deferred tool's schema. Calls to it are scanned out of the
/// conversation to compute the per-turn active tool set; see
/// `crate::tools::load_tool::extract_loaded_tool_names_from_events`.
pub(crate) const LOAD_TOOL_NAME: &str = "tool_load";
/// Name of the tool that searches the registry by keyword, `tool_load`'s sibling. Also what the
/// Responses wire's own search item is recorded as, so one dispatch door serves both.
pub(crate) const SEARCH_TOOL_NAME: &str = "tool_search";

/// How a wire carries the tools a request defers.
///
/// What the model sees of a deferred tool is the same under all three: a catalog entry until it
/// loads the tool, the schema afterwards. What differs is where the schema travels, which decides
/// whether a load rewrites the cached prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeferredToolDelivery {
    /// A loaded tool joins the `tools` array on the next request, which moves everything cached
    /// behind the array. The one shape every endpoint takes.
    Appended,
    /// Every deferred tool is declared from the first request with `defer_loading`, and a load
    /// answers with `tool_reference` blocks the endpoint expands where they sit, so the array
    /// never changes. Anthropic's endpoints.
    Referenced,
    /// Every deferred tool is declared with `defer_loading`, and the endpoint's own `tool_search`
    /// with client execution both finds and loads, answered by a `tool_search_output` carrying
    /// the definitions; meka's two discovery tools are not declared beside it. OpenAI's
    /// Responses endpoints.
    Searched,
}

impl DeferredToolDelivery {
    /// Whether deferred tools are declared on the first request rather than appended on a load.
    pub(crate) const fn declares(self) -> bool {
        !matches!(self, Self::Appended)
    }

    /// Whether the wire loads a deferred tool through its own search item, named by exact name,
    /// rather than through `tool_load`.
    pub(crate) const fn loads_by_search(self) -> bool {
        matches!(self, Self::Searched)
    }
}

/// The tool the model is told to load a deferred tool with: `tool_search` by exact name on a wire
/// whose own search item does both jobs, `tool_load` elsewhere.
pub(crate) const fn load_door(loads_by_search: bool) -> &'static str {
    if loads_by_search {
        SEARCH_TOOL_NAME
    } else {
        LOAD_TOOL_NAME
    }
}

/// Whether `name` is one of the two tools that exist to find and load deferred tools, which the
/// request and the context block carry only while something is deferred.
pub(crate) fn is_discovery_tool(name: &str) -> bool {
    name == LOAD_TOOL_NAME || name == SEARCH_TOOL_NAME
}

/// The parameters of the search tool, as the wire declares them. Where meka declares the function,
/// `query` alone and required: `tool_load` is what loads. Under
/// [`DeferredToolDelivery::Searched`] the endpoint's own search item is the one door that
/// registers a definition, so it takes `names` beside `query`, and the dispatch of its calls is
/// held to that shape rather than the function's.
pub(crate) fn search_tool_parameters(delivery: DeferredToolDelivery) -> serde_json::Value {
    const QUERY: &str = "Words to look for in tool names and descriptions.";
    match delivery {
        DeferredToolDelivery::Searched => serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": QUERY},
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact tool names to load instead of searching.",
                },
            },
        }),
        DeferredToolDelivery::Appended | DeferredToolDelivery::Referenced => serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": QUERY},
            },
            "required": ["query"],
        }),
    }
}

/// A tool as every backend advertises it to the model: name, description and JSON schema.
#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct ToolDefinition {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: serde_json::Value,
    /// Human-readable title for the tool, optionally set by MCP servers. Providers may render this
    /// in UIs instead of the machine name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) title: Option<String>,
    /// MCP `tool.annotations`: hints such as `readOnlyHint`, `destructiveHint`, `openWorldHint`.
    /// Passed through verbatim as JSON; providers that don't recognize the field ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) annotations: Option<serde_json::Value>,
    /// MCP `tool.meta` payload, forwarded verbatim so permission heuristics and audit logs can
    /// access it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) meta: Option<serde_json::Value>,
}
#[cfg(test)]
impl ToolDefinition {
    /// Test-only convenience constructor. Production code builds `ToolDefinition` as a struct
    /// literal and explicitly sets the MCP-specific `title`/`annotations`/`meta` fields; this
    /// helper just keeps test fixtures terse.
    pub(crate) fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            title: None,
            annotations: None,
            meta: None,
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    /// The model has entered a thinking block, carrying the server's running estimate of the
    /// thinking tokens spent so far when it offers one (`None` at the start of the block, before
    /// any estimate has arrived).
    ///
    /// Separate from [`Self::ThinkingDelta`] because thinking can be *silent*: under Claude's
    /// `redact-thinking` beta or display updates every delta carries an empty string, so a UI
    /// driven by deltas alone shows nothing at all for the whole reasoning phase. This event is
    /// the liveness signal, and the estimate is what distinguishes "still working" from
    /// "wedged".
    ThinkingProgress {
        estimated_tokens: Option<u64>,
    },
    ThinkingComplete {
        /// Whatever the provider gave to carry this reasoning forward, in its own shape. Passed
        /// through to [`ContentBlock::Thinking`] untouched.
        opaque: Option<OpaqueReasoning>,
    },
    /// A complete `redacted_thinking` block (the `redact-thinking` beta). `data` is opaque and
    /// arrives whole in the `content_block_start` event, so there is no delta/complete pair.
    RedactedThinking {
        data: String,
    },
    ToolUseStart {
        id: String,
        name: String,
    },
    ToolUseEnd {
        input: serde_json::Value,
    },
    /// Emitted in lieu of `ToolUseEnd` when the accumulated tool-call arguments fail to parse as
    /// JSON. The agent layer must not execute the tool; it should surface the parse error back to
    /// the model as a `ToolResult { is_error: true }` instead.
    ToolCallRefused {
        id: String,
        name: String,
        reason: String,
    },
    MessageEnd {
        stop_reason: StopReason,
    },
    Usage(TokenUsage),
    /// User-visible advisory from the provider layer (e.g. "redacted N old images to fit the
    /// 32 MiB request limit"). The agent translates this into
    /// [`crate::frontend::FrontendEvent::Notice`] so every frontend renders it consistently.
    /// Distinct from `Error`: the request itself is proceeding successfully; the notice
    /// describes a side-effect the user should know about.
    Notice(Notice),
    Error(String),
}
/// Sentinel key inserted into `ToolUse::input` when the upstream tool-call arguments failed to
/// parse. `resolve_and_execute_tool` checks for this and short-circuits to an error result instead
/// of invoking the tool with a potentially surprising default-filled object.
pub(crate) const INVALID_TOOL_ARGS_MARKER: &str = "_meka_invalid_arguments";
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    /// The model declined to comply with the request. Claude's API surfaces this as `stop_reason:
    /// "refusal"`; OpenAI's responses API has the equivalent. The string carries the model's
    /// refusal text when the provider includes one, empty otherwise.
    Refusal(String),
    Unknown(String),
}

/// Whether a request keeps the profile's thinking setting or turns it off for this one call. A
/// request can only turn thinking *off*: nothing resurrects a mode the profile disabled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ThinkingOverride {
    #[default]
    Inherit,
    Off,
}

/// One completion, as every backend receives it: the prompt, the conversation, the tools on offer,
/// and the per-call thinking override. Borrowed, because the redact-and-retry loop re-invokes a
/// body builder over a substituted message slice and nothing else changes.
#[derive(Clone, Debug)]
pub(crate) struct CompletionRequest<'a> {
    pub(crate) system_prompt: &'a str,
    pub(crate) messages: &'a [Message],
    pub(crate) tools: &'a [ToolDefinition],
    /// The deferred tools a declaring wire lists beside `tools` with `defer_loading`; empty on a
    /// wire that appends, and on every request that carries no deferred tool.
    pub(crate) deferred_tools: &'a [ToolDefinition],
    pub(crate) thinking: ThinkingOverride,
    /// Who the request is for; see [`crate::provider::Attribution`].
    pub(crate) attribution: crate::provider::Attribution,
}

impl<'a> CompletionRequest<'a> {
    pub(crate) fn new(
        system_prompt: &'a str,
        messages: &'a [Message],
        tools: &'a [ToolDefinition],
    ) -> Self {
        Self {
            system_prompt,
            messages,
            tools,
            deferred_tools: &[],
            thinking: ThinkingOverride::Inherit,
            attribution: crate::provider::Attribution::default(),
        }
    }

    /// Declare `deferred_tools` beside the tools, for a wire that carries them that way.
    pub(crate) fn declaring(mut self, deferred_tools: &'a [ToolDefinition]) -> Self {
        self.deferred_tools = deferred_tools;
        self
    }

    /// Say who the request is for.
    pub(crate) fn attributed(mut self, attribution: crate::provider::Attribution) -> Self {
        self.attribution = attribution;
        self
    }

    /// The same request with thinking turned off, for a call whose answer is not worth reasoning
    /// over: the compaction summary.
    pub(crate) fn without_thinking(self) -> Self {
        Self {
            thinking: ThinkingOverride::Off,
            ..self
        }
    }
}
