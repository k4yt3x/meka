//! Anthropic Claude provider. Handles the Messages API (streaming and
//! non-streaming), OAuth/PKCE token exchange and refresh, request-body
//! patching for the Claude Code billing/attestation header, and per-call
//! extended-thinking overrides.

use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::error::{AgshError, Result};
use crate::session::TokenStore;

use super::{
    AuthCredential, ContentBlock, DEFAULT_CLAUDE_CLIENT_ID, Message, Provider, Role, StopReason,
    StreamEvent, TokenUsage, ToolDefinition,
};

/// Claude Code version string. Single source of truth defined in `build.rs`.
const CC_VERSION: &str = env!("CC_VERSION");

/// Claude Code system prompt prefix.
const CC_SYSTEM_PROMPT_PREFIX: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

/// Fingerprint salt. Must match claude-code `src/utils/fingerprint.ts`.
const FINGERPRINT_SALT: &str = "59cf53e54c78";

/// `SHA256(SALT + msg[4] + msg[7] + msg[20] + version)[:3]`.
fn compute_fingerprint(message_text: &str, version: &str) -> String {
    let indices = [4, 7, 20];
    let chars: String = indices
        .iter()
        .map(|&index| message_text.chars().nth(index).unwrap_or('0'))
        .collect();

    let input = format!("{}{}{}", FINGERPRINT_SALT, chars, version);
    let hash = Sha256::digest(input.as_bytes());
    let hex = format!("{:x}", hash);
    hex[..3].to_string()
}

/// Extracts the text content of the first user message.
#[allow(dead_code)]
fn extract_first_user_message_text(messages: &[Message]) -> String {
    for message in messages {
        if message.role == Role::User {
            for block in &message.content {
                if let ContentBlock::Text { text } = block {
                    return text.clone();
                }
            }
        }
    }
    String::new()
}

/// Computes the fingerprint from the first user message.
#[allow(dead_code)]
fn compute_fingerprint_from_messages(messages: &[Message]) -> String {
    let first_message_text = extract_first_user_message_text(messages);
    compute_fingerprint(&first_message_text, CC_VERSION)
}

/// Static fingerprint from an empty message for cross-session cache sharing.
static STATIC_FINGERPRINT: LazyLock<String> = LazyLock::new(|| compute_fingerprint("", CC_VERSION));

fn now_epoch_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

/// Generates the billing header with a `cch=00000` placeholder.
fn generate_billing_header() -> String {
    format!(
        "x-anthropic-billing-header: cc_version={}.{}; cc_entrypoint=cli; cch=00000;",
        CC_VERSION, *STATIC_FINGERPRINT
    )
}

// xxHash64 with Claude-specific seed. See ATTESTATION.md for details.

const XXH64_PRIME1: u64 = 0x9e3779b185ebca87;
const XXH64_PRIME2: u64 = 0xc2b2ae3d27d4eb4f;
const XXH64_PRIME3: u64 = 0x165667b19e3779f9;
const XXH64_PRIME4: u64 = 0x85ebca77c2b2ae63;
const XXH64_PRIME5: u64 = 0x27d4eb2f165667c5;

/// Claude Code attestation seed.
const CCH_XXH64_SEED: u64 = 0x6e52736ac806831e;

fn xxh64_round(acc: u64, lane: u64) -> u64 {
    acc.wrapping_add(lane.wrapping_mul(XXH64_PRIME2))
        .rotate_left(31)
        .wrapping_mul(XXH64_PRIME1)
}

fn xxh64_merge_round(acc: u64, val: u64) -> u64 {
    (acc ^ xxh64_round(0, val))
        .wrapping_mul(XXH64_PRIME1)
        .wrapping_add(XXH64_PRIME4)
}

fn xxh64_avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(XXH64_PRIME2);
    h ^= h >> 29;
    h = h.wrapping_mul(XXH64_PRIME3);
    h ^= h >> 32;
    h
}

fn read_u32_le(buf: &[u8], offset: usize) -> u64 {
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ]) as u64
}

fn read_u64_le(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
        buf[offset + 4],
        buf[offset + 5],
        buf[offset + 6],
        buf[offset + 7],
    ])
}

fn xxh64(input: &[u8], seed: u64) -> u64 {
    let len = input.len();
    let mut p = 0usize;
    let mut h64: u64;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(XXH64_PRIME1).wrapping_add(XXH64_PRIME2);
        let mut v2 = seed.wrapping_add(XXH64_PRIME2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(XXH64_PRIME1);

        let limit = len - 32;
        while p <= limit {
            v1 = xxh64_round(v1, read_u64_le(input, p));
            p += 8;
            v2 = xxh64_round(v2, read_u64_le(input, p));
            p += 8;
            v3 = xxh64_round(v3, read_u64_le(input, p));
            p += 8;
            v4 = xxh64_round(v4, read_u64_le(input, p));
            p += 8;
        }

        h64 = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed.wrapping_add(XXH64_PRIME5);
    }

    h64 = h64.wrapping_add(len as u64);

    while p + 8 <= len {
        let k1 = xxh64_round(0, read_u64_le(input, p));
        p += 8;
        h64 ^= k1;
        h64 = h64
            .rotate_left(27)
            .wrapping_mul(XXH64_PRIME1)
            .wrapping_add(XXH64_PRIME4);
    }

    if p + 4 <= len {
        h64 ^= read_u32_le(input, p).wrapping_mul(XXH64_PRIME1);
        p += 4;
        h64 = h64
            .rotate_left(23)
            .wrapping_mul(XXH64_PRIME2)
            .wrapping_add(XXH64_PRIME3);
    }

    while p < len {
        h64 ^= (input[p] as u64).wrapping_mul(XXH64_PRIME5);
        p += 1;
        h64 = h64.rotate_left(11).wrapping_mul(XXH64_PRIME1);
    }

    xxh64_avalanche(h64)
}

/// Replaces the `cch=00000` placeholder with xxHash64(body) & 0xFFFFF.
/// Anchors the search to the billing header to avoid false matches in messages.
fn patch_request_body(body_json: &str) -> Result<String> {
    const BILLING_PREFIX: &str = "x-anthropic-billing-header:";
    const PLACEHOLDER: &str = "cch=00000";

    let billing_start = body_json.find(BILLING_PREFIX).ok_or_else(|| {
        AgshError::Provider("x-anthropic-billing-header not found in request body".into())
    })?;

    let idx = body_json[billing_start..]
        .find(PLACEHOLDER)
        .map(|relative| billing_start + relative)
        .ok_or_else(|| {
            AgshError::Provider(
                "cch=00000 attestation placeholder not found in billing header".into(),
            )
        })?;

    let digest = xxh64(body_json.as_bytes(), CCH_XXH64_SEED);
    let token = format!("{:05x}", digest & 0xfffff);

    let mut patched = String::with_capacity(body_json.len());
    patched.push_str(&body_json[..idx + 4]); // up to and including "cch="
    patched.push_str(&token);
    patched.push_str(&body_json[idx + 9..]); // skip past "00000"
    Ok(patched)
}

/// Builds the User-Agent string matching claude-code's format.
fn claude_user_agent() -> String {
    format!("claude-cli/{} (external, cli)", CC_VERSION)
}

/// Stainless SDK versions. Must match the release corresponding to `CC_VERSION`.
const STAINLESS_BUN_VERSION: &str = "1.2.15";
const STAINLESS_SDK_VERSION: &str = "0.52.1";

/// Maps `std::env::consts::ARCH` to Node.js/Bun `process.arch` names.
fn stainless_arch() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x64",
        "x86" => "ia32",
        "aarch64" => "arm64",
        "arm" => "arm",
        "s390x" => "s390x",
        "powerpc64" => "ppc64",
        other => other,
    }
}

fn stainless_os() -> &'static str {
    match std::env::consts::OS {
        "macos" => "MacOS",
        "windows" => "Windows",
        "linux" => "Linux",
        "freebsd" => "FreeBSD",
        other => other,
    }
}

/// Applies all HTTP headers in the order the Stainless SDK + Claude Code
/// would produce on the wire. See `buildHeaders` in `@anthropic-ai/sdk`.
fn apply_headers(
    request: reqwest::RequestBuilder,
    auth_header_name: &str,
    auth_header_value: &str,
    session_id: &str,
    betas: Option<&str>,
) -> reqwest::RequestBuilder {
    // --- SDK buildDefaultHeaders() ---
    // Accept, User-Agent, retry/timeout, platform headers, anthropic-version
    let mut request = request
        .header("accept", "application/json")
        .header("User-Agent", claude_user_agent())
        .header("x-stainless-retry-count", "0")
        .header("x-stainless-timeout", "600")
        .header("x-stainless-lang", "js")
        .header("x-stainless-package-version", STAINLESS_SDK_VERSION)
        .header("x-stainless-os", stainless_os())
        .header("x-stainless-arch", stainless_arch())
        .header("x-stainless-runtime", "bun")
        .header("x-stainless-runtime-version", STAINLESS_BUN_VERSION)
        .header("anthropic-version", "2023-06-01")
        // --- authHeaders ---
        .header(auth_header_name, auth_header_value)
        // --- Claude Code defaultHeaders (x-app, session-id; User-Agent updates in-place above) ---
        .header("x-app", "cli")
        .header("X-Claude-Code-Session-Id", session_id)
        // --- bodyHeaders ---
        .header("content-type", "application/json")
        // --- per-request headers ---
        .header("x-client-request-id", Uuid::new_v4().to_string());

    if let Some(betas) = betas {
        request = request.header("anthropic-beta", betas);
    }

    request
}

pub struct ClaudeProvider {
    client: reqwest::Client,
    credential: tokio::sync::RwLock<AuthCredential>,
    base_url: String,
    model: String,
    client_id: String,
    oauth_token_url: String,
    token_store: Option<Arc<TokenStore>>,
    session_id: String,
    thinking_enabled: bool,
    thinking_budget_tokens: u64,
    thinking_override: AtomicI8,
}

/// Parse (major, minor) version from a Claude model string.
/// E.g., "claude-opus-4-6-20250514" → Some((4, 6)), "claude-sonnet-4" → Some((4, 0)).
fn parse_claude_model_version(model: &str) -> Option<(u32, u32)> {
    for family in &["opus", "sonnet", "haiku"] {
        if let Some(pos) = model.find(family) {
            let after = &model[pos + family.len()..];
            let after = after.strip_prefix('-')?;
            let mut parts = after.splitn(3, '-');
            let major: u32 = parts.next()?.parse().ok()?;
            let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            return Some((major, minor));
        }
    }
    None
}

fn model_supports_adaptive_thinking(model: &str) -> bool {
    parse_claude_model_version(model)
        .is_some_and(|(major, minor)| major > 4 || (major == 4 && minor >= 6))
}

impl ClaudeProvider {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        credential: AuthCredential,
        model: String,
        base_url: Option<String>,
        client_id: Option<String>,
        oauth_token_url: Option<String>,
        token_store: Option<Arc<TokenStore>>,
        thinking_enabled: bool,
        thinking_budget_tokens: u64,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            credential: tokio::sync::RwLock::new(credential),
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            model,
            client_id: client_id.unwrap_or_else(|| DEFAULT_CLAUDE_CLIENT_ID.to_string()),
            oauth_token_url: oauth_token_url
                .unwrap_or_else(|| "https://api.anthropic.com/v1/oauth/token".to_string()),
            token_store,
            session_id: Uuid::new_v4().to_string(),
            thinking_enabled,
            thinking_budget_tokens,
            thinking_override: AtomicI8::new(-1),
        }
    }

    fn is_thinking_enabled(&self) -> bool {
        match self.thinking_override.load(Ordering::Relaxed) {
            0 => false,
            1 => true,
            _ => self.thinking_enabled,
        }
    }

    fn compute_betas(&self, auth_header_name: &str) -> Option<String> {
        let mut parts = Vec::new();
        if auth_header_name == "Authorization" {
            parts.push("claude-code-20250219");
            parts.push("oauth-2025-04-20");
        }
        if self.is_thinking_enabled() {
            parts.push("interleaved-thinking-2025-05-14");
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(","))
        }
    }

    async fn ensure_valid_credential(&self) -> Result<(&'static str, String)> {
        {
            let credential = self.credential.read().await;
            match &*credential {
                AuthCredential::ApiKey(key) => {
                    return Ok(("x-api-key", key.clone()));
                }
                AuthCredential::OAuthToken {
                    access_token,
                    expires_at,
                    ..
                } => {
                    let needs_refresh = if let Some(exp) = expires_at {
                        now_epoch_millis() + 300_000 >= *exp
                    } else {
                        false
                    };

                    if !needs_refresh {
                        return Ok(("Authorization", format!("Bearer {}", access_token)));
                    }
                }
            }
        }

        // Token expired — attempt refresh
        let mut credential = self.credential.write().await;

        // Double-check after acquiring write lock
        if let AuthCredential::OAuthToken {
            access_token,
            expires_at,
            ..
        } = &*credential
        {
            let needs_refresh = if let Some(exp) = expires_at {
                now_epoch_millis() + 300_000 >= *exp
            } else {
                false
            };

            if !needs_refresh {
                return Ok(("Authorization", format!("Bearer {}", access_token)));
            }
        }

        let refresh_token = match &*credential {
            AuthCredential::OAuthToken { refresh_token, .. } => refresh_token.clone(),
            _ => None,
        };

        let Some(refresh_token) = refresh_token else {
            return Err(AgshError::Provider(
                "OAuth access token expired and no refresh token available".to_string(),
            ));
        };

        let new_credential = self.refresh_oauth_token(&refresh_token).await?;
        let (header_name, header_value) = new_credential.auth_header();

        if let Some(store) = &self.token_store
            && let Err(error) = store.save_oauth_token("claude", &new_credential).await
        {
            tracing::warn!("failed to persist refreshed OAuth token: {}", error);
        }

        *credential = new_credential;
        Ok((header_name, header_value))
    }

    async fn refresh_oauth_token(&self, refresh_token: &str) -> Result<AuthCredential> {
        tracing::info!("refreshing OAuth token");

        let response = self
            .client
            .post(&self.oauth_token_url)
            .json(&serde_json::json!({
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            }))
            .send()
            .await
            .map_err(|error| {
                AgshError::Provider(format!("OAuth token refresh request failed: {}", error))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AgshError::Provider(format!(
                "OAuth token refresh failed ({}): {}",
                status, body
            )));
        }

        #[derive(Deserialize)]
        struct RefreshResponse {
            access_token: String,
            refresh_token: Option<String>,
            expires_in: Option<u64>,
        }

        let data: RefreshResponse = response.json().await.map_err(|error| {
            AgshError::Provider(format!("failed to parse refresh response: {}", error))
        })?;

        let expires_at = data
            .expires_in
            .map(|seconds| now_epoch_millis() + (seconds as i64) * 1000);

        Ok(AuthCredential::OAuthToken {
            access_token: data.access_token,
            refresh_token: data
                .refresh_token
                .or_else(|| Some(refresh_token.to_string())),
            expires_at,
        })
    }

    pub(super) fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let message_count = messages.len();
        let mut claude_messages: Vec<serde_json::Value> = messages
            .iter()
            .enumerate()
            .map(|(message_index, message)| {
                let role = match message.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };

                let is_last_message = message_index + 1 == message_count;
                let block_count = message.content.len();

                let content: Vec<serde_json::Value> = message
                    .content
                    .iter()
                    .enumerate()
                    .map(|(block_index, block)| {
                        let mut value = match block {
                            ContentBlock::Text { text } => {
                                serde_json::json!({
                                    "type": "text",
                                    "text": text,
                                })
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                serde_json::json!({
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": input,
                                })
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } => {
                                serde_json::json!({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": content,
                                    "is_error": is_error,
                                })
                            }
                            ContentBlock::Thinking {
                                thinking,
                                signature,
                            } => {
                                let mut obj = serde_json::json!({
                                    "type": "thinking",
                                    "thinking": thinking
                                });
                                if let Some(sig) = signature {
                                    obj["signature"] = serde_json::json!(sig);
                                }
                                obj
                            }
                        };

                        if is_last_message
                            && block_index + 1 == block_count
                            && let Some(obj) = value.as_object_mut()
                        {
                            obj.insert(
                                "cache_control".to_string(),
                                serde_json::json!({"type": "ephemeral"}),
                            );
                        }

                        value
                    })
                    .collect();

                serde_json::json!({
                    "role": role,
                    "content": content,
                })
            })
            .collect();

        // Strip trailing thinking blocks from the last assistant message
        // (Claude API requirement).
        if let Some(last_assistant) = claude_messages
            .iter_mut()
            .rev()
            .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("assistant"))
            && let Some(content) = last_assistant
                .get_mut("content")
                .and_then(|c| c.as_array_mut())
        {
            while content
                .last()
                .and_then(|b| b.get("type"))
                .and_then(|t| t.as_str())
                == Some("thinking")
            {
                content.pop();
            }
            if content.is_empty() {
                content.push(serde_json::json!({
                    "type": "text",
                    "text": "[No message content]"
                }));
            }
        }

        let metadata_user_id = serde_json::json!({
            "device_id": "agsh",
            "account_uuid": "",
            "session_id": self.session_id,
        })
        .to_string();

        // `system` must precede `messages` so the billing header's `cch=00000`
        // is always the first occurrence in the serialized JSON.
        let mut body = serde_json::Map::new();

        if !system_prompt.is_empty() {
            let billing_header = generate_billing_header();
            body.insert(
                "system".to_string(),
                serde_json::json!([
                    {
                        "type": "text",
                        "text": billing_header
                    },
                    {
                        "type": "text",
                        "text": CC_SYSTEM_PROMPT_PREFIX,
                        "cache_control": { "type": "ephemeral" }
                    },
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": { "type": "ephemeral" }
                    }
                ]),
            );
        }

        body.insert("model".to_string(), serde_json::json!(self.model));
        body.insert("messages".to_string(), serde_json::json!(claude_messages));

        if self.is_thinking_enabled() {
            if model_supports_adaptive_thinking(&self.model) {
                body.insert("max_tokens".to_string(), serde_json::json!(64_000));
                body.insert(
                    "thinking".to_string(),
                    serde_json::json!({ "type": "adaptive" }),
                );
            } else {
                let budget = self.thinking_budget_tokens;
                let max_tokens = std::cmp::max(budget * 2, 32_000);
                body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
                body.insert(
                    "thinking".to_string(),
                    serde_json::json!({
                        "type": "enabled",
                        "budget_tokens": budget
                    }),
                );
            }
        } else {
            body.insert("max_tokens".to_string(), serde_json::json!(32_000));
        }

        body.insert("stream".to_string(), serde_json::json!(stream));
        body.insert(
            "metadata".to_string(),
            serde_json::json!({ "user_id": metadata_user_id }),
        );

        if !tools.is_empty() {
            let tool_count = tools.len();
            let claude_tools: Vec<serde_json::Value> = tools
                .iter()
                .enumerate()
                .map(|(index, tool)| {
                    let mut schema = serde_json::json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    });
                    if index + 1 == tool_count
                        && let Some(obj) = schema.as_object_mut()
                    {
                        obj.insert(
                            "cache_control".to_string(),
                            serde_json::json!({"type": "ephemeral"}),
                        );
                    }
                    schema
                })
                .collect();
            body.insert("tools".to_string(), serde_json::json!(claude_tools));
        }

        serde_json::Value::Object(body)
    }

    pub(super) fn parse_non_streaming_response(
        &self,
        response: &serde_json::Value,
    ) -> Result<(Message, StopReason, TokenUsage)> {
        let stop_reason_str = response
            .get("stop_reason")
            .and_then(|reason| reason.as_str())
            .unwrap_or("end_turn");

        let stop_reason = parse_claude_stop_reason(stop_reason_str);

        let token_usage = TokenUsage {
            input_tokens: response
                .get("usage")
                .and_then(|u| u.get("input_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: response
                .get("usage")
                .and_then(|u| u.get("output_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        };

        let content_array = response
            .get("content")
            .and_then(|content| content.as_array())
            .ok_or_else(|| AgshError::Provider("no content array in response".to_string()))?;

        let mut content_blocks = Vec::new();

        for block in content_array {
            let block_type = block
                .get("type")
                .and_then(|block_type| block_type.as_str())
                .unwrap_or("");

            match block_type {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|text| text.as_str()) {
                        content_blocks.push(ContentBlock::Text {
                            text: text.to_string(),
                        });
                    }
                }
                "tool_use" => {
                    let id = block
                        .get("id")
                        .and_then(|id| id.as_str())
                        .ok_or_else(|| {
                            AgshError::Provider("tool_use block missing 'id' field".to_string())
                        })?
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|name| name.as_str())
                        .ok_or_else(|| {
                            AgshError::Provider("tool_use block missing 'name' field".to_string())
                        })?
                        .to_string();
                    let input = block.get("input").cloned().unwrap_or_else(|| {
                        tracing::warn!("missing 'input' in tool_use block");
                        serde_json::json!({})
                    });

                    content_blocks.push(ContentBlock::ToolUse { id, name, input });
                }
                "thinking" => {
                    if let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) {
                        let signature = block
                            .get("signature")
                            .and_then(|s| s.as_str())
                            .map(|s| s.to_string());
                        content_blocks.push(ContentBlock::Thinking {
                            thinking: thinking.to_string(),
                            signature,
                        });
                    }
                }
                _ => {
                    tracing::warn!("unknown Claude content block type: {}", block_type);
                }
            }
        }

        Ok((
            Message {
                role: Role::Assistant,
                content: content_blocks,
            },
            stop_reason,
            token_usage,
        ))
    }
}

#[async_trait]
impl Provider for ClaudeProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason, TokenUsage)> {
        let body = self.build_request_body(system_prompt, messages, tools, false);
        let body_json = serde_json::to_string(&body)
            .map_err(|error| AgshError::Provider(format!("failed to serialize body: {}", error)))?;
        let body_json = if !system_prompt.is_empty() {
            patch_request_body(&body_json)?
        } else {
            body_json
        };
        let (auth_header_name, auth_header_value) = self.ensure_valid_credential().await?;

        let betas = self.compute_betas(auth_header_name);

        let request = apply_headers(
            self.client.post(format!("{}/v1/messages", self.base_url)),
            auth_header_name,
            &auth_header_value,
            &self.session_id,
            betas.as_deref(),
        );

        let response = request
            .body(body_json)
            .send()
            .await
            .map_err(|error| AgshError::Provider(format!("HTTP request failed: {}", error)))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|error| AgshError::Provider(format!("failed to read response: {}", error)))?;

        if !status.is_success() {
            return Err(AgshError::Provider(format!(
                "API returned status {}: {}",
                status, response_text
            )));
        }

        let response_json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|error| AgshError::Provider(format!("invalid JSON response: {}", error)))?;

        self.parse_non_streaming_response(&response_json)
    }

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::UnboundedSender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        use eventsource_stream::Eventsource;
        use futures::StreamExt;

        let body = self.build_request_body(system_prompt, messages, tools, true);
        let body_json = serde_json::to_string(&body)
            .map_err(|error| AgshError::Provider(format!("failed to serialize body: {}", error)))?;
        let body_json = if !system_prompt.is_empty() {
            patch_request_body(&body_json)?
        } else {
            body_json
        };
        let (auth_header_name, auth_header_value) = self.ensure_valid_credential().await?;

        let betas = self.compute_betas(auth_header_name);

        let request = apply_headers(
            self.client
                .post(format!("{}/v1/messages", self.base_url))
                .header("accept-encoding", "identity"),
            auth_header_name,
            &auth_header_value,
            &self.session_id,
            betas.as_deref(),
        );

        let response = request
            .body(body_json)
            .send()
            .await
            .map_err(|error| AgshError::Provider(format!("HTTP request failed: {}", error)))?;

        let status = response.status();
        if !status.is_success() {
            let response_text = response.text().await.unwrap_or_default();
            return Err(AgshError::Provider(format!(
                "API returned status {}: {}",
                status, response_text
            )));
        }

        let mut event_stream = response.bytes_stream().eventsource();

        let mut current_tool_input = String::new();
        let mut in_tool_use = false;
        let mut in_thinking = false;
        let mut current_thinking_signature: Option<String> = None;

        loop {
            tokio::select! {
                _ = cancellation.cancelled() => {
                    return Err(AgshError::Interrupted);
                }
                event = event_stream.next() => {
                    let Some(event) = event else {
                        break;
                    };

                    match event {
                        Ok(event) => {
                            let data: serde_json::Value = match serde_json::from_str(&event.data) {
                                Ok(data) => data,
                                Err(error) => {
                                    tracing::warn!("failed to parse SSE data: {}", error);
                                    continue;
                                }
                            };

                            match event.event.as_str() {
                                "content_block_start" => {
                                    let Some(content_block) = data.get("content_block") else {
                                        continue;
                                    };
                                    let block_type = content_block
                                        .get("type")
                                        .and_then(|block_type| block_type.as_str())
                                        .unwrap_or("");

                                    if block_type == "thinking" {
                                        in_thinking = true;
                                    } else if block_type == "tool_use" {
                                        let id = content_block
                                            .get("id")
                                            .and_then(|id| id.as_str())
                                            .ok_or_else(|| {
                                                AgshError::Provider(
                                                    "tool_use block missing 'id' field".to_string(),
                                                )
                                            })?
                                            .to_string();
                                        let name = content_block
                                            .get("name")
                                            .and_then(|name| name.as_str())
                                            .ok_or_else(|| {
                                                AgshError::Provider(
                                                    "tool_use block missing 'name' field"
                                                        .to_string(),
                                                )
                                            })?
                                            .to_string();

                                        current_tool_input.clear();
                                        in_tool_use = true;
                                        if event_sender.send(StreamEvent::ToolUseStart {
                                            id,
                                            name,
                                        }).is_err() {
                                            tracing::trace!("stream event receiver dropped");
                                            break;
                                        }
                                    }
                                }
                                "content_block_delta" => {
                                    let Some(delta) = data.get("delta") else {
                                        continue;
                                    };
                                    let delta_type = delta
                                        .get("type")
                                        .and_then(|delta_type| delta_type.as_str())
                                        .unwrap_or("");

                                    match delta_type {
                                        "thinking_delta" => {
                                            if let Some(thinking) = delta.get("thinking").and_then(|t| t.as_str())
                                                && !thinking.is_empty()
                                                    && event_sender.send(
                                                        StreamEvent::ThinkingDelta(thinking.to_string()),
                                                    ).is_err() {
                                                        tracing::trace!("stream event receiver dropped");
                                                        break;
                                                    }
                                        }
                                        "text_delta" => {
                                            if let Some(text) = delta.get("text").and_then(|text| text.as_str())
                                                && !text.is_empty()
                                                    && event_sender.send(
                                                        StreamEvent::TextDelta(text.to_string()),
                                                    ).is_err() {
                                                        tracing::trace!("stream event receiver dropped");
                                                        break;
                                                    }
                                        }
                                        "signature_delta" => {
                                            if let Some(sig) = delta.get("signature").and_then(|s| s.as_str()) {
                                                current_thinking_signature = Some(
                                                    current_thinking_signature
                                                        .map_or_else(|| sig.to_string(), |existing| existing + sig),
                                                );
                                            }
                                        }
                                        "input_json_delta" => {
                                            if let Some(partial_json) =
                                                delta.get("partial_json").and_then(|partial_json| partial_json.as_str())
                                            {
                                                current_tool_input.push_str(partial_json);
                                                if event_sender.send(
                                                    StreamEvent::ToolInputDelta(
                                                        partial_json.to_string(),
                                                    ),
                                                ).is_err() {
                                                    tracing::trace!("stream event receiver dropped");
                                                    break;
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                "content_block_stop" => {
                                    if in_thinking {
                                        in_thinking = false;
                                        let signature = current_thinking_signature.take();
                                        if event_sender
                                            .send(StreamEvent::ThinkingComplete { signature })
                                            .is_err()
                                        {
                                            tracing::trace!("stream event receiver dropped");
                                            break;
                                        }
                                    } else if in_tool_use {
                                        let input = if current_tool_input.is_empty() {
                                            serde_json::json!({})
                                        } else {
                                            match serde_json::from_str(&current_tool_input) {
                                                Ok(value) => value,
                                                Err(error) => {
                                                    tracing::warn!("failed to parse tool input JSON: {}", error);
                                                    serde_json::json!({})
                                                }
                                            }
                                        };
                                        if event_sender
                                            .send(StreamEvent::ToolUseEnd { input })
                                            .is_err()
                                        {
                                            tracing::trace!("stream event receiver dropped");
                                            break;
                                        }
                                        current_tool_input.clear();
                                        in_tool_use = false;
                                    }
                                }
                                "message_delta" => {
                                    let Some(delta) = data.get("delta") else {
                                        continue;
                                    };
                                    if let Some(usage) = data.get("usage") {
                                        let token_usage = TokenUsage {
                                            input_tokens: usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                                            output_tokens: usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                                        };
                                        if event_sender.send(StreamEvent::Usage(token_usage)).is_err() {
                                            tracing::trace!("stream event receiver dropped");
                                            break;
                                        }
                                    }
                                    if let Some(stop_reason_str) =
                                        delta.get("stop_reason").and_then(|reason| reason.as_str())
                                    {
                                        let stop_reason = parse_claude_stop_reason(stop_reason_str);
                                        if event_sender
                                            .send(StreamEvent::MessageEnd { stop_reason })
                                            .is_err()
                                        {
                                            tracing::trace!("stream event receiver dropped");
                                            break;
                                        }
                                    }
                                }
                                "message_stop" => {
                                    break;
                                }
                                "message_start" => {
                                    if let Some(usage) = data.get("message").and_then(|m| m.get("usage")) {
                                        let token_usage = TokenUsage {
                                            input_tokens: usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                                            output_tokens: usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                                        };
                                        if event_sender.send(StreamEvent::Usage(token_usage)).is_err() {
                                            tracing::trace!("stream event receiver dropped");
                                            break;
                                        }
                                    }
                                }
                                "ping" => {}
                                other => {
                                    tracing::debug!("unknown Claude SSE event: {}", other);
                                }
                            }
                        }
                        Err(error) => {
                            if event_sender.send(StreamEvent::Error(error.to_string())).is_err() {
                                tracing::trace!("stream event receiver dropped");
                            }
                            return Err(AgshError::StreamError(error.to_string()));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "claude"
    }

    fn set_thinking_override(&self, enabled: Option<bool>) {
        let value = match enabled {
            None => -1,
            Some(false) => 0,
            Some(true) => 1,
        };
        self.thinking_override.store(value, Ordering::Relaxed);
    }
}

fn parse_claude_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        other => StopReason::Unknown(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolResultContent;

    fn test_provider() -> ClaudeProvider {
        ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
            false,
            10000,
        )
    }

    #[test]
    fn test_static_fingerprint_matches_empty_message() {
        assert_eq!(*STATIC_FINGERPRINT, compute_fingerprint("", CC_VERSION));
    }

    #[test]
    fn test_compute_fingerprint_from_messages_matches_manual() {
        let messages = vec![Message::user("hello world, this is a test message!")];
        let from_messages = compute_fingerprint_from_messages(&messages);
        let first_text = extract_first_user_message_text(&messages);
        let manual = compute_fingerprint(&first_text, CC_VERSION);
        assert_eq!(from_messages, manual);
    }

    #[test]
    fn test_fingerprint_known_values() {
        let fingerprint = compute_fingerprint("hello", CC_VERSION);
        assert_eq!(fingerprint.len(), 3);
        assert!(fingerprint.chars().all(|c| c.is_ascii_hexdigit()));

        let fingerprint2 = compute_fingerprint("hello", CC_VERSION);
        assert_eq!(fingerprint, fingerprint2);

        let fingerprint3 = compute_fingerprint("this is a longer test message!!", CC_VERSION);
        assert_ne!(fingerprint, fingerprint3);
    }

    #[test]
    fn test_fingerprint_empty_message() {
        let fingerprint = compute_fingerprint("", CC_VERSION);
        assert_eq!(fingerprint.len(), 3);
        assert!(fingerprint.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_extract_first_user_message_text() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "assistant text".to_string(),
                }],
            },
            Message::user("user text"),
        ];
        assert_eq!(extract_first_user_message_text(&messages), "user text");

        let empty: Vec<Message> = vec![];
        assert_eq!(extract_first_user_message_text(&empty), "");
    }

    #[test]
    fn test_claude_request_body_simple() {
        let provider = test_provider();

        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body("system prompt", &messages, &[], false);

        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["stream"], false);

        let system = body["system"].as_array().expect("system should be array");
        assert_eq!(system.len(), 3);

        assert_eq!(system[0]["type"], "text");
        let billing = system[0]["text"].as_str().unwrap();
        let expected_prefix = format!("x-anthropic-billing-header: cc_version={}.", CC_VERSION);
        assert!(billing.starts_with(&expected_prefix), "{}", billing);
        assert!(billing.contains("cc_entrypoint=cli"));
        assert!(billing.contains("cch=00000"));
        assert!(system[0].get("cache_control").is_none());

        let after_version = billing.strip_prefix(&expected_prefix).unwrap();
        let fingerprint = after_version.split(';').next().unwrap().trim();
        assert_eq!(fingerprint, STATIC_FINGERPRINT.as_str());

        assert_eq!(system[1]["type"], "text");
        assert_eq!(system[1]["text"], CC_SYSTEM_PROMPT_PREFIX);
        assert!(system[1].get("cache_control").is_some());

        assert_eq!(system[2]["type"], "text");
        assert_eq!(system[2]["text"], "system prompt");
        assert!(system[2].get("cache_control").is_some());

        let body_json = serde_json::to_string(&body).unwrap();
        let system_pos = body_json.find("\"system\"").unwrap();
        let messages_pos = body_json.find("\"messages\"").unwrap();
        assert!(system_pos < messages_pos);

        let user_id_str = body["metadata"]["user_id"].as_str().unwrap();
        let user_id_parsed: serde_json::Value = serde_json::from_str(user_id_str).unwrap();
        assert!(user_id_parsed.get("device_id").is_some());
        assert!(user_id_parsed.get("session_id").is_some());

        let claude_messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(claude_messages.len(), 1);
        assert_eq!(claude_messages[0]["role"], "user");

        let content = claude_messages[0]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "hello");
        assert!(content[0].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_request_body_with_tools() {
        let provider = test_provider();

        let tools = vec![ToolDefinition::new(
            "read_file".to_string(),
            "Read a file".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        )];

        let body = provider.build_request_body("", &[], &tools, false);
        let claude_tools = body["tools"].as_array().expect("tools should be array");
        assert_eq!(claude_tools.len(), 1);
        assert_eq!(claude_tools[0]["name"], "read_file");
        assert_eq!(claude_tools[0]["description"], "Read a file");
        assert!(claude_tools[0].get("input_schema").is_some());
        assert!(claude_tools[0].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_request_body_with_tool_calls() {
        let provider = test_provider();

        let messages = vec![
            Message::user("read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "file contents here".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];

        let body = provider.build_request_body("", &messages, &[], false);
        let claude_messages = body["messages"]
            .as_array()
            .expect("messages should be array");

        assert_eq!(claude_messages.len(), 3);
        assert_eq!(claude_messages[0]["role"], "user");

        assert_eq!(claude_messages[1]["role"], "assistant");
        let assistant_content = claude_messages[1]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(assistant_content[0]["type"], "tool_use");
        assert_eq!(assistant_content[0]["id"], "toolu_1");
        assert_eq!(assistant_content[0]["name"], "read_file");

        assert_eq!(claude_messages[2]["role"], "user");
        let result_content = claude_messages[2]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(result_content[0]["type"], "tool_result");
        assert_eq!(result_content[0]["tool_use_id"], "toolu_1");

        let first_content = claude_messages[0]["content"]
            .as_array()
            .expect("content should be array");
        assert!(first_content[0].get("cache_control").is_none());
        assert!(assistant_content[0].get("cache_control").is_none());
        assert!(result_content[0].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_parse_non_streaming_text() {
        let provider = test_provider();

        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Hello there!"
            }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        });

        let (message, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(message.text_content(), "Hello there!");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_claude_parse_non_streaming_tool_use() {
        let provider = test_provider();

        let response = serde_json::json!({
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll read that file for you."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "read_file",
                    "input": {"path": "/tmp/test.txt"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 20, "output_tokens": 15 }
        });

        let (message, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(stop_reason, StopReason::ToolUse);
        assert_eq!(message.text_content(), "I'll read that file for you.");

        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);

        if let ContentBlock::ToolUse { id, name, input } = &tool_uses[0] {
            assert_eq!(id, "toolu_abc");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "/tmp/test.txt");
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn test_patch_request_body_replaces_placeholder() {
        let messages = vec![Message::user("hello")];
        let provider = test_provider();
        let body = provider.build_request_body("system prompt", &messages, &[], false);
        let body_json = serde_json::to_string(&body).unwrap();

        assert!(body_json.contains("cch=00000"));

        let patched = patch_request_body(&body_json).unwrap();
        assert!(!patched.contains("cch=00000"));
        let cch_idx = patched.find("cch=").expect("cch= must be present");
        let token = &patched[cch_idx + 4..cch_idx + 9];
        assert_eq!(token.len(), 5);
        assert!(token.chars().all(|c| c.is_ascii_hexdigit()), "{}", token);

        let patched2 = patch_request_body(&body_json).unwrap();
        assert_eq!(patched, patched2);
    }

    #[test]
    fn test_patch_request_body_ignores_cch_in_messages() {
        let messages = vec![Message::user(
            "The billing header contains cch=00000 as a placeholder.",
        )];
        let provider = test_provider();
        let body = provider.build_request_body("system prompt", &messages, &[], false);
        let body_json = serde_json::to_string(&body).unwrap();

        let count = body_json.matches("cch=00000").count();
        assert_eq!(count, 2, "expected 2 occurrences of cch=00000 in body");

        let patched = patch_request_body(&body_json).unwrap();

        let billing_start = patched.find("x-anthropic-billing-header:").unwrap();
        let billing_region = &patched[billing_start..billing_start + 200];
        assert!(!billing_region.contains("cch=00000"));
        assert!(patched.contains("cch=00000"));
    }

    #[test]
    fn test_xxh64_basic() {
        assert_eq!(xxh64(b"", 0), 0xef46db3751d8e999);
        assert_eq!(xxh64(b"abc", 0), 0x44bc2cf5ad770999);
    }

    #[test]
    fn test_xxh64_claude_seed_short_body() {
        let body = r#"{"test":"cch=00000"}"#;
        let digest = xxh64(body.as_bytes(), CCH_XXH64_SEED);
        let token = format!("{:05x}", digest & 0xfffff);
        assert_eq!(token, "14d28");
    }

    #[test]
    fn test_xxh64_claude_seed_realistic_body() {
        let body = concat!(
            r#"{"system":[{"type":"text","text":"x-anthropic-billing-header:"#,
            r#" cc_version=2.1.86.123; cc_entrypoint=cli; cch=00000;"},{"type"#,
            r#":"text","text":"You are Claude Code","cache_control":{"type":"e"#,
            r#"phemeral"}}],"model":"claude-sonnet-4-20250514","messages":[{"r"#,
            r#"ole":"user","content":[{"type":"text","text":"hello"}]}],"max_t"#,
            r#"okens":8192,"stream":false,"metadata":{"user_id":"agsh"}}"#,
        );

        let digest = xxh64(body.as_bytes(), CCH_XXH64_SEED);
        let token = format!("{:05x}", digest & 0xfffff);

        let patched = patch_request_body(body).unwrap();
        assert!(patched.contains(&format!("cch={}", token)));
        assert!(!patched.contains("cch=00000"));
    }

    #[test]
    fn test_claude_no_system_prompt_when_empty() {
        let provider = test_provider();

        let body = provider.build_request_body("", &[], &[], false);
        assert!(body.get("system").is_none());
    }

    #[test]
    fn test_claude_parse_missing_tool_use_id() {
        let provider = test_provider();

        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "name": "read_file",
                "input": {"path": "/tmp/test.txt"}
            }],
            "stop_reason": "tool_use"
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_claude_parse_missing_tool_use_name() {
        let provider = test_provider();

        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_abc",
                "input": {"path": "/tmp/test.txt"}
            }],
            "stop_reason": "tool_use"
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_fingerprint_boundary_length_messages() {
        let fp5 = compute_fingerprint("abcde", CC_VERSION);
        assert_eq!(fp5.len(), 3);

        let fp8 = compute_fingerprint("abcdefgh", CC_VERSION);
        assert_eq!(fp8.len(), 3);

        let fp21 = compute_fingerprint("abcdefghijklmnopqrstu", CC_VERSION);
        assert_eq!(fp21.len(), 3);

        assert_ne!(fp5, fp8);
        assert_ne!(fp8, fp21);
    }

    #[test]
    fn test_fingerprint_short_message_all_fallback() {
        let fp_short = compute_fingerprint("abc", CC_VERSION);
        let fp_empty = compute_fingerprint("", CC_VERSION);
        assert_eq!(fp_short, fp_empty);
    }

    #[test]
    fn test_fingerprint_multibyte_chars() {
        let msg = "日本語のテスト文字列を使ったメッセージです！！！";
        assert!(msg.chars().count() > 20);
        let fp = compute_fingerprint(msg, CC_VERSION);
        assert_eq!(fp.len(), 3);
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));

        assert_eq!(msg.chars().nth(4), Some('テ'));
        assert_eq!(msg.chars().nth(7), Some('文'));
        assert_eq!(msg.chars().nth(20), Some('す'));
    }

    #[test]
    fn test_fingerprint_different_version() {
        let fp_a = compute_fingerprint("hello", "1.0.0");
        let fp_b = compute_fingerprint("hello", "2.0.0");
        assert_eq!(fp_a.len(), 3);
        assert_eq!(fp_b.len(), 3);
        assert_ne!(fp_a, fp_b);
    }

    #[test]
    fn test_extract_first_user_message_text_no_text_block() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "result".to_string(),
                }],
                is_error: false,
            }],
        }];
        assert_eq!(extract_first_user_message_text(&messages), "");
    }

    #[test]
    fn test_extract_first_user_message_text_multiple_users() {
        let messages = vec![
            Message::user("first user message"),
            Message::user("second user message"),
        ];
        assert_eq!(
            extract_first_user_message_text(&messages),
            "first user message"
        );
    }

    #[test]
    fn test_extract_first_user_message_text_only_assistants() {
        let messages = vec![
            Message::assistant_text("hello"),
            Message::assistant_text("world"),
        ];
        assert_eq!(extract_first_user_message_text(&messages), "");
    }

    #[test]
    fn test_compute_fingerprint_from_messages_empty() {
        let empty: Vec<Message> = vec![];
        assert_eq!(
            compute_fingerprint_from_messages(&empty),
            compute_fingerprint("", CC_VERSION)
        );
    }

    #[test]
    fn test_compute_fingerprint_from_messages_no_user() {
        let messages = vec![Message::assistant_text("I'm an assistant")];
        assert_eq!(
            compute_fingerprint_from_messages(&messages),
            compute_fingerprint("", CC_VERSION)
        );
    }

    // All xxHash64 expected values cross-validated against Python xxhash.

    #[test]
    fn test_xxh64_one_byte() {
        assert_eq!(xxh64(b"x", 0), 0x5c80c09683041123);
    }

    #[test]
    fn test_xxh64_three_bytes() {
        assert_eq!(xxh64(b"abc", 0), 0x44bc2cf5ad770999);
    }

    #[test]
    fn test_xxh64_four_bytes() {
        assert_eq!(xxh64(b"abcd", 0), 0xde0327b0d25d92cc);
    }

    #[test]
    fn test_xxh64_seven_bytes() {
        assert_eq!(xxh64(b"abcdefg", 0), 0x1860940e2902822d);
    }

    #[test]
    fn test_xxh64_eight_bytes() {
        assert_eq!(xxh64(b"abcdefgh", 0), 0x3ad351775b4634b7);
    }

    #[test]
    fn test_xxh64_sixteen_bytes() {
        assert_eq!(xxh64(b"abcdefghijklmnop", 0), 0x71ce8137ca2dd53d);
    }

    #[test]
    fn test_xxh64_thirty_one_bytes() {
        let input = b"abcdefghijklmnopqrstuvwxyz01234";
        assert_eq!(input.len(), 31);
        assert_eq!(xxh64(input, 0), 0x16058c7b947da137);
    }

    #[test]
    fn test_xxh64_thirty_two_bytes() {
        let input = b"abcdefghijklmnopqrstuvwxyz012345";
        assert_eq!(input.len(), 32);
        assert_eq!(xxh64(input, 0), 0xbf2cd639b4143b80);
    }

    #[test]
    fn test_xxh64_with_nonzero_seed() {
        let input = b"hello world";
        let h0 = xxh64(input, 0);
        let h1 = xxh64(input, 1);
        let h_claude = xxh64(input, CCH_XXH64_SEED);
        assert_ne!(h0, h1);
        assert_ne!(h0, h_claude);
        assert_ne!(h1, h_claude);
    }

    #[test]
    fn test_patch_request_body_missing_billing_header() {
        let body = r#"{"system":[],"messages":[]}"#;
        let result = patch_request_body(body);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("x-anthropic-billing-header not found"));
    }

    #[test]
    fn test_patch_request_body_billing_header_without_placeholder() {
        let body = r#"{"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.86.abc; cc_entrypoint=cli;"}]}"#;
        let result = patch_request_body(body);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("cch=00000"));
    }

    #[test]
    fn test_patch_request_body_preserves_length() {
        let provider = test_provider();
        let body = provider.build_request_body("prompt", &[Message::user("hi")], &[], false);
        let body_json = serde_json::to_string(&body).unwrap();
        let patched = patch_request_body(&body_json).unwrap();
        assert_eq!(body_json.len(), patched.len());
    }

    #[test]
    fn test_patch_request_body_cch_in_tool_result() {
        let messages = vec![
            Message::user("run the tool"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "echo cch=00000"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "output: cch=00000".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let provider = test_provider();
        let body = provider.build_request_body("system prompt", &messages, &[], false);
        let body_json = serde_json::to_string(&body).unwrap();
        assert!(body_json.matches("cch=00000").count() >= 2);

        let patched = patch_request_body(&body_json).unwrap();

        let billing_start = patched.find("x-anthropic-billing-header:").unwrap();
        let billing_end = patched[billing_start..].find(';').unwrap() + billing_start;
        let billing_region = &patched[billing_start..billing_end + 30];
        assert!(!billing_region.contains("cch=00000"));
        assert!(patched.contains("output: cch=00000"));
    }

    #[test]
    fn test_claude_request_body_stream_true() {
        let provider = test_provider();
        let body = provider.build_request_body("prompt", &[Message::user("hi")], &[], true);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_claude_request_body_system_and_tools_together() {
        let provider = test_provider();
        let tools = vec![ToolDefinition::new(
            "bash".to_string(),
            "Run a shell command".to_string(),
            serde_json::json!({"type": "object", "properties": {}}),
        )];
        let body =
            provider.build_request_body("system prompt", &[Message::user("hi")], &tools, true);

        assert!(body.get("system").is_some());
        assert!(body.get("tools").is_some());
        assert_eq!(body["stream"], true);

        let json = serde_json::to_string(&body).unwrap();
        assert!(json.find("\"system\"").unwrap() < json.find("\"messages\"").unwrap());

        let tools_array = body["tools"].as_array().unwrap();
        assert!(tools_array.last().unwrap().get("cache_control").is_some());
    }

    #[test]
    fn test_claude_request_body_metadata_fields() {
        let provider = test_provider();
        let body = provider.build_request_body("prompt", &[Message::user("hi")], &[], false);

        let user_id_str = body["metadata"]["user_id"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(user_id_str).unwrap();

        assert_eq!(parsed["device_id"], "agsh");
        assert_eq!(parsed["account_uuid"], "");
        let session_id = parsed["session_id"].as_str().unwrap();
        assert!(Uuid::parse_str(session_id).is_ok(), "{}", session_id);
    }

    #[test]
    fn test_claude_request_body_no_tools_key_when_empty() {
        let provider = test_provider();
        let body = provider.build_request_body("prompt", &[Message::user("hi")], &[], false);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn test_claude_parse_missing_content_array() {
        let provider = test_provider();
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn"
        });
        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_claude_parse_missing_stop_reason_defaults_to_end_turn() {
        let provider = test_provider();
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}]
        });
        let (_, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_claude_parse_max_tokens_stop_reason() {
        let provider = test_provider();
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "truncated"}],
            "stop_reason": "max_tokens"
        });
        let (_, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");
        assert_eq!(stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn test_claude_parse_unknown_stop_reason() {
        let provider = test_provider();
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "something_new"
        });
        let (_, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");
        assert_eq!(
            stop_reason,
            StopReason::Unknown("something_new".to_string())
        );
    }

    #[test]
    fn test_claude_parse_empty_content_array() {
        let provider = test_provider();
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": "end_turn"
        });
        let (message, _, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");
        assert!(message.content.is_empty());
        assert_eq!(message.text_content(), "");
    }

    #[test]
    fn test_claude_parse_thinking_block() {
        let provider = test_provider();
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm..."},
                {"type": "text", "text": "answer"}
            ],
            "stop_reason": "end_turn"
        });
        let (message, _, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");
        assert_eq!(message.content.len(), 2);
        assert!(
            matches!(&message.content[0], ContentBlock::Thinking { thinking, .. } if thinking == "hmm...")
        );
        assert_eq!(message.text_content(), "answer");
    }

    #[test]
    fn test_claude_parse_unknown_block_type_skipped() {
        let provider = test_provider();
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "totally_unknown", "data": "xyz"},
                {"type": "text", "text": "answer"}
            ],
            "stop_reason": "end_turn"
        });
        let (message, _, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.text_content(), "answer");
    }

    #[test]
    fn test_claude_parse_tool_use_missing_input_defaults() {
        let provider = test_provider();
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "list_files"
            }],
            "stop_reason": "tool_use"
        });
        let (message, _, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");
        if let ContentBlock::ToolUse { input, .. } = &message.content[0] {
            assert_eq!(*input, serde_json::json!({}));
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn test_parse_claude_stop_reason_all_variants() {
        assert_eq!(parse_claude_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(parse_claude_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(
            parse_claude_stop_reason("max_tokens"),
            StopReason::MaxTokens
        );
        assert_eq!(
            parse_claude_stop_reason("something_else"),
            StopReason::Unknown("something_else".to_string())
        );
    }

    #[test]
    fn test_claude_user_agent_format() {
        let ua = claude_user_agent();
        assert!(ua.starts_with("claude-cli/"));
        assert!(ua.contains(CC_VERSION));
        assert!(ua.ends_with("(external, cli)"));
    }

    #[test]
    fn test_generate_billing_header_format() {
        let header = generate_billing_header();
        assert!(header.starts_with("x-anthropic-billing-header:"));
        assert!(header.contains(&format!("cc_version={}", CC_VERSION)));
        assert!(header.contains("cc_entrypoint=cli"));
        assert!(header.contains("cch=00000"));
        assert!(header.ends_with("cch=00000;"));
    }

    #[test]
    fn test_stainless_arch_returns_nonempty() {
        assert!(!stainless_arch().is_empty());
    }

    #[test]
    fn test_now_epoch_millis_reasonable() {
        let ms = now_epoch_millis();
        assert!(ms > 1_577_836_800_000);
        assert!(ms < 4_102_444_800_000);
    }

    // ---- Cache prefix stability tests ----
    //
    // These tests simulate multi-turn conversations and tool-use loops to
    // verify that the serialized request bodies share a stable prefix across
    // successive API calls, which is the fundamental requirement for KV cache
    // reuse. A "prefix" here means: the system prompt, tool schemas, and all
    // previously-sent messages must serialize identically (ignoring the
    // `cache_control` marker, which intentionally moves to the newest tail).

    /// Strips every `cache_control` key from every content block in a message
    /// so two messages can be compared purely on semantic content.
    fn strip_cache_control(message: &serde_json::Value) -> serde_json::Value {
        let mut message = message.clone();
        if let Some(content) = message.get_mut("content").and_then(|c| c.as_array_mut()) {
            for block in content.iter_mut() {
                if let Some(obj) = block.as_object_mut() {
                    obj.remove("cache_control");
                }
            }
        }
        message
    }

    /// Strips `cache_control` from every tool schema in an array.
    fn strip_tool_cache_control(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|tool| {
                let mut tool = tool.clone();
                if let Some(obj) = tool.as_object_mut() {
                    obj.remove("cache_control");
                }
                tool
            })
            .collect()
    }

    /// Asserts that the first `shared_count` messages in two request bodies
    /// are semantically identical (ignoring `cache_control` movement), and
    /// that the system prompt and tool schemas are identical.
    fn assert_prefix_stable(
        body_a: &serde_json::Value,
        body_b: &serde_json::Value,
        shared_message_count: usize,
    ) {
        // System prompt must be byte-identical (before cch patching).
        assert_eq!(
            body_a["system"], body_b["system"],
            "system prompt diverged between requests"
        );

        // Tool schemas must be identical (content-wise, ignoring cache_control
        // which is always on the last tool and doesn't affect tokens).
        let tools_a = body_a["tools"].as_array();
        let tools_b = body_b["tools"].as_array();
        match (tools_a, tools_b) {
            (Some(a), Some(b)) => {
                assert_eq!(
                    strip_tool_cache_control(a),
                    strip_tool_cache_control(b),
                    "tool schemas diverged between requests"
                );
            }
            (None, None) => {}
            _ => panic!("tools presence diverged between requests"),
        }

        let msgs_a = body_a["messages"]
            .as_array()
            .expect("messages array in body_a");
        let msgs_b = body_b["messages"]
            .as_array()
            .expect("messages array in body_b");

        assert!(
            msgs_a.len() >= shared_message_count,
            "body_a has {} messages, expected at least {}",
            msgs_a.len(),
            shared_message_count
        );
        assert!(
            msgs_b.len() >= shared_message_count,
            "body_b has {} messages, expected at least {}",
            msgs_b.len(),
            shared_message_count
        );

        for i in 0..shared_message_count {
            let a = strip_cache_control(&msgs_a[i]);
            let b = strip_cache_control(&msgs_b[i]);
            assert_eq!(a, b, "message at index {} diverged between requests", i);
        }
    }

    /// Counts the total number of `cache_control` markers across all content
    /// blocks in the messages array.
    fn count_message_cache_controls(body: &serde_json::Value) -> usize {
        let mut count = 0;
        if let Some(messages) = body["messages"].as_array() {
            for message in messages {
                if let Some(content) = message["content"].as_array() {
                    for block in content {
                        if block.get("cache_control").is_some() {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    fn test_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "read_file".to_string(),
                "Read a file".to_string(),
                serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            ),
            ToolDefinition::new(
                "execute_command".to_string(),
                "Run a shell command".to_string(),
                serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            ),
        ]
    }

    #[test]
    fn test_multi_turn_prefix_is_stable() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        // Turn 1: single user message
        let messages_t1 = vec![Message::user("What files are in /tmp?")];
        let body_t1 = provider.build_request_body(system, &messages_t1, &tools, true);

        // Turn 2: previous turn + assistant response + new user message
        let messages_t2 = vec![
            Message::user("What files are in /tmp?"),
            Message::assistant_text("There are 3 files in /tmp."),
            Message::user("Show me the first one."),
        ];
        let body_t2 = provider.build_request_body(system, &messages_t2, &tools, true);

        // Turn 3: previous turns + another exchange
        let messages_t3 = vec![
            Message::user("What files are in /tmp?"),
            Message::assistant_text("There are 3 files in /tmp."),
            Message::user("Show me the first one."),
            Message::assistant_text("Here is the content of file1.txt."),
            Message::user("Delete it."),
        ];
        let body_t3 = provider.build_request_body(system, &messages_t3, &tools, true);

        // The first message is shared across all three requests.
        assert_prefix_stable(&body_t1, &body_t2, 1);
        // The first three messages are shared between turn 2 and turn 3.
        assert_prefix_stable(&body_t2, &body_t3, 3);
        // The first message is shared across turn 1 and turn 3.
        assert_prefix_stable(&body_t1, &body_t3, 1);
    }

    /// Simulates a two-turn conversation where the user toggles the
    /// permission level between turns and verifies that the cacheable
    /// prefix (system prompt + tools array + historical messages) is
    /// byte-identical across the toggle. This is the regression guard for
    /// Option 1 of the higher-permission-visibility work — it proves that
    /// `/permission <level>` mid-session does not invalidate the Anthropic
    /// prompt cache.
    ///
    /// Covers the full agent request-body assembly:
    ///   - [`ToolRegistry::tool_catalogue`] / [`ToolRegistry::definitions_active`]
    ///   - [`crate::context::build_system_prompt`]
    ///   - [`crate::context::build_turn_context`]
    ///   - [`ClaudeProvider::build_request_body`]
    #[tokio::test]
    async fn test_permission_toggle_preserves_cache_prefix() {
        use std::path::Path;

        use crate::context::{build_system_prompt, build_turn_context};
        use crate::permission::{Permission, SharedPermission};
        use crate::session::SessionManager;
        use crate::tools::ToolRegistry;

        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("in-memory session manager");
        let shared_permission = SharedPermission::new(Permission::Read);
        let shared_session_id = std::sync::Arc::new(tokio::sync::RwLock::new(None));
        let todo_list = std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let registry = ToolRegistry::build_default(
            crate::config::WebClientConfig::default(),
            shared_permission,
            true,
            crate::sandbox::detect(),
            todo_list,
            session_manager,
            shared_session_id,
            crate::tools::BuiltinToolFilter::default(),
        )
        .expect("default web client config should build cleanly");

        let provider = test_provider();

        // ---- Turn 1 @ Read ---------------------------------------------
        // The agent fetches these once per turn. None of them take the
        // current permission — that's the invariant we're testing.
        let catalogue = registry.tool_catalogue();
        let system = build_system_prompt(&catalogue, true, &[], None, &[]);
        let tools = registry.definitions_active();

        let u1_text = {
            let block = build_turn_context(Permission::Read, &[]);
            format!("{}\n\n{}", block, "list files under /tmp")
        };
        let messages_t1 = vec![Message::user(&u1_text)];
        let body_t1 = provider.build_request_body(&system, &messages_t1, &tools, true);

        // ---- /permission write toggle ---------------------------------
        // (In real code this happens on a different thread via
        // `SharedPermission::set`. Here we just re-read the catalogue and
        // rebuild everything to prove the outputs don't depend on it.)

        // ---- Turn 2 @ Write -------------------------------------------
        let catalogue_t2 = registry.tool_catalogue();
        let system_t2 = build_system_prompt(&catalogue_t2, true, &[], None, &[]);
        let tools_t2 = registry.definitions_active();

        let u2_text = {
            let block = build_turn_context(Permission::Write, &[]);
            format!("{}\n\n{}", block, "now write 'hi' to /tmp/out.txt")
        };
        let messages_t2 = vec![
            Message::user(&u1_text),
            Message::assistant_text("There are three files in /tmp."),
            Message::user(&u2_text),
        ];
        let body_t2 = provider.build_request_body(&system_t2, &messages_t2, &tools_t2, true);

        // 1. The system prompt is identical. (Breakpoint 2 cache-hit.)
        assert_eq!(
            body_t1["system"], body_t2["system"],
            "system prompt diverged across /permission toggle — cache prefix invalidated"
        );

        // 2. The tools array is identical. (Breakpoint 3 cache-hit.)
        //    Reuse the existing helper which tolerates cache_control
        //    movement between the last-tool position across requests.
        assert_prefix_stable(&body_t1, &body_t2, 1);

        // 3. The turn-1 user message is preserved verbatim in turn-2's
        //    history — historical messages must never mutate on toggle,
        //    otherwise breakpoint 4 (messages cache) cascades.
        let t1_msg = strip_cache_control(&body_t1["messages"][0]);
        let t2_msg0 = strip_cache_control(&body_t2["messages"][0]);
        assert_eq!(
            t1_msg, t2_msg0,
            "turn-1 user message changed after permission toggle"
        );

        // 4. Sanity: the two user messages do differ in their permission
        //    context (fresh content on each turn, not cached yet).
        assert!(u1_text.contains("Current permission level: read"));
        assert!(u2_text.contains("Current permission level: write"));
        assert_ne!(u1_text, u2_text);
    }

    /// Same invariant, but exercises every pairwise permission toggle
    /// (16 combinations). Catches any permission state that sneaks back
    /// into the cacheable prefix.
    #[tokio::test]
    async fn test_permission_independence_all_levels() {
        use std::path::Path;

        use crate::context::build_system_prompt;
        use crate::permission::{Permission, SharedPermission};
        use crate::session::SessionManager;
        use crate::tools::ToolRegistry;

        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("in-memory session manager");
        let shared_permission = SharedPermission::new(Permission::Read);
        let shared_session_id = std::sync::Arc::new(tokio::sync::RwLock::new(None));
        let todo_list = std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let registry = ToolRegistry::build_default(
            crate::config::WebClientConfig::default(),
            shared_permission.clone(),
            true,
            crate::sandbox::detect(),
            todo_list,
            session_manager,
            shared_session_id,
            crate::tools::BuiltinToolFilter::default(),
        )
        .expect("default web client config should build cleanly");

        let provider = test_provider();
        let levels = [
            Permission::None,
            Permission::Read,
            Permission::Ask,
            Permission::Write,
        ];

        let mut bodies = Vec::with_capacity(levels.len());
        for &level in &levels {
            shared_permission.set(level);
            let catalogue = registry.tool_catalogue();
            let system = build_system_prompt(&catalogue, true, &[], None, &[]);
            let tools = registry.definitions_active();
            let messages = vec![Message::user("hello")];
            bodies.push(provider.build_request_body(&system, &messages, &tools, true));
        }

        // Every pair must agree on the cacheable prefix.
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                assert_eq!(
                    bodies[i]["system"], bodies[j]["system"],
                    "system prompt differs between {:?} and {:?}",
                    levels[i], levels[j]
                );
                assert_prefix_stable(&bodies[i], &bodies[j], 1);
            }
        }
    }

    #[test]
    fn test_tool_loop_prefix_is_stable() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        // Iteration 1 of tool loop: user asks, model about to respond
        let messages_iter1 = vec![Message::user("Read /tmp/test.txt")];
        let body_iter1 = provider.build_request_body(system, &messages_iter1, &tools, true);

        // Iteration 2: model made a tool call, tool result came back
        let messages_iter2 = vec![
            Message::user("Read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "hello world".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body_iter2 = provider.build_request_body(system, &messages_iter2, &tools, true);

        // Iteration 3: model made another tool call
        let messages_iter3 = vec![
            Message::user("Read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "hello world".to_string(),
                    }],
                    is_error: false,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_2".to_string(),
                    name: "execute_command".to_string(),
                    input: serde_json::json!({"command": "wc -l /tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_2".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "1 /tmp/test.txt".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body_iter3 = provider.build_request_body(system, &messages_iter3, &tools, true);

        // Prefix from iter1 is stable in iter2 and iter3
        assert_prefix_stable(&body_iter1, &body_iter2, 1);
        assert_prefix_stable(&body_iter2, &body_iter3, 3);
        assert_prefix_stable(&body_iter1, &body_iter3, 1);
    }

    #[test]
    fn test_exactly_one_message_cache_control_per_request() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        // Single message
        let body1 = provider.build_request_body(system, &[Message::user("hello")], &tools, true);
        assert_eq!(count_message_cache_controls(&body1), 1);

        // Three messages
        let body3 = provider.build_request_body(
            system,
            &[
                Message::user("hello"),
                Message::assistant_text("hi"),
                Message::user("bye"),
            ],
            &tools,
            true,
        );
        assert_eq!(count_message_cache_controls(&body3), 1);

        // Five messages with tool use
        let body5 = provider.build_request_body(
            system,
            &[
                Message::user("read file"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "t1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "/tmp/x"}),
                    }],
                },
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "t1".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "data".to_string(),
                        }],
                        is_error: false,
                    }],
                },
                Message::assistant_text("Here's the file."),
                Message::user("thanks"),
            ],
            &tools,
            true,
        );
        assert_eq!(count_message_cache_controls(&body5), 1);
    }

    #[test]
    fn test_cache_control_shifts_to_new_last_message() {
        let provider = test_provider();
        let system = "system";

        // Build with 2 messages — cache_control should be on message[1]
        let messages_a = vec![Message::user("hello"), Message::assistant_text("hi")];
        let body_a = provider.build_request_body(system, &messages_a, &[], false);
        let msgs_a = body_a["messages"].as_array().unwrap();

        // Message 0 should NOT have cache_control
        let block_0 = &msgs_a[0]["content"].as_array().unwrap()[0];
        assert!(block_0.get("cache_control").is_none());
        // Message 1 (last) SHOULD have cache_control
        let block_1 = &msgs_a[1]["content"].as_array().unwrap()[0];
        assert!(block_1.get("cache_control").is_some());

        // Now append a third message — cache_control should move to message[2]
        let messages_b = vec![
            Message::user("hello"),
            Message::assistant_text("hi"),
            Message::user("bye"),
        ];
        let body_b = provider.build_request_body(system, &messages_b, &[], false);
        let msgs_b = body_b["messages"].as_array().unwrap();

        // Messages 0 and 1 should NOT have cache_control
        assert!(
            msgs_b[0]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            msgs_b[1]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        // Message 2 (new last) SHOULD have cache_control
        assert!(
            msgs_b[2]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_some()
        );
    }

    #[test]
    fn test_system_prompt_identical_across_turns() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        let body1 = provider.build_request_body(system, &[Message::user("turn 1")], &tools, true);
        let body2 = provider.build_request_body(
            system,
            &[
                Message::user("turn 1"),
                Message::assistant_text("response 1"),
                Message::user("turn 2"),
            ],
            &tools,
            true,
        );
        let body3 = provider.build_request_body(
            system,
            &[
                Message::user("turn 1"),
                Message::assistant_text("response 1"),
                Message::user("turn 2"),
                Message::assistant_text("response 2"),
                Message::user("turn 3"),
            ],
            &tools,
            true,
        );

        // System prompt must be byte-identical across all turns.
        assert_eq!(body1["system"], body2["system"]);
        assert_eq!(body2["system"], body3["system"]);

        // Model, max_tokens, metadata must also be identical.
        assert_eq!(body1["model"], body2["model"]);
        assert_eq!(body1["max_tokens"], body2["max_tokens"]);
        assert_eq!(body1["metadata"], body2["metadata"]);
        assert_eq!(body2["model"], body3["model"]);
        assert_eq!(body2["max_tokens"], body3["max_tokens"]);
        assert_eq!(body2["metadata"], body3["metadata"]);
    }

    #[test]
    fn test_tool_schemas_stable_across_turns() {
        let provider = test_provider();
        let tools = test_tools();

        let body1 = provider.build_request_body("system", &[Message::user("a")], &tools, true);
        let body2 = provider.build_request_body(
            "system",
            &[
                Message::user("a"),
                Message::assistant_text("b"),
                Message::user("c"),
            ],
            &tools,
            true,
        );

        // Tool schemas (including cache_control on the last tool) must be
        // identical when the same tools are provided.
        assert_eq!(body1["tools"], body2["tools"]);
    }

    #[test]
    fn test_multi_block_message_cache_control_on_last_block_only() {
        let provider = test_provider();

        // An assistant message with text + tool_use (multiple blocks)
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Let me read that file.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                },
            ],
        }];
        let body = provider.build_request_body("system", &messages, &[], false);
        let msg = &body["messages"].as_array().unwrap()[0];
        let blocks = msg["content"].as_array().unwrap();

        // First block (text) should NOT have cache_control
        assert!(blocks[0].get("cache_control").is_none());
        // Second block (tool_use, the last block of the last message) SHOULD
        assert!(blocks[1].get("cache_control").is_some());
    }

    #[test]
    fn test_long_conversation_prefix_stability() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        // Build up a 10-turn conversation incrementally and verify each step
        // preserves the prefix from the previous step.
        let mut messages: Vec<Message> = Vec::new();
        let mut previous: Option<(serde_json::Value, usize)> = None;

        for turn in 0..10 {
            messages.push(Message::user(format!("User message {}", turn)));
            let body = provider.build_request_body(system, &messages, &tools, true);

            if let Some((prev_body, prev_msg_count)) = &previous {
                // The shared prefix is exactly the messages that were in the
                // previous request body.
                assert_prefix_stable(prev_body, &body, *prev_msg_count);
            }

            assert_eq!(count_message_cache_controls(&body), 1);

            let msg_count = messages.len();
            // Simulate assistant response
            messages.push(Message::assistant_text(format!("Response {}", turn)));
            previous = Some((body, msg_count));
        }
    }

    #[test]
    fn test_tool_loop_with_multiple_sequential_calls() {
        let provider = test_provider();
        let system = "system";
        let tools = test_tools();

        // Simulate a user request that triggers 4 sequential tool calls.
        // Each iteration of the loop adds an assistant tool_use + user
        // tool_result pair. Verify the prefix is stable across all iterations.
        let mut messages: Vec<Message> = vec![Message::user("do several things")];

        let mut previous_body: Option<serde_json::Value> = None;
        let mut previous_len = 0;

        for i in 0..4 {
            let body = provider.build_request_body(system, &messages, &tools, true);

            if let Some(prev) = &previous_body {
                assert_prefix_stable(prev, &body, previous_len);
            }

            assert_eq!(
                count_message_cache_controls(&body),
                1,
                "iteration {} should have exactly 1 message cache_control",
                i
            );

            previous_len = messages.len();
            previous_body = Some(body);

            // Simulate tool call and result
            messages.push(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: format!("toolu_{}", i),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": format!("/tmp/file{}", i)}),
                }],
            });
            messages.push(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: format!("toolu_{}", i),
                    content: vec![ToolResultContent::Text {
                        text: format!("contents of file{}", i),
                    }],
                    is_error: false,
                }],
            });
        }

        // Final body after all tool calls
        let final_body = provider.build_request_body(system, &messages, &tools, true);
        assert_prefix_stable(previous_body.as_ref().unwrap(), &final_body, previous_len);
        assert_eq!(count_message_cache_controls(&final_body), 1);
    }

    #[test]
    fn test_empty_messages_produces_no_cache_control() {
        let provider = test_provider();
        let body = provider.build_request_body("system", &[], &[], false);
        assert_eq!(count_message_cache_controls(&body), 0);
        assert!(body["messages"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_cache_control_on_tool_result_block() {
        let provider = test_provider();

        // When the last message is a tool_result, cache_control should still
        // appear on its last content block.
        let messages = vec![
            Message::user("read file"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "file data".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body = provider.build_request_body("system", &messages, &[], false);
        let msgs = body["messages"].as_array().unwrap();

        // Only the tool_result message (last) should have cache_control
        assert!(
            msgs[0]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            msgs[1]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            msgs[2]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_some()
        );
        assert_eq!(count_message_cache_controls(&body), 1);
    }

    #[test]
    fn test_claude_cache_control_on_last_message_only() {
        let provider = test_provider();

        let messages = vec![
            Message::user("first"),
            Message::assistant_text("response"),
            Message::user("second"),
        ];
        let body = provider.build_request_body("system", &messages, &[], false);
        let claude_messages = body["messages"].as_array().unwrap();

        let first_content = claude_messages[0]["content"].as_array().unwrap();
        assert!(first_content[0].get("cache_control").is_none());

        let second_content = claude_messages[1]["content"].as_array().unwrap();
        assert!(second_content[0].get("cache_control").is_none());

        let third_content = claude_messages[2]["content"].as_array().unwrap();
        assert!(third_content[0].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_cache_control_on_last_tool() {
        let provider = test_provider();

        let tools = vec![
            ToolDefinition::new(
                "read_file".to_string(),
                "Read a file".to_string(),
                serde_json::json!({"type": "object"}),
            ),
            ToolDefinition::new(
                "write_file".to_string(),
                "Write a file".to_string(),
                serde_json::json!({"type": "object"}),
            ),
        ];
        let body = provider.build_request_body("system", &[Message::user("hi")], &tools, false);
        let claude_tools = body["tools"].as_array().unwrap();

        assert!(claude_tools[0].get("cache_control").is_none());
        assert!(claude_tools[1].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_no_message_cache_control_when_empty() {
        let provider = test_provider();
        let body = provider.build_request_body("system", &[], &[], false);
        let claude_messages = body["messages"].as_array().unwrap();
        assert!(claude_messages.is_empty());
    }
}
