//! Session and account status as the REPL and the CLI show them, and the hint printed when a
//! session has no usable provider.

use super::*;
use crate::streams::{write_stderr, write_stderr_line};

pub(crate) fn render_session_id(label: &str, id: &str) {
    write_stderr_line(format!("{label}: {id}").with(Color::DarkGrey));
}
/// The per-turn token-usage line, `[in 12.3k / cache hit 96% / out 1.2k]`, dimmed and preceded by
/// a blank line. `in` is the total of all three input-token tiers (live, cache-write, cache-read),
/// and the hit rate is `cache_read / in`.
pub(crate) fn render_token_usage(usage: &crate::stats::TokenUsage) {
    let total_in = usage
        .input_tokens
        .saturating_add(usage.cache_creation_input_tokens)
        .saturating_add(usage.cache_read_input_tokens);
    let cache_hit_pct = if total_in == 0 {
        0
    } else {
        ((usage.cache_read_input_tokens as f64) / (total_in as f64) * 100.0).round() as u64
    };
    write_stderr_line("");
    write_stderr_line(
        format!(
            "[in {} / cache hit {}% / out {}]",
            format_token_count(total_in),
            cache_hit_pct,
            format_token_count(usage.output_tokens),
        )
        .with(Color::DarkGrey),
    );
}
/// The resolved model parameters shown at the top of the `/status` report, borrowed from the active
/// config plus the provider's settled effort. All optional so a mis-selected profile still renders.
pub(crate) struct ModelStatus<'a> {
    pub(crate) model: Option<&'a str>,
    /// Active profile name (e.g. `claude-max`).
    pub(crate) profile: Option<&'a str>,
    /// The account the profile bills.
    pub(crate) account: Option<&'a str>,
    /// The backend the profile's account names.
    pub(crate) backend: Option<crate::config::Backend>,
    /// The reasoning effort sent on the wire, or `None` when the request sends none.
    pub(crate) effort: Option<&'a str>,
    pub(crate) thinking: crate::config::ThinkingMode,
}
/// The body of the session-status block, without ANSI and without the header line.
///
/// Split out because every non-REPL frontend needs the same numbers in a different envelope; with
/// no shared body each re-implements the formatting and they drift. Pairs with
/// [`render_session_status`], which is this plus the colored header, printed. The same shape as
/// [`format_account_usage`] / [`render_account_usage`], for the same reason.
pub(crate) fn format_session_status(
    snap: &crate::stats::SessionStatsSnapshot,
    model: &ModelStatus,
    message_count: usize,
    context_tokens: u64,
    context_window: u64,
) -> String {
    let total_in = snap.total_input_tokens();
    let mut out = String::new();
    // Ordered like the profile these lines are resolved from: the account and its backend first,
    // then the model, then the model-tied knobs in the order `[profiles.<name>]` declares them
    // (`context_window`, `effort`, `thinking`), so the block reads beside the config it came from.
    // The cumulative counters follow, and answer a different question.
    if let Some(profile) = model.profile {
        out.push_str(&format!("  Profile:         {profile}\n"));
    }
    // The backend rides with the account rather than the profile because it is the account's
    // fact: two profiles on one account state it the same.
    // Both come off one settings lookup, so they are present together or absent together.
    if let (Some(account), Some(backend)) = (model.account, model.backend) {
        out.push_str(&format!("  Account:         {account} ({backend})\n"));
    }
    if let Some(name) = model.model {
        out.push_str(&format!("  Model:           {name}\n"));
    }
    // Live context occupancy: how full the window was on the last request. Distinct from the
    // cumulative "Input tokens" total below, which sums every turn's usage for the whole session.
    //
    // Shown from turn zero, at `0 / <window>`, rather than waiting for occupancy to be non-zero.
    // The window is the profile's `context_window` or a documented default, and meka neither probes
    // for it nor checks it against the model, which makes this the only place a user can confirm
    // the number their session budgets against. Getting it wrong is otherwise invisible until
    // compaction misbehaves several turns in.
    if context_window > 0 {
        let pct = ((context_tokens as f64 / context_window as f64) * 100.0).round() as u64;
        let remaining = context_window.saturating_sub(context_tokens);
        out.push_str(&format!(
            "  Context:         {} / {} ({}% used, {} left)\n",
            format_token_count(context_tokens),
            format_token_count(context_window),
            pct,
            format_token_count(remaining)
        ));
    }
    if let Some(effort) = model.effort {
        out.push_str(&format!("  Effort:          {effort}\n"));
    }
    // Anthropic-only, and omitted elsewhere for the same reason `Effort` is omitted when unset: a
    // status block should report what the request carries, and `thinking` is not a field an OpenAI
    // request has. Naming an encoding there would read as a setting that is in force.
    if model
        .backend
        .is_some_and(crate::config::Backend::takes_thinking)
    {
        out.push_str(&format!("  Thinking:        {}\n", model.thinking.name()));
    }
    out.push_str(&format!("  Turns:           {}\n", snap.turns));
    out.push_str(&format!(
        "  Input tokens:    {}  (cache hit: {}%)\n",
        format_token_count(total_in),
        snap.cache_hit_pct()
    ));
    out.push_str(&format!(
        "  Output tokens:   {}\n",
        format_token_count(snap.output_tokens)
    ));
    if snap.redactions > 0 {
        out.push_str(&format!(
            "  Redactions:      {} ({} image{}, ~{} freed)\n",
            snap.redactions,
            snap.redacted_images,
            if snap.redacted_images == 1 { "" } else { "s" },
            crate::text::format_size(usize::try_from(snap.redacted_bytes).unwrap_or(usize::MAX))
        ));
    } else {
        out.push_str("  Redactions:      0\n");
    }
    out.push_str(&format!("  Messages:        {message_count}\n"));
    out
}
/// Print the status under its heading.
pub(crate) fn render_session_status(text: &str) {
    render_heading("Session status");
    write_stderr(text);
}

/// Plain-text (no ANSI) rendering of account rate-limit usage, shared by the REPL/ACP `/usage`
/// command and the `meka account usage` CLI. Kept ANSI-free so the CLI can pipe it into scripts
/// unchanged; the trailing newline lets callers `print!`/`eprint!` it directly.
pub(crate) fn format_account_usage(usage: &crate::provider::AccountUsage) -> String {
    let mut out = String::from("Account usage\n");
    if usage.windows.is_empty() {
        out.push_str("  (no usage windows reported)\n");
    }
    for window in &usage.windows {
        let percent = window.used_percent.clamp(0.0, 100.0);
        let reset = window
            .resets_at
            .map(format_reset_time)
            .map(|when| format!("  (resets {when})"))
            .unwrap_or_default();
        out.push_str(&format!(
            "  {:<18} {} {:>3}% used{}\n",
            window.label,
            usage_bar(percent),
            percent.round() as u64,
            reset
        ));
    }
    if let Some(extra) = &usage.extra_usage
        && let Some(line) = format_extra_usage(extra)
    {
        out.push_str(&format!("  Extra usage: {line}\n"));
    }
    if let Some(note) = &usage.note {
        out.push_str(&format!("  {note}\n"));
    }
    out
}
/// One-line summary of extra-usage / credits state, or `None` when there's nothing worth showing
/// (disabled with no balance and nothing spent).
pub(super) fn format_extra_usage(extra: &crate::provider::ExtraUsage) -> Option<String> {
    let has_data =
        extra.enabled || extra.used.is_some_and(|used| used > 0.0) || extra.balance.is_some();
    if !has_data {
        return None;
    }
    let mut parts = vec![if extra.enabled { "enabled" } else { "disabled" }.to_string()];
    if let Some(utilization) = extra.utilization {
        parts.push(format!("{}% used", utilization.round() as i64));
    }
    if let Some(used) = extra.used {
        parts.push(format!(
            "{} spent",
            format_money(used, extra.currency.as_deref())
        ));
    }
    if let Some(balance) = extra.balance {
        parts.push(format!(
            "{} balance",
            format_money(balance, extra.currency.as_deref())
        ));
    }
    Some(parts.join(" · "))
}
/// Format a monetary amount: `$3.00` for USD/unknown, `3.00 EUR` otherwise.
pub(super) fn format_money(amount: f64, currency: Option<&str>) -> String {
    match currency {
        Some("USD") | None => format!("${amount:.2}"),
        Some(other) => format!("{amount:.2} {other}"),
    }
}
/// REPL `/usage` rendering: the shared plain text to stderr (REPL UI feedback). The "not available"
/// case is handled by the caller via `render_hint`.
pub(crate) fn render_account_usage(usage: &crate::provider::AccountUsage) {
    write_stderr(format_account_usage(usage));
}
/// A fixed-width `[####------]` gauge for a 0-100 percentage.
pub(super) fn usage_bar(percent: f64) -> String {
    const CELLS: usize = 10;
    let filled = ((percent / 100.0) * CELLS as f64).round() as usize;
    let filled = filled.min(CELLS);
    let mut bar = String::with_capacity(CELLS + 2);
    bar.push('[');
    for cell in 0..CELLS {
        bar.push(if cell < filled { '#' } else { '-' });
    }
    bar.push(']');
    bar
}
/// Format a reset instant (Unix seconds) as "relative, local clock", e.g. "in 4h 12m, 2026-07-02
/// 02:10 +02:00". Falls back to a plain clock when the timestamp is in the past or unparseable.
/// Shared with the ACP `/usage` text builder.
pub(crate) fn format_reset_time(epoch_seconds: i64) -> String {
    let Some(when) = chrono::DateTime::from_timestamp(epoch_seconds, 0) else {
        return "unknown".to_string();
    };
    let clock = crate::text::format_timestamp(when, crate::text::Precision::Minutes);
    let minutes = when.signed_duration_since(chrono::Utc::now()).num_minutes();
    if minutes <= 0 {
        return format!("now, {clock}");
    }
    let relative = if minutes >= 24 * 60 {
        format!(
            "in {}d {}h",
            minutes / (24 * 60),
            (minutes % (24 * 60)) / 60
        )
    } else if minutes >= 60 {
        format!("in {}h {}m", minutes / 60, minutes % 60)
    } else {
        format!("in {minutes}m")
    };
    format!("{relative}, {clock}")
}
/// A session whose recorded profile is not one the config has, and a profile it could be
/// moved to, for [`render_profile_setup_hint`].
///
/// The caller establishes both facts before building this. The hint is only right when the row's
/// own profile is what could not be resolved: it would otherwise send a user whose profile is
/// merely missing its credential to repin a session that is bound exactly where it belongs. And
/// there has to be somewhere to move it, so a config with no profiles at all gets the generic
/// example instead of a command with nothing to put in it.
pub(crate) struct MissingSessionProfile<'a> {
    /// The session `--profile` would repin, which is the only thing that can rewrite the binding.
    pub(crate) session_id: uuid::Uuid,
    /// The configured profile to suggest moving to.
    pub(crate) move_to: &'a str,
}
/// The one-line hint under the error printed when the agent fails to build.
///
/// One line, because the error above it has already said everything else. When the session's own
/// profile is what could not be resolved, the session id is the single fact that error cannot
/// reach, and `-r <id> --profile <name>` is the only command that rewrites a row's binding, so
/// that is what this adds. No line suggests recreating the missing profile: meka never saw it, so
/// any account or model it named would be invented.
///
/// `None` says nothing about *why* setup failed: the caller prints the error first, and it is as
/// often a configured profile missing its credential as no profile at all.
pub(crate) fn render_profile_setup_hint(missing: Option<MissingSessionProfile<'_>>) {
    match missing {
        Some(missing) => write_stderr_line(format!(
            "Move this session onto a configured profile: `meka -r {} --profile {}`",
            missing.session_id, missing.move_to
        )),
        None => write_stderr_line("Run `meka profile list` to see the configured profiles."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window is reported from turn zero, before there is any occupancy to divide into it.
    ///
    /// It is not inferred from the model name, so `/status` is the only place a user can check
    /// the number their session budgets against, and a wrong one is invisible until compaction
    /// misbehaves several turns later. Waiting for the first turn to show it means the setting can
    /// only be verified by spending a turn, which is the wrong way round.
    #[test]
    fn the_context_window_is_reported_before_the_first_turn() {
        use crate::config::ThinkingMode;

        let snap = crate::stats::SessionStats::default().snapshot();
        let model = ModelStatus {
            model: Some("some-local-model"),
            profile: Some("local"),
            account: Some("acme"),
            backend: Some(crate::config::Backend::AnthropicMessages),
            effort: None,
            thinking: ThinkingMode::Adaptive,
        };

        // Nothing sent yet: the window still has to appear, at zero occupancy.
        let fresh = format_session_status(&snap, &model, 0, 0, 262_144);
        assert!(fresh.contains("Context:"), "{fresh}");
        assert!(
            fresh.contains("0 / 262.1k"),
            "the configured window: {fresh}"
        );

        // Once a turn has run, the same line carries the occupancy.
        let used = format_session_status(&snap, &model, 2, 65_536, 262_144);
        assert!(used.contains("25% used"), "{used}");

        // An unknown window (sub-agents, tests) still has nothing to report.
        let unknown = format_session_status(&snap, &model, 0, 0, 0);
        assert!(!unknown.contains("Context:"), "{unknown}");
    }

    /// The resolved-profile lines come in the order `[profiles.<name>]` declares the same fields,
    /// so the block and the config it was resolved from can be read side by side.
    #[test]
    fn the_status_block_follows_the_profile_field_order() {
        use crate::config::ThinkingMode;

        let snap = crate::stats::SessionStats::default().snapshot();
        let body = format_session_status(
            &snap,
            &ModelStatus {
                model: Some("some-model"),
                profile: Some("p"),
                account: Some("a"),
                backend: Some(crate::config::Backend::AnthropicMessages),
                effort: Some("high"),
                thinking: ThinkingMode::Adaptive,
            },
            7,
            1_024,
            262_144,
        );

        let labels: Vec<&str> = body
            .lines()
            .filter_map(|line| line.trim_start().split(':').next())
            .collect();
        assert_eq!(
            labels,
            vec![
                // `account`, `model`, `context_window`, `effort`, `thinking`: the profile's own
                // order, for the fields that come from it.
                "Profile",
                "Account",
                "Model",
                "Context",
                "Effort",
                "Thinking",
                // Then what the session has spent, which no profile field describes.
                "Turns",
                "Input tokens",
                "Output tokens",
                "Redactions",
                "Messages",
            ],
            "{body}"
        );
        assert!(
            body.contains("  Profile:         p\n")
                && body.contains("  Account:         a (anthropic-messages)\n"),
            "the backend is the account's fact and sits beside it: {body}"
        );
    }

    /// `/status` reports what the request actually carries, not what meka happens to hold.
    ///
    /// Both of these lines are conditional for the same reason: `effort` is omitted when the
    /// profile sets none, because the provider then picks its own, and `thinking` is omitted on a
    /// backend whose requests have no such field. Printing either unconditionally states a setting
    /// that is not in force - which is exactly what the status block exists to rule out.
    #[test]
    fn the_status_block_omits_settings_the_request_does_not_carry() {
        use crate::config::ThinkingMode;

        let snap = crate::stats::SessionStats::default().snapshot();
        let body = |backend: crate::config::Backend, effort: Option<&'static str>| {
            format_session_status(
                &snap,
                &ModelStatus {
                    model: Some("some-model"),
                    profile: Some("p"),
                    account: None,
                    backend: Some(backend),
                    effort,
                    thinking: ThinkingMode::Adaptive,
                },
                0,
                0,
                0,
            )
        };

        let claude = body(crate::config::Backend::AnthropicMessages, Some("xhigh"));
        assert!(claude.contains("Thinking:"), "{claude}");
        assert!(claude.contains("Effort:"), "{claude}");

        // An OpenAI request has no `thinking` field, whatever mode the struct carries.
        let openai = body(crate::config::Backend::OpenAiChatCompletions, Some("high"));
        assert!(!openai.contains("Thinking:"), "{openai}");

        // Unset effort means the provider's own default, so there is no tier to report.
        let unset = body(crate::config::Backend::AnthropicMessages, None);
        assert!(!unset.contains("Effort:"), "{unset}");
    }

    #[test]
    fn format_account_usage_is_ansi_free() {
        let usage = crate::provider::AccountUsage {
            windows: vec![crate::provider::UsageWindow {
                label: "5-hour (session)".into(),
                used_percent: 23.0,
                resets_at: None,
            }],
            extra_usage: None,
            note: None,
        };
        let out = format_account_usage(&usage);
        assert!(
            !out.contains('\u{1b}'),
            "must be ANSI-free for piping: {out:?}"
        );
        // Disabled/empty extra usage adds no line.
        assert!(!out.contains("Extra usage"), "got: {out:?}");
    }

    #[test]
    fn format_account_usage_shows_enabled_extra_usage() {
        let usage = crate::provider::AccountUsage {
            windows: vec![],
            extra_usage: Some(crate::provider::ExtraUsage {
                enabled: true,
                utilization: Some(70.0),
                used: Some(3.5),
                balance: Some(5.0),
                currency: None,
            }),
            note: None,
        };
        let out = format_account_usage(&usage);
        assert!(
            out.contains("Extra usage: enabled · 70% used · $3.50 spent · $5.00 balance"),
            "got: {out:?}"
        );
    }
}
