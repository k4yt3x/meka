//! Which profile serves a turn: the session's own, or one of its `fallbacks` while that one
//! cannot.
//!
//! A fallback is a substitution for one turn, never a repin. The session's row keeps naming its
//! profile, [`Agent::restore_named_profile`] puts it back when the turn ends, and the turn's row
//! records which profile served it. Both profiles sit in the published cell
//! ([`crate::provider::PublishedProfile`]): a worker spawned mid-turn binds to the named one, and
//! a repin ends the substitution. A profile that failed over is skipped at the start of later
//! turns for a cooldown the registry keeps ([`crate::provider::ProviderRegistry::is_unavailable`]),
//! so an outage is not probed on every turn; when every profile in the chain is skipped the named
//! one is tried anyway, because something has to probe. The breaker decides only where a turn
//! starts: a running turn whose profile fails moves to the next profile in its chain it has not
//! tried, breaker or not, so it fails only once every profile has refused it.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use super::{Agent, recovery::should_retry_provider_error};
use crate::{
    error::{MekaError, Result},
    frontend::{FrontendEvent, Notice},
    provider::{CompletionRequest, ResolvedProfile, resolved_profile, retry::fallback_cooldown},
};

/// Puts the named profile back when a compaction run outside a turn ends, however it ends. Inside
/// a turn it does nothing: the turn restores at its own end, and a compaction's fallback is the
/// turn's to keep.
pub(super) struct RestoreWhenIdle<'a> {
    pub(super) agent: &'a Agent,
}

impl Drop for RestoreWhenIdle<'_> {
    fn drop(&mut self) {
        if self.agent.current_turn().is_none() {
            self.agent.restore_named_profile();
        }
    }
}

impl Agent {
    /// The fallback serving now, when one is: `Some` while the published profile is not the
    /// session's own.
    pub(crate) fn serving_fallback(&self) -> Option<String> {
        self.cells.profile.serving_fallback()
    }

    /// Bind `fallback` for the rest of the turn, keeping the named profile to go back to.
    fn serve_from(&self, fallback: ResolvedProfile) {
        self.cells.profile.serve(&fallback);
    }

    /// Put the session's own profile back and forget which profiles the turn tried, at the end of
    /// whatever a fallback served.
    pub(crate) fn restore_named_profile(&self) {
        self.cells.profile.restore();
        crate::sync::lock(&self.profiles_tried).clear();
    }

    /// The profile the session names, whether or not a fallback is serving in its place.
    fn named_profile_name(&self) -> String {
        self.cells.profile.named().profile
    }

    /// At a turn's start: stay on the named profile, or when its breaker is open bind the first
    /// fallback whose breaker is closed. With every breaker open the named profile is tried
    /// anyway.
    pub(super) async fn bind_serving_profile(&self) {
        let named = self.named_profile_name();
        if !self.providers.is_unavailable(&named) {
            return;
        }
        for fallback in self.providers.fallbacks_of(&named) {
            if self.providers.is_unavailable(&fallback) {
                continue;
            }
            match resolved_profile(&self.providers, fallback.clone()).await {
                Ok(resolved) => {
                    tracing::warn!(
                        "profile '{named}' failed lately; this turn starts on '{fallback}'"
                    );
                    self.announce_fallback(&named, &fallback).await;
                    self.serve_from(resolved);
                    return;
                }
                Err(error) => {
                    tracing::warn!("fallback profile '{fallback}' cannot be built: {error}");
                }
            }
        }
    }

    /// After a request failed in a way the retry policy will not retry: open the failed profile's
    /// breaker and bind the next profile in the chain this turn has not tried, answering whether
    /// there was one. The breaker is not consulted here, only at a turn's start: the alternative
    /// at this point is failing the turn, so a turn asks each profile in its chain at most once,
    /// in order from the one that failed and wrapping round to the ones its start skipped, and
    /// fails only once every one of them has refused it. `content_started` keeps the guard the
    /// retry has, since a fallback re-sends the request the same way a retry does and must not
    /// double-emit either.
    pub(super) async fn fall_back(&self, error: &MekaError, content_started: bool) -> bool {
        if content_started || !error.is_profile_unavailable() {
            return false;
        }
        let failed = self.cells.profile.current().profile;
        self.providers
            .mark_unavailable(&failed, fallback_cooldown(error.resumes_after()));
        self.note_tried(&failed);
        let named = self.named_profile_name();
        let mut chain = vec![named.clone()];
        chain.extend(self.providers.fallbacks_of(&named));
        let failed_at = chain
            .iter()
            .position(|profile| *profile == failed)
            .unwrap_or(0);
        let untried: Vec<String> = {
            let tried = crate::sync::lock(&self.profiles_tried);
            chain
                .iter()
                .skip(failed_at + 1)
                .chain(chain.iter().take(failed_at))
                .filter(|profile| !tried.contains(profile))
                .cloned()
                .collect()
        };
        for candidate in untried {
            self.note_tried(&candidate);
            // The named profile is in the cell already; a fallback has to be built.
            let fallback = if candidate == named {
                None
            } else {
                match resolved_profile(&self.providers, candidate.clone()).await {
                    Ok(resolved) => Some(resolved),
                    Err(build_error) => {
                        tracing::warn!(
                            "fallback profile '{candidate}' cannot be built: {build_error}"
                        );
                        continue;
                    }
                }
            };
            tracing::warn!(
                "profile '{failed}' failed ({error}); this turn continues on '{candidate}'"
            );
            self.announce_fallback(&failed, &candidate).await;
            match fallback {
                Some(fallback) => self.serve_from(fallback),
                None => self.cells.profile.restore(),
            }
            return true;
        }
        if chain.len() > 1 {
            tracing::warn!(
                "every profile in the chain ({chain}) refused this turn; it fails on '{failed}'",
                chain = chain.join(", ")
            );
        }
        false
    }

    /// Remember that this turn sent its request to `profile`, so it is not asked twice.
    fn note_tried(&self, profile: &str) {
        let mut tried = crate::sync::lock(&self.profiles_tried);
        if !tried.iter().any(|tried| tried == profile) {
            tried.push(profile.to_string());
        }
    }

    async fn announce_fallback(&self, unavailable: &str, fallback: &str) {
        self.cells
            .frontend
            .emit(FrontendEvent::Notice(Notice::warn(format!(
                "Profile '{unavailable}' is unavailable; this turn runs on '{fallback}'."
            ))))
            .await;
    }

    /// Run [`crate::provider::Provider::complete`] under the same retry policy a streamed turn
    /// gets, so one transient 429 mid-compaction is not terminal, and with the same fallback when
    /// the retries are spent. The provider is read per attempt, so a fallback bound by one
    /// attempt is what the next one sends to.
    ///
    /// `content_started` is `false` by construction: nothing streamed, so a retry cannot
    /// double-emit. `cancellation` gates the *waits*, as the loops in `run_streaming` and
    /// `run_turn` do, or a `Retry-After` could hold Ctrl+C for a minute; `attempt_cancellation`
    /// gates the request itself, and the two differ on purpose. The checkpoint hands its turn's
    /// token to both, so a stop drops a reply it is still waiting on. The summarizer hands
    /// `attempt_cancellation` a token nothing fires: a canceled token there means the checkpoint
    /// was interrupted and `compact_session` has fallen back to `summarize_via_provider`, the
    /// tier that guarantees the window shrinks, so refusing to send would leave a `/compact` the
    /// user just pressed Ctrl+C on with the window as full as before.
    /// `an_interrupt_ends_the_checkpoint_and_falls_back` pins it.
    pub(super) async fn complete_with_retry(
        &self,
        what: &str,
        request: CompletionRequest<'_>,
        // The tools to send once a fallback binds another profile, whose wire may carry another
        // set, or carry its deferred tools another way; `None` for a request that carries none.
        rebuild: Option<&(dyn Fn() -> crate::tools::ActiveTools + Sync)>,
        cancellation: &CancellationToken,
        attempt_cancellation: &CancellationToken,
    ) -> Result<crate::provider::Completion> {
        let mut started = std::time::Instant::now();
        let mut retries = 0_u32;
        let mut rebuilt: Option<crate::tools::ActiveTools> = None;
        loop {
            let provider: Arc<dyn crate::provider::Provider> = self.provider();
            let attempt = match &rebuilt {
                Some(active) => CompletionRequest {
                    tools: &active.definitions,
                    deferred_tools: &active.deferred_definitions,
                    ..request.clone()
                },
                None => request.clone(),
            };
            match provider
                .complete(attempt, attempt_cancellation.clone())
                .await
            {
                Ok(completed) => {
                    self.providers.mark_available(&self.profile());
                    return Ok(completed);
                }
                Err(error) => {
                    let Some(delay) =
                        should_retry_provider_error(&error, false, retries, started.elapsed())
                    else {
                        // The profile's retries are spent; a fallback starts the sequence over
                        // on a budget of its own.
                        if self.fall_back(&error, false).await {
                            started = std::time::Instant::now();
                            retries = 0;
                            rebuilt = rebuild.map(|rebuild| rebuild());
                            continue;
                        }
                        return Err(error);
                    };
                    let ceiling = crate::provider::retry::MAX_PROVIDER_RETRIES;
                    tracing::warn!(
                        "{what} failed transiently (attempt {attempt}/{ceiling}), retrying in {delay:?}: \
                         {error}",
                        attempt = retries + 1
                    );
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
                    }
                    retries += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::{
        agent::{TurnCancellation, TurnInput, TurnOutcome, tests::build_test_agent_on},
        conversation::Conversation,
        provider::{
            ProviderRegistry,
            mock::{MockEvent, MockProvider, MockStopReason},
        },
        store::Store,
    };

    /// A profile `main` with `backup` as its one fallback, each on a scripted provider of its own.
    struct Family {
        agent: Agent,
        store: Store,
        session: uuid::Uuid,
        providers: Arc<ProviderRegistry>,
        frontend: Arc<crate::frontend::testing::RecordingFrontend>,
        main: Arc<MockProvider>,
        backup: Arc<MockProvider>,
    }

    async fn family(
        main_rounds: Vec<Vec<MockEvent>>,
        backup_rounds: Vec<Vec<MockEvent>>,
    ) -> Family {
        family_over(
            MockProvider::from_rounds(main_rounds),
            MockProvider::from_rounds(backup_rounds),
        )
        .await
    }

    async fn family_over(main: MockProvider, backup: MockProvider) -> Family {
        let store = Store::for_test().await;
        let account = |name: &str| {
            (name.to_string(), crate::config::AccountConfig {
                backend: "anthropic-messages".to_string(),
                ..Default::default()
            })
        };
        let profile = |name: &str, fallbacks: Vec<&str>| {
            (name.to_string(), crate::config::ProfileConfig {
                account: name.to_string(),
                model: Some("m".to_string()),
                fallbacks: fallbacks.into_iter().map(str::to_string).collect(),
                ..Default::default()
            })
        };
        let providers = Arc::new(ProviderRegistry::for_test_over(
            [account("main"), account("backup")].into_iter().collect(),
            [profile("main", vec!["backup"]), profile("backup", vec![])]
                .into_iter()
                .collect(),
            store.token_store(),
        ));
        let main = Arc::new(main);
        let backup = Arc::new(backup);
        providers.install_scripted_for(
            "main",
            Arc::clone(&main) as Arc<dyn crate::provider::Provider>,
        );
        providers.install_scripted_for(
            "backup",
            Arc::clone(&backup) as Arc<dyn crate::provider::Provider>,
        );
        let resolved = resolved_profile(&providers, "main".to_string())
            .await
            .expect("main resolves");
        let frontend = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        let agent = build_test_agent_on(
            crate::session::SessionMaterials {
                providers: Arc::clone(&providers),
                ..crate::session::SessionMaterials::for_test(store.clone())
            },
            resolved,
            frontend.clone(),
            crate::tools::ToolRegistry::new(),
        );
        let session = store
            .create_session(None, "main".to_string())
            .await
            .expect("create session");
        agent.cells().session_id.set(session);
        Family {
            agent,
            store,
            session,
            providers,
            frontend,
            main,
            backup,
        }
    }

    fn answer(text: &str) -> Vec<MockEvent> {
        vec![
            MockEvent::Text {
                text: text.to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]
    }

    fn exhausted() -> Vec<MockEvent> {
        vec![MockEvent::FailExhausted {
            message: "insufficient_quota".to_string(),
            resets_after_secs: Some(600),
        }]
    }

    fn overloaded() -> Vec<MockEvent> {
        vec![MockEvent::FailRetryable {
            message: "529 overloaded".to_string(),
            retry_after_secs: None,
        }]
    }

    async fn run(family: &Family) -> Result<TurnOutcome> {
        let mut messages = Conversation::new();
        family
            .agent
            .run_turn(
                &mut messages,
                TurnInput::from_parts("hello".to_string(), Vec::new()).expect("a prompt"),
                TurnCancellation::new(),
            )
            .await
    }

    async fn last_turn_profile(family: &Family) -> Option<String> {
        family
            .store
            .turn_store()
            .load_turns(family.session)
            .await
            .expect("turns")
            .last()
            .expect("a turn")
            .profile
            .clone()
    }

    fn notices(family: &Family) -> Vec<String> {
        family
            .frontend
            .events()
            .into_iter()
            .filter_map(|event| match event {
                FrontendEvent::Notice(notice) => Some(notice.text),
                _ => None,
            })
            .collect()
    }

    /// A quota refusal is never retried in place: the turn continues on the fallback, the row
    /// says so, the notice says so, and the session is back on its own profile afterwards.
    #[tokio::test]
    async fn a_turn_whose_profile_is_exhausted_continues_on_its_fallback() {
        let family = family(vec![exhausted()], vec![answer("from backup")]).await;
        let outcome = run(&family).await.expect("the fallback answers");
        assert_eq!(outcome, TurnOutcome::EndTurn);
        assert_eq!(
            family.main.streams().len(),
            1,
            "one refusal, no retry in place"
        );
        assert_eq!(family.backup.streams().len(), 1);
        assert_eq!(last_turn_profile(&family).await.as_deref(), Some("backup"));
        assert_eq!(
            family.agent.profile(),
            "main",
            "the turn's end restores the named profile"
        );
        assert_eq!(family.agent.serving_fallback(), None);
        assert!(family.providers.is_unavailable("main"));
        assert!(!family.providers.is_unavailable("backup"));
        assert!(
            notices(&family)
                .iter()
                .any(|text| text == "Profile 'main' is unavailable; this turn runs on 'backup'."),
            "{:?}",
            notices(&family)
        );
    }

    /// A tool that exists to be carried or not.
    struct Probe;

    #[async_trait::async_trait]
    impl crate::tools::Tool for Probe {
        fn definition(&self) -> crate::provider::ToolDefinition {
            crate::provider::ToolDefinition {
                name: "probe_inspect".to_string(),
                description: "probe".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                title: None,
                annotations: None,
                meta: None,
            }
        }

        fn required_permission(&self) -> crate::permission::Permission {
            crate::permission::Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _context: crate::tools::ToolContext,
        ) -> Result<crate::tools::ToolOutput> {
            Ok(crate::tools::ToolOutput::text("probed".to_string(), false))
        }
    }

    /// A fallback onto a wire that refuses a tool the failed profile's wire carried sends what the
    /// serving wire carries, not the request built for the profile that failed.
    #[tokio::test]
    async fn a_fallback_sends_the_tools_the_serving_wire_carries() {
        let family = family_over(
            MockProvider::from_rounds(vec![exhausted()]),
            MockProvider::from_rounds(vec![answer("from backup")]).refusing("probe_inspect"),
        )
        .await;
        family
            .agent
            .tool_registry
            .register(Arc::new(Probe))
            .expect("register");
        family
            .agent
            .tool_registry
            .publish_profile(&family.agent.cells().profile);
        run(&family).await.expect("the fallback answers");
        let carried = |streams: Vec<crate::provider::mock::StreamRequest>| -> Vec<String> {
            streams
                .last()
                .expect("a request")
                .tools
                .iter()
                .map(|tool| tool.name.clone())
                .collect()
        };
        assert!(carried(family.main.streams()).contains(&"probe_inspect".to_string()));
        assert!(
            !carried(family.backup.streams()).contains(&"probe_inspect".to_string()),
            "the fallback was sent a tool its wire refuses"
        );
    }

    /// While the named profile's breaker is open the next turn starts on the fallback without
    /// asking it; once the breaker closes the named profile is back.
    #[tokio::test]
    async fn the_next_turn_starts_on_the_fallback_while_the_breaker_is_open() {
        let family = family(vec![exhausted(), answer("main again")], vec![
            answer("one"),
            answer("two"),
        ])
        .await;
        run(&family).await.expect("first turn");
        run(&family).await.expect("second turn");
        assert_eq!(
            family.main.streams().len(),
            1,
            "the open breaker spared the named profile"
        );
        assert_eq!(family.backup.streams().len(), 2);
        assert_eq!(last_turn_profile(&family).await.as_deref(), Some("backup"));
        assert_eq!(
            notices(&family).len(),
            2,
            "a turn started on a fallback says so as one that fell back does"
        );

        family.providers.mark_available("main");
        run(&family).await.expect("third turn");
        assert_eq!(family.main.streams().len(), 2);
        assert_eq!(last_turn_profile(&family).await, None);
    }

    /// With every breaker open the named profile is tried anyway, because something has to probe.
    #[tokio::test]
    async fn with_every_breaker_open_the_named_profile_is_tried() {
        let family = family(vec![answer("recovered")], vec![]).await;
        family
            .providers
            .mark_unavailable("main", Duration::from_secs(300));
        family
            .providers
            .mark_unavailable("backup", Duration::from_secs(300));
        run(&family).await.expect("the probe succeeds");
        assert_eq!(family.main.streams().len(), 1);
        assert!(family.backup.streams().is_empty());
        assert_eq!(last_turn_profile(&family).await, None);
        assert!(
            !family.providers.is_unavailable("main"),
            "an answer closes the breaker"
        );
    }

    /// The breaker decides where a turn starts and nothing more: a fallback marked by a transient
    /// failure is still tried once the profile the turn started on refuses, so a turn fails only
    /// once every profile in the chain has refused it.
    #[tokio::test]
    async fn a_marked_fallback_is_still_tried_once_the_turn_is_running() {
        let family = family(vec![exhausted()], vec![answer("from backup")]).await;
        family
            .providers
            .mark_unavailable("main", Duration::from_secs(3_600));
        family
            .providers
            .mark_unavailable("backup", Duration::from_secs(300));
        run(&family).await.expect("the marked fallback answers");
        assert_eq!(
            family.main.streams().len(),
            1,
            "with every breaker open the named profile is probed first"
        );
        assert_eq!(family.backup.streams().len(), 1);
        assert_eq!(last_turn_profile(&family).await.as_deref(), Some("backup"));
        assert!(
            !family.providers.is_unavailable("backup"),
            "its answer closed its breaker"
        );
        assert!(family.providers.is_unavailable("main"));
    }

    /// A turn that started past a marked profile wraps round to it when the rest of the chain
    /// refuses, and asks each profile once: a chain that refuses everywhere fails rather than
    /// going round again.
    #[tokio::test]
    async fn a_turn_wraps_to_the_profiles_its_start_skipped_and_asks_each_once() {
        let family = family(vec![answer("recovered")], vec![exhausted()]).await;
        family
            .providers
            .mark_unavailable("main", Duration::from_secs(300));
        run(&family)
            .await
            .expect("the named profile answers after the fallback refused");
        assert_eq!(
            family.backup.streams().len(),
            1,
            "the turn started on the fallback"
        );
        assert_eq!(family.main.streams().len(), 1);
        assert_eq!(
            last_turn_profile(&family).await,
            None,
            "the session's own profile served in the end"
        );

        let refusing = self::family(vec![exhausted()], vec![exhausted()]).await;
        refusing
            .providers
            .mark_unavailable("main", Duration::from_secs(300));
        run(&refusing).await.expect_err("every profile refused");
        assert_eq!(refusing.backup.streams().len(), 1);
        assert_eq!(
            refusing.main.streams().len(),
            1,
            "asked once, not round again"
        );
    }

    /// A spent retry budget falls back the same way, and the fallback gets a budget of its own.
    #[tokio::test(start_paused = true)]
    async fn a_spent_retry_budget_falls_back() {
        let family = family(vec![overloaded(), overloaded(), overloaded()], vec![
            overloaded(),
            answer("from backup"),
        ])
        .await;
        run(&family).await.expect("the fallback's retry answers");
        assert_eq!(
            family.main.streams().len(),
            3,
            "the initial attempt and both retries"
        );
        assert_eq!(
            family.backup.streams().len(),
            2,
            "a retry of its own on the fallback"
        );
        assert_eq!(last_turn_profile(&family).await.as_deref(), Some("backup"));
    }

    /// Output the user has already seen blocks a fallback as it blocks a retry.
    #[tokio::test]
    async fn output_already_shown_blocks_a_fallback() {
        let family = family(
            vec![vec![
                MockEvent::Text {
                    text: "half an answer".to_string(),
                },
                MockEvent::FailRetryable {
                    message: "connection reset".to_string(),
                    retry_after_secs: None,
                },
            ]],
            vec![answer("never asked")],
        )
        .await;
        run(&family)
            .await
            .expect_err("the turn fails where the stream broke");
        assert!(family.backup.streams().is_empty());
        assert!(!family.providers.is_unavailable("main"));
        assert_eq!(last_turn_profile(&family).await, None);
    }

    /// A refusal another profile would repeat, such as a rejected credential, is not a fallback's
    /// to answer.
    #[tokio::test]
    async fn an_error_another_profile_would_repeat_does_not_fall_back() {
        let family = family(
            vec![vec![MockEvent::Fail {
                message: "401 authentication_error".to_string(),
            }]],
            vec![answer("never asked")],
        )
        .await;
        run(&family).await.expect_err("the turn fails");
        assert!(family.backup.streams().is_empty());
        assert!(!family.providers.is_unavailable("main"));
        assert!(notices(&family).is_empty());
    }

    /// Every profile in the chain refused: the turn fails on the last one tried, the row names
    /// it, and every breaker is open.
    #[tokio::test]
    async fn the_chain_exhausted_fails_the_turn_on_the_last_profile() {
        let family = family(vec![exhausted()], vec![exhausted()]).await;
        let error = run(&family).await.expect_err("nothing left to try");
        assert!(
            matches!(error, MekaError::ProviderExhausted { .. }),
            "{error}"
        );
        assert_eq!(last_turn_profile(&family).await.as_deref(), Some("backup"));
        assert!(family.providers.is_unavailable("main"));
        assert!(family.providers.is_unavailable("backup"));
        assert_eq!(family.agent.profile(), "main");
    }

    /// The non-streaming door falls back too, and a compaction run between turns puts the named
    /// profile back when it ends, while one inside a turn leaves the turn's fallback in place.
    #[tokio::test]
    async fn a_completion_falls_back_and_an_idle_compaction_restores() {
        let family = family(vec![exhausted()], vec![answer("from backup")]).await;
        let token = CancellationToken::new();
        let request = CompletionRequest::new("system", &[], &[]);
        family
            .agent
            .complete_with_retry("a test call", request.clone(), None, &token, &token)
            .await
            .expect("the fallback completes");
        assert_eq!(family.agent.serving_fallback().as_deref(), Some("backup"));
        assert_eq!(family.main.completions().len(), 1);
        assert_eq!(family.backup.completions().len(), 1);

        *crate::sync::lock(&family.agent.current_turn) = Some(uuid::Uuid::new_v4());
        drop(RestoreWhenIdle {
            agent: &family.agent,
        });
        assert_eq!(
            family.agent.serving_fallback().as_deref(),
            Some("backup"),
            "inside a turn the compaction's fallback is the turn's to keep"
        );
        *crate::sync::lock(&family.agent.current_turn) = None;
        drop(RestoreWhenIdle {
            agent: &family.agent,
        });
        assert_eq!(family.agent.serving_fallback(), None);
        assert_eq!(family.agent.profile(), "main");
    }

    /// A repin while a fallback serves is the explicit act, and it wins: the fallback ends with
    /// the move, and the turn's end has nothing to put back.
    #[tokio::test]
    async fn a_repin_while_a_fallback_serves_ends_the_fallback() {
        let family = family(vec![exhausted()], vec![answer("from backup")]).await;
        let token = CancellationToken::new();
        family
            .agent
            .complete_with_retry(
                "a test call",
                CompletionRequest::new("system", &[], &[]),
                None,
                &token,
                &token,
            )
            .await
            .expect("the fallback completes");
        assert_eq!(family.agent.serving_fallback().as_deref(), Some("backup"));

        let backup = resolved_profile(&family.providers, "backup".to_string())
            .await
            .expect("backup resolves");
        family.agent.set_provider(backup);
        assert_eq!(
            family.agent.serving_fallback(),
            None,
            "the pin is the session's own profile now"
        );
        family.agent.restore_named_profile();
        assert_eq!(
            family.agent.profile(),
            "backup",
            "the turn's end does not undo the pin"
        );
    }
}
