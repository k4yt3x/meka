//! Plumbing for forwarding MCP `notifications/progress` updates from servers up into the active
//! frontend. rmcp announces a `progressToken` on every request it sends; the server then emits
//! progress notifications carrying that token. The [`McpClientContext`] keeps a map from the
//! server and token to the call's frontend so the rmcp notification dispatch (which runs on a
//! separately-spawned task; see `rmcp::service::spawn_service_task`) can find the per-session UI
//! even though it can't read the caller's task-local. RAII guards remove the entry when the
//! in-flight call finishes so orphaned tokens don't pile up.
//!
//! [`McpClientContext`]: crate::mcp::McpClientContext

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use rmcp::model::{NumberOrString, ProgressNotificationParam, ProgressToken};

use crate::frontend::{Frontend, FrontendEvent, ProgressUpdate};

/// The in-flight calls of one MCP client context. Per-call entries are keyed by the server and
/// its progress token; the entry also owns the per-session frontend so the dispatch path can route
/// events to the correct UI without relying on task-local propagation (which doesn't survive
/// rmcp's spawned-handler tasks).
#[derive(Default)]
pub(crate) struct ProgressRegistry {
    entries: Arc<Mutex<HashMap<Key, Entry>>>,
}

/// What one in-flight request is found by. rmcp mints the token from a counter that starts at
/// zero on every connection, so two servers with calls in flight hand out the same numbers and a
/// token names a request only beside the server whose connection minted it.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    server_name: String,
    token: String,
}

struct Entry {
    tool_name: String,
    tool_use_id: Option<String>,
    /// The session's frontend that initiated this tool call. `None` outside an agent-driven call
    /// site (e.g. an MCP connection probe). When `None`, `dispatch` falls back to a tracing log
    /// and the user sees nothing.
    frontend: Option<Arc<dyn Frontend>>,
    /// When the call is cut at its own bound, which an elicitation it raises cannot outlive.
    deadline: tokio::time::Instant,
}

impl ProgressRegistry {
    /// Register an in-flight request to `server_name` under `token`, the one rmcp minted for it
    /// and the server echoes. The frontend is the calling session's, stored alongside the call
    /// context so the rmcp notification handler can later look it up by token. Returns a
    /// [`ProgressGuard`] that removes the entry when dropped, the `_progress_guard` binding the
    /// MCP tool adapter holds while the request is in flight.
    pub(crate) fn register(
        &self,
        token: &ProgressToken,
        server_name: String,
        tool_name: String,
        tool_use_id: Option<String>,
        frontend: Option<Arc<dyn Frontend>>,
        deadline: tokio::time::Instant,
    ) -> ProgressGuard {
        let key = Key {
            server_name,
            token: token_key(token),
        };
        crate::sync::lock(&self.entries).insert(key.clone(), Entry {
            tool_name,
            tool_use_id,
            frontend,
            deadline,
        });
        ProgressGuard {
            entries: Arc::clone(&self.entries),
            key: Some(key),
        }
    }

    /// Called from `MekaClientHandler::on_progress` (the `ClientHandler` trait impl in
    /// `src/mcp/handler.rs`) with the server the notification came from. Looks up the registered
    /// context by server and token, forwards the update to the per-session frontend, and emits an
    /// `info!` log line for the off-frontend audience (e.g. a tail on the meka stderr).
    pub(crate) async fn dispatch(&self, server_name: &str, params: ProgressNotificationParam) {
        let key = Key {
            server_name: server_name.to_string(),
            token: token_key(&params.progress_token),
        };
        let snapshot = {
            let entries = crate::sync::lock(&self.entries);
            entries.get(&key).map(|e| {
                (
                    e.tool_name.clone(),
                    e.tool_use_id.clone(),
                    e.frontend.clone(),
                )
            })
        };
        let Some((tool_name, tool_use_id, frontend)) = snapshot else {
            tracing::debug!(
                "MCP '{server_name}' progress for unknown token '{token}' (likely canceled); \
                 ignored",
                token = key.token
            );
            return;
        };
        let update = ProgressUpdate {
            server_name: server_name.to_string(),
            tool_name,
            tool_use_id,
            progress: params.progress,
            total: params.total,
            // Sanitized at ingress like a connector's stderr: the line below reaches the terminal
            // at `-v` with whatever escapes the server put in it.
            message: params
                .message
                .map(|message| crate::text::sanitize_to_line(&message, usize::MAX)),
        };
        let total = update
            .total
            .map(|total| format!("/{total}"))
            .unwrap_or_default();
        let message = update
            .message
            .as_deref()
            .map(|message| format!(", {message}"))
            .unwrap_or_default();
        tracing::info!(
            "MCP '{server_name}' {tool_name} progress: {progress}{total}{message}",
            tool_name = update.tool_name,
            progress = update.progress,
        );
        if let Some(frontend) = frontend {
            frontend.emit(FrontendEvent::McpProgress(update)).await;
        }
    }

    /// Where an elicitation from `server_name` goes: the frontend of the tool calls in flight on
    /// that server, when they all belong to one. The server's request lands on the rmcp handler
    /// task with only its own request id and the originating server, and MCP's elicitation is a
    /// server-initiated request with no link back to the tool call that provoked it, so the
    /// in-flight calls are all there is to go on. One session with parallel calls to the server is
    /// unambiguous; two sessions on the same server are not, and picking one would surface the
    /// prompt in the wrong editor and answer the other session's call with what this one typed.
    ///
    /// The route's deadline is the latest of those calls' own: whichever of them asked, the prompt
    /// is answerable for no longer than that call waits, and a call that ends sooner is canceled
    /// with its question by the server.
    pub(crate) fn elicitation_route(&self, server_name: &str) -> ElicitationRoute {
        let calls: Vec<(Arc<dyn Frontend>, tokio::time::Instant)> =
            crate::sync::lock(&self.entries)
                .iter()
                .filter(|(key, _entry)| key.server_name == server_name)
                .filter_map(|(_key, entry)| {
                    entry
                        .frontend
                        .clone()
                        .map(|frontend| (frontend, entry.deadline))
                })
                .collect();
        let mut route = ElicitationRoute::Nobody;
        for (frontend, deadline) in calls {
            route = match route {
                ElicitationRoute::Nobody => ElicitationRoute::To { frontend, deadline },
                ElicitationRoute::To {
                    frontend: known,
                    deadline: latest,
                } if std::ptr::addr_eq(Arc::as_ptr(&known), Arc::as_ptr(&frontend)) => {
                    ElicitationRoute::To {
                        frontend: known,
                        deadline: latest.max(deadline),
                    }
                }
                _ => return ElicitationRoute::Ambiguous,
            };
        }
        route
    }

    /// Test helper: whether a request to `server_name` is registered under `token`.
    #[cfg(test)]
    pub(crate) fn is_registered(&self, server_name: &str, token: &ProgressToken) -> bool {
        crate::sync::lock(&self.entries).contains_key(&Key {
            server_name: server_name.to_string(),
            token: token_key(token),
        })
    }
}

/// The registry's spelling of a token: rmcp mints numbers, and a number and its decimal string
/// are one token however a server writes it back.
fn token_key(token: &ProgressToken) -> String {
    match &token.0 {
        NumberOrString::String(text) => text.to_string(),
        NumberOrString::Number(number) => number.to_string(),
    }
}

/// Where an elicitation goes, from [`ProgressRegistry::elicitation_route`].
pub(crate) enum ElicitationRoute {
    /// The one frontend with calls in flight on the server, and the latest of their deadlines.
    To {
        frontend: Arc<dyn Frontend>,
        deadline: tokio::time::Instant,
    },
    /// No call on the server is in flight: it elicited outside a tool call, or the call's guard
    /// is already gone.
    Nobody,
    /// Calls from more than one frontend are in flight on the server.
    Ambiguous,
}

/// RAII guard that removes the registry entry when dropped.
pub(crate) struct ProgressGuard {
    entries: Arc<Mutex<HashMap<Key, Entry>>>,
    key: Option<Key>,
}

impl Drop for ProgressGuard {
    fn drop(&mut self) {
        if let Some(key) = self.key.take() {
            crate::sync::lock(&self.entries).remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::frontend::{Frontend, FrontendEvent, testing::RecordingFrontend};

    /// A token as rmcp mints it for a request: a number.
    /// A call bound no test reaches.
    fn far() -> tokio::time::Instant {
        tokio::time::Instant::now() + std::time::Duration::from_secs(3600)
    }

    fn minted(number: i64) -> ProgressToken {
        ProgressToken(NumberOrString::Number(number))
    }

    /// The progress messages `recorder` was shown, in order.
    fn progress_messages(recorder: &RecordingFrontend) -> Vec<(String, Option<String>)> {
        recorder
            .events()
            .iter()
            .filter_map(|event| match event {
                FrontendEvent::McpProgress(update) => {
                    Some((update.tool_name.clone(), update.message.clone()))
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn guard_cleans_up_on_drop() {
        let registry = ProgressRegistry::default();
        let token = minted(1);
        {
            let _guard = registry.register(&token, "srv".into(), "tool".into(), None, None, far());
            assert!(
                registry.is_registered("srv", &token),
                "token missing from registry while guard is still alive"
            );
        }
        assert!(
            !registry.is_registered("srv", &token),
            "the token lingered after the guard dropped: ProgressGuard::drop didn't clean up"
        );
    }

    #[tokio::test]
    async fn dispatch_without_registration_is_noop() {
        let registry = ProgressRegistry::default();
        registry
            .dispatch(
                "srv",
                ProgressNotificationParam::new(
                    ProgressToken(NumberOrString::String(Arc::from("unknown"))),
                    1.0,
                ),
            )
            .await;
        // If it didn't panic we're fine.
    }

    /// The new contract: progress dispatch routes through the per-call frontend stored on the
    /// registry entry, not a process-global sink. This proves session A's frontend can receive an
    /// update while a hypothetical session B running the same test concurrently doesn't.
    #[tokio::test]
    async fn dispatch_forwards_to_per_call_frontend() {
        let recorder: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let frontend: Arc<dyn Frontend> = recorder.clone();

        let registry = ProgressRegistry::default();
        let _guard = registry.register(
            &minted(7),
            "srv".into(),
            "tool".into(),
            Some("tu-1".into()),
            Some(frontend),
            far(),
        );

        // The token as a server echoes it: the number rmcp minted, unchanged.
        registry
            .dispatch(
                "srv",
                ProgressNotificationParam::new(minted(7), 0.5)
                    .with_total(1.0)
                    .with_message("halfway"),
            )
            .await;

        let events = recorder.events();
        let update = events
            .iter()
            .filter_map(|event| match event {
                FrontendEvent::McpProgress(update) => Some(update),
                _ => None,
            })
            .next_back()
            .expect("frontend should have received exactly one McpProgress");
        assert_eq!(update.tool_use_id.as_deref(), Some("tu-1"));
        assert_eq!(update.progress, 0.5);
        assert_eq!(update.message.as_deref(), Some("halfway"));
    }

    /// Multi-session isolation: two concurrent calls with distinct frontends must each receive
    /// only their own progress updates. This is the property that makes per-session ACP routing
    /// safe: session A's MCP server progress can't leak to session B.
    #[tokio::test]
    async fn dispatch_isolates_concurrent_calls() {
        let recorder_a: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let recorder_b: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let frontend_a: Arc<dyn Frontend> = recorder_a.clone();
        let frontend_b: Arc<dyn Frontend> = recorder_b.clone();

        let registry = ProgressRegistry::default();
        let _guard_a = registry.register(
            &minted(1),
            "srv".into(),
            "tool-a".into(),
            Some("ta".into()),
            Some(frontend_a),
            far(),
        );
        let _guard_b = registry.register(
            &minted(2),
            "srv".into(),
            "tool-b".into(),
            Some("tb".into()),
            Some(frontend_b),
            far(),
        );

        registry
            .dispatch(
                "srv",
                ProgressNotificationParam::new(minted(1), 1.0).with_message("for-a"),
            )
            .await;
        // A server that writes the number back as a string names the same call.
        registry
            .dispatch(
                "srv",
                ProgressNotificationParam::new(
                    ProgressToken(NumberOrString::String(Arc::from("2"))),
                    2.0,
                )
                .with_message("for-b"),
            )
            .await;

        assert_eq!(
            progress_messages(&recorder_a),
            vec![("tool-a".to_string(), Some("for-a".to_string()))],
            "frontend A must receive only its own update"
        );
        assert_eq!(
            progress_messages(&recorder_b),
            vec![("tool-b".to_string(), Some("for-b".to_string()))],
            "frontend B must receive only its own update"
        );
    }

    /// rmcp mints tokens from a counter that starts at zero on every connection, so two servers
    /// with calls in flight share numbers. Each server's call keeps its own entry: a notification
    /// finds the call on the server it came from, an elicitation finds that server's frontend,
    /// and one call ending leaves the other's entry standing.
    #[tokio::test]
    async fn a_token_names_a_request_only_beside_its_server() {
        let recorder_a: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let recorder_b: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let frontend_a: Arc<dyn Frontend> = recorder_a.clone();
        let frontend_b: Arc<dyn Frontend> = recorder_b.clone();

        let registry = ProgressRegistry::default();
        let guard_a = registry.register(
            &minted(1),
            "srv-a".into(),
            "tool-a".into(),
            Some("ta".into()),
            Some(Arc::clone(&frontend_a)),
            far(),
        );
        let _guard_b = registry.register(
            &minted(1),
            "srv-b".into(),
            "tool-b".into(),
            Some("tb".into()),
            Some(frontend_b),
            far(),
        );

        registry
            .dispatch(
                "srv-a",
                ProgressNotificationParam::new(minted(1), 1.0).with_message("for-a"),
            )
            .await;
        assert_eq!(
            progress_messages(&recorder_a),
            vec![("tool-a".to_string(), Some("for-a".to_string()))],
            "server A's progress is A's call's, shown to A's frontend"
        );
        assert!(
            progress_messages(&recorder_b).is_empty(),
            "server B's frontend saw a notification from server A"
        );
        assert!(
            matches!(
                registry.elicitation_route("srv-a"),
                ElicitationRoute::To { frontend: ref found, .. }
                    if std::ptr::addr_eq(Arc::as_ptr(found), Arc::as_ptr(&frontend_a))
            ),
            "server A's elicitation goes to the frontend of A's call"
        );

        drop(guard_a);
        assert!(!registry.is_registered("srv-a", &minted(1)));
        assert!(
            registry.is_registered("srv-b", &minted(1)),
            "server A's call ending took server B's entry with it"
        );
    }

    /// An elicitation routes to the frontend whose call is in flight on the server, observed by
    /// emitting through the route and finding the event on the recorder registered with.
    #[tokio::test]
    async fn an_elicitation_routes_to_the_frontend_with_a_call_in_flight() {
        let recorder: Arc<RecordingFrontend> = Arc::new(RecordingFrontend::new());
        let frontend: Arc<dyn Frontend> = recorder.clone();
        let registry = ProgressRegistry::default();
        let _guard = registry.register(
            &minted(3),
            "unique-srv-for-test".into(),
            "tool".into(),
            None,
            Some(frontend),
            far(),
        );
        let ElicitationRoute::To {
            frontend: found, ..
        } = registry.elicitation_route("unique-srv-for-test")
        else {
            panic!("one frontend with a call in flight is the route");
        };
        found
            .emit(FrontendEvent::Notice(crate::frontend::Notice::info(
                "probe",
            )))
            .await;
        assert!(
            recorder
                .events()
                .iter()
                .any(|e| matches!(e, FrontendEvent::Notice(n) if n.text == "probe")),
            "the found frontend must be the one originally registered",
        );
    }

    /// The route carries the latest deadline among its frontend's calls on the server, which is
    /// as long as whichever of them asked can still be waiting.
    #[test]
    fn an_elicitation_route_ends_with_the_last_call_that_could_have_asked() {
        let frontend: Arc<dyn Frontend> = Arc::new(RecordingFrontend::new());
        let registry = ProgressRegistry::default();
        let sooner = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        let later = sooner + std::time::Duration::from_secs(10);
        let _first = registry.register(
            &minted(8),
            "srv".into(),
            "tool".into(),
            None,
            Some(Arc::clone(&frontend)),
            later,
        );
        let _second = registry.register(
            &minted(9),
            "srv".into(),
            "tool".into(),
            None,
            Some(Arc::clone(&frontend)),
            sooner,
        );
        assert!(matches!(
            registry.elicitation_route("srv"),
            ElicitationRoute::To { deadline, .. } if deadline == later
        ));
    }

    /// Routed only when every call in flight on the server belongs to one frontend: parallel
    /// calls from one session route, calls from two sessions do not, and the handler declines
    /// rather than guess which editor to ask. Another server's calls do not count.
    #[test]
    fn an_elicitation_with_calls_from_two_frontends_in_flight_is_not_routed() {
        let frontend_a: Arc<dyn Frontend> = Arc::new(RecordingFrontend::new());
        let frontend_b: Arc<dyn Frontend> = Arc::new(RecordingFrontend::new());
        let registry = ProgressRegistry::default();
        assert!(matches!(
            registry.elicitation_route("srv"),
            ElicitationRoute::Nobody
        ));

        let _guard_a = registry.register(
            &minted(4),
            "srv".into(),
            "tool".into(),
            Some("ta".into()),
            Some(Arc::clone(&frontend_a)),
            far(),
        );
        let _guard_a2 = registry.register(
            &minted(5),
            "srv".into(),
            "tool".into(),
            Some("ta2".into()),
            Some(Arc::clone(&frontend_a)),
            far(),
        );
        assert!(
            matches!(
                registry.elicitation_route("srv"),
                ElicitationRoute::To { frontend: ref found, .. }
                    if std::ptr::addr_eq(Arc::as_ptr(found), Arc::as_ptr(&frontend_a))
            ),
            "parallel calls from one frontend route to it"
        );

        let _guard_b = registry.register(
            &minted(6),
            "srv".into(),
            "tool".into(),
            Some("tb".into()),
            Some(Arc::clone(&frontend_b)),
            far(),
        );
        assert!(matches!(
            registry.elicitation_route("srv"),
            ElicitationRoute::Ambiguous
        ));
        assert!(matches!(
            registry.elicitation_route("other"),
            ElicitationRoute::Nobody
        ));
    }
}
