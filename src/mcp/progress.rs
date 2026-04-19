//! Plumbing for forwarding MCP `notifications/progress` updates from servers
//! up into the shell UI. The MCP protocol requires the client to announce a
//! per-request `progressToken`; the server then emits progress notifications
//! carrying that token. We keep a global map from token → sink so the
//! notification handler can route updates to the correct in-flight tool
//! call, with cleanup on drop to avoid leaks when calls time out or error.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use rmcp::model::{NumberOrString, ProgressNotificationParam, ProgressToken};

pub type ProgressSink = Box<dyn Fn(ProgressUpdate) + Send + Sync + 'static>;

/// Single progress update forwarded to the shell.
#[derive(Clone, Debug)]
pub struct ProgressUpdate {
    pub server_name: String,
    pub tool_name: String,
    /// The provider's `tool_use_id` for the in-flight call, when one was
    /// supplied. Reserved for a future renderer that correlates progress
    /// lines to the tool-use entry the LLM is waiting on. Not currently
    /// consumed by the shell, but always populated at the emit site so
    /// downstream code doesn't need a plumbing migration later.
    #[allow(dead_code)]
    pub tool_use_id: Option<String>,
    pub progress: f64,
    pub total: Option<f64>,
    pub message: Option<String>,
}

/// Global progress registry. Singleton so the client handler (constructed at
/// startup) and every concurrent tool call share the same routing table.
struct Registry {
    /// Map from opaque progress token → (context, sink).
    entries: Mutex<HashMap<String, Entry>>,
    /// Optional callback for the shell UI. Populated once the agent loop
    /// wires itself up; until then, updates are silently dropped.
    ui: OnceLock<ProgressSink>,
}

struct Entry {
    server_name: String,
    tool_name: String,
    tool_use_id: Option<String>,
}

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| Registry {
        entries: Mutex::new(HashMap::new()),
        ui: OnceLock::new(),
    })
}

/// Install a UI sink. Subsequent progress notifications fan out here in
/// addition to tracing logs. Only the first sink wins; later calls are
/// no-ops.
pub fn set_ui_sink(sink: ProgressSink) {
    let _ = registry().ui.set(sink);
}

/// Register a freshly-generated progress token for an in-flight tool call.
/// Returns a [`ProgressGuard`] that removes the entry when dropped so
/// orphaned tokens don't pile up.
pub fn register(
    server_name: String,
    tool_name: String,
    tool_use_id: Option<String>,
) -> (ProgressToken, ProgressGuard) {
    let token_str = format!("agsh-{}", uuid::Uuid::new_v4());
    let token = ProgressToken(NumberOrString::String(token_str.clone().into()));
    registry()
        .entries
        .lock()
        .expect("progress mutex poisoned")
        .insert(
            token_str.clone(),
            Entry {
                server_name,
                tool_name,
                tool_use_id,
            },
        );
    (
        token,
        ProgressGuard {
            key: Some(token_str),
        },
    )
}

/// RAII guard that removes the progress-token entry when dropped.
pub struct ProgressGuard {
    key: Option<String>,
}

impl Drop for ProgressGuard {
    fn drop(&mut self) {
        if let Some(key) = self.key.take()
            && let Ok(mut entries) = registry().entries.lock()
        {
            entries.remove(&key);
        }
    }
}

/// Called from [`crate::mcp::AgshClientHandler::on_progress`]. Looks up the
/// registered context and forwards an update to the UI sink (if any) plus a
/// tracing log at info level.
pub fn dispatch(params: ProgressNotificationParam) {
    let key = match &params.progress_token.0 {
        NumberOrString::String(s) => s.to_string(),
        NumberOrString::Number(n) => n.to_string(),
    };
    let entry = {
        let entries = registry().entries.lock().expect("progress mutex poisoned");
        entries.get(&key).map(|e| Entry {
            server_name: e.server_name.clone(),
            tool_name: e.tool_name.clone(),
            tool_use_id: e.tool_use_id.clone(),
        })
    };
    let Some(entry) = entry else {
        tracing::debug!(
            "MCP progress for unknown token '{}' (likely cancelled); ignored",
            key
        );
        return;
    };
    let update = ProgressUpdate {
        server_name: entry.server_name,
        tool_name: entry.tool_name,
        tool_use_id: entry.tool_use_id,
        progress: params.progress,
        total: params.total,
        message: params.message,
    };
    tracing::info!(
        "MCP '{}' {} progress: {}{}{}",
        update.server_name,
        update.tool_name,
        update.progress,
        update.total.map(|t| format!("/{}", t)).unwrap_or_default(),
        update
            .message
            .as_deref()
            .map(|m| format!(" — {}", m))
            .unwrap_or_default()
    );
    if let Some(sink) = registry().ui.get() {
        sink(update);
    }
}

/// Test helper: count outstanding progress-token registrations. Used by the
/// adapter tests to confirm that [`ProgressGuard`] cleans up on drop.
#[cfg(test)]
pub(crate) fn outstanding_count() -> usize {
    registry()
        .entries
        .lock()
        .map(|e| e.len())
        .unwrap_or_default()
}

// Needed so the `uuid` crate's dependency-only presence is documented; other
// code already uses it via `crate::session`.
#[allow(unused_imports)]
use uuid as _uuid_used_for_v4;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn guard_cleans_up_on_drop() {
        // Cargo runs unit tests in parallel and the registry is a shared
        // global, so we can only assert *deltas*, not absolute values:
        // other test threads may be holding their own guards alongside us.
        let before = outstanding_count();
        let inside;
        {
            let (_token, _guard) = register("srv".into(), "tool".into(), None);
            inside = outstanding_count();
            assert!(
                inside > before,
                "expected count to rise after register (before={}, inside={})",
                before,
                inside
            );
        }
        let after = outstanding_count();
        assert!(
            after < inside,
            "expected count to drop after guard fell out of scope (inside={}, after={})",
            inside,
            after
        );
    }

    #[test]
    fn dispatch_without_registration_is_noop() {
        dispatch(ProgressNotificationParam {
            progress_token: ProgressToken(NumberOrString::String(Arc::from("unknown"))),
            progress: 1.0,
            total: None,
            message: None,
        });
        // If it didn't panic we're fine.
    }

    #[test]
    fn dispatch_forwards_to_sink() {
        use std::sync::Mutex as StdMutex;

        let captured: Arc<StdMutex<Vec<ProgressUpdate>>> = Arc::new(StdMutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);
        let sink: ProgressSink = Box::new(move |update| {
            if let Ok(mut vec) = captured_clone.lock() {
                vec.push(update);
            }
        });
        if registry().ui.set(sink).is_err() {
            eprintln!("progress UI sink already installed; skipping forwarding test");
            return;
        }

        let (token, _guard) = register("srv".into(), "tool".into(), Some("tu-1".into()));
        let key = match &token.0 {
            NumberOrString::String(s) => s.to_string(),
            NumberOrString::Number(n) => n.to_string(),
        };
        dispatch(ProgressNotificationParam {
            progress_token: ProgressToken(NumberOrString::String(Arc::from(key))),
            progress: 0.5,
            total: Some(1.0),
            message: Some("halfway".into()),
        });
        let vec = captured.lock().expect("mutex");
        let update = vec.last().expect("sink should receive update");
        assert_eq!(update.tool_use_id.as_deref(), Some("tu-1"));
        assert_eq!(update.progress, 0.5);
    }
}
