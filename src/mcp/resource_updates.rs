//! In-memory ledger of resources that have been reported as changed via
//! `notifications/resources/updated`. The agent can query this via the `mcp_resource_updates_list`
//! builtin tool to see which resources need refreshing without subscribing again. One per
//! [`McpClientContext`], which is what a notification arrives on.
//!
//! [`McpClientContext`]: crate::mcp::McpClientContext

use std::{
    collections::HashMap,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

type Ledger = HashMap<(String /* server */, String /* uri */), u64>;

#[derive(Default)]
pub(crate) struct ResourceUpdates {
    ledger: Mutex<Ledger>,
}

/// The most entries the ledger will hold. Keyed by `(server, uri)`, so a server that invents a
/// fresh URI per notification grows it without bound, and nothing ever removes an entry. The bound
/// is generous: a server with more than this many *distinct* resources changing in one process
/// lifetime is not one the agent can act on resource by resource anyway.
const MAX_LEDGER_ENTRIES: usize = 10_000;

impl ResourceUpdates {
    /// Record that a resource was updated. Stamp is unix seconds.
    pub(crate) fn record(&self, server_name: &str, uri: &str) {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut state = crate::sync::lock(&self.ledger);
        let key = (server_name.to_string(), uri.to_string());
        // Re-recording a URI already in the ledger is just a restamp and cannot grow it.
        if state.len() >= MAX_LEDGER_ENTRIES && !state.contains_key(&key) {
            // Drop the oldest rather than refusing the new one: the agent asks this list what to
            // re-read, and the freshest changes are the ones it has not seen.
            if let Some(oldest) = state
                .iter()
                .min_by_key(|(_, stamp)| **stamp)
                .map(|(key, _)| key.clone())
            {
                state.remove(&oldest);
                tracing::debug!(
                    "resource update ledger is full at {MAX_LEDGER_ENTRIES} entries; evicted {server}:{uri}",
                    server = oldest.0,
                    uri = oldest.1
                );
            }
        }
        state.insert(key, stamp);
    }

    /// Snapshot every recorded update. Returned entries are sorted by server name then URI for
    /// stable output.
    pub(crate) fn snapshot(&self) -> Vec<(String, String, u64)> {
        let state = crate::sync::lock(&self.ledger);
        let mut out: Vec<(String, String, u64)> = state
            .iter()
            .map(|((server, uri), stamp)| (server.clone(), uri.clone(), *stamp))
            .collect();
        drop(state);
        out.sort_by(|a, b| (a.0.as_str(), a.1.as_str()).cmp(&(b.0.as_str(), b.1.as_str())));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recorded_update_appears_in_the_snapshot() {
        let updates = ResourceUpdates::default();
        updates.record("srv", "file:///a");
        let snap = updates.snapshot();
        assert!(snap.iter().any(|(s, u, _)| s == "srv" && u == "file:///a"));
    }

    /// The ledger lives as long as its context and is fed by a *server's* notifications, so an
    /// unbounded one is memory a remote peer decides the size of. Raising `MAX_LEDGER_ENTRIES` to
    /// `usize::MAX` left every suite green.
    #[test]
    fn the_ledger_stops_growing_at_its_ceiling() {
        let updates = ResourceUpdates::default();
        for index in 0..MAX_LEDGER_ENTRIES + 500 {
            updates.record("srv-flood", &format!("file:///{index}"));
        }
        let total = updates.snapshot().len();
        assert!(
            total <= MAX_LEDGER_ENTRIES,
            "the ledger grew past its ceiling: {total} > {MAX_LEDGER_ENTRIES}"
        );
    }
}
