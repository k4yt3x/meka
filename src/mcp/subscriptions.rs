//! What the model subscribed to on each server, and the stream a modern server's notifications
//! arrive on.
//!
//! A server of the era with an `initialize` handshake pushes `tools/list_changed` and
//! `resources/updated` on its own and takes `resources/subscribe` per URI. A server of the
//! 2026-07-28 era pushes nothing unsolicited: every notification kind a client wants rides one
//! `subscriptions/listen` stream the client opened, acknowledged with the subset the server will
//! honor, and the URIs the client wants updates for are part of that stream's filter. So the
//! subscribed set lives here, per entry, and is applied to whichever connection the entry holds:
//! as the per-URI requests on a legacy one, as the stream's filter on a modern one, and again on
//! every connection the entry adopts after the first.

use std::{collections::BTreeSet, sync::Arc, time::Duration};

use rmcp::{
    model::{
        ServerNotification, SubscribeRequestParams, SubscriptionFilter, UnsubscribeRequestParams,
    },
    service::{Subscription, SubscriptionEnd},
};
use tokio_util::sync::CancellationToken;

use super::{ServerEntry, bounded, handler};
use crate::error::{MekaError, Result};

/// How long the pump waits before opening another stream in place of one that ended while its
/// connection lives. Doubled after each stream that ended within [`REOPEN_MAX_BACKOFF`] of
/// opening, up to that bound, so a server that ends every stream at once is asked again at a
/// walking pace rather than in a loop; a stream that lived longer starts the schedule over.
const REOPEN_INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const REOPEN_MAX_BACKOFF: Duration = Duration::from_secs(30);

/// The wait before opening a stream in place of one that lived `lived`: the schedule starts over
/// after a stream that lived out the longest wait, and continues from `backoff` after a shorter
/// one.
fn backoff_before_reopen(backoff: Duration, lived: Duration) -> Duration {
    if lived >= REOPEN_MAX_BACKOFF {
        REOPEN_INITIAL_BACKOFF
    } else {
        backoff
    }
}

/// The wait after `backoff`: doubled, up to the longest.
fn next_backoff(backoff: Duration) -> Duration {
    backoff.saturating_mul(2).min(REOPEN_MAX_BACKOFF)
}

/// One server entry's subscribed set, and the pump draining its stream while one is open.
#[derive(Default)]
pub(crate) struct Subscriptions {
    uris: std::sync::Mutex<BTreeSet<String>>,
    /// Held across a re-issue, so two tools changing the set at once open their streams in turn
    /// and the pump installed last drains a stream opened for the set as it stood by then.
    pump: tokio::sync::Mutex<Option<Pump>>,
}

/// What ends the task draining one stream: canceled when a later stream replaces it.
struct Pump {
    stop: CancellationToken,
}

impl Subscriptions {
    /// The URIs the model asked for, in one order.
    pub(crate) fn uris(&self) -> Vec<String> {
        crate::sync::lock(&self.uris).iter().cloned().collect()
    }

    fn insert(&self, uri: &str) {
        crate::sync::lock(&self.uris).insert(uri.to_string());
    }

    fn remove(&self, uri: &str) {
        crate::sync::lock(&self.uris).remove(uri);
    }
}

/// Whether the entry's connection is of the era whose notifications ride a listen stream.
async fn is_modern(entry: &ServerEntry) -> bool {
    entry
        .protocol_version()
        .await
        .is_some_and(|version| !version.has_initialize())
}

/// Ask `entry`'s server for updates to `uri` from now on.
///
/// Succeeds only when the server will send them: a legacy server's refusal is its error, a modern
/// server's is the acknowledgment leaving the URI out. The set keeps neither, so a URI the server
/// will not watch is not asked for again on every later connection.
pub(crate) async fn subscribe(
    entry: &Arc<ServerEntry>,
    uri: String,
    cancellation: &CancellationToken,
) -> Result<()> {
    entry.subscriptions.insert(&uri);
    let outcome = if is_modern(entry).await {
        reopen_stream(entry, cancellation, Mending::Reconnect)
            .await
            .and_then(|accepted| {
                if accepted.contains(&uri) {
                    Ok(())
                } else {
                    Err(MekaError::McpConnection {
                        server_name: entry.server_name.clone(),
                        message: format!("does not accept a subscription to '{uri}'"),
                    })
                }
            })
    } else {
        legacy_subscribe(entry, &uri, cancellation, Mending::Reconnect).await
    };
    if outcome.is_err() {
        entry.subscriptions.remove(&uri);
    }
    outcome
}

/// Stop asking for updates to `uri`.
pub(crate) async fn unsubscribe(
    entry: &Arc<ServerEntry>,
    uri: String,
    cancellation: &CancellationToken,
) -> Result<()> {
    entry.subscriptions.remove(&uri);
    if is_modern(entry).await {
        reopen_stream(entry, cancellation, Mending::Reconnect)
            .await
            .map(|_accepted| ())
    } else {
        legacy_unsubscribe(entry, &uri, cancellation).await
    }
}

/// Whether a request that meets a connection a fresh one would mend asks for one: a closed
/// transport, or a rejection as unauthorized under `headers_helper`. The carry-over onto a
/// connection just adopted never does. The fresh connection would carry the set over again, and a
/// server that rejects it on every connection would be reconnected to without end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mending {
    Reconnect,
    Never,
}

/// [`resume`] as a task's future, boxed under a named type: the stream it opens can, elsewhere, ask
/// for a reconnect, whose adoption spawns this again, and the future's own type would otherwise
/// contain itself.
pub(super) fn resumption(entry: Arc<ServerEntry>) -> futures::future::BoxFuture<'static, ()> {
    Box::pin(async move { resume(&entry).await })
}

/// Carry the set over onto the connection `entry` has just adopted, and open the stream a modern
/// one delivers on. What a legacy server refuses is logged: the model asked on a connection that
/// is gone, and nothing is waiting on the answer.
///
/// Runs beside the connection rather than ahead of it, so by the time it runs a shutdown may have
/// closed what it was to carry the set onto; nothing is owed then, and nothing is said.
pub(super) async fn resume(entry: &Arc<ServerEntry>) {
    let cancellation = CancellationToken::new();
    if entry.connects.stopping().is_cancelled() || entry.require_connected().await.is_err() {
        return;
    }
    if is_modern(entry).await {
        if let Err(error) = reopen_stream(entry, &cancellation, Mending::Never).await {
            tracing::debug!(
                "MCP server '{server_name}' opened no notification stream: {error}",
                server_name = entry.server_name
            );
        }
        return;
    }
    for uri in entry.subscriptions.uris() {
        if let Err(error) = legacy_subscribe(entry, &uri, &cancellation, Mending::Never).await {
            if entry.connects.stopping().is_cancelled() {
                return;
            }
            tracing::warn!(
                "failed to subscribe to '{uri}' again on MCP server '{server_name}': {error}",
                server_name = entry.server_name
            );
        }
    }
}

/// Open a stream for the set as it stands and make its pump the entry's, ending the previous pump
/// once the new stream is acknowledged so no notification falls between the two. Returns the URIs
/// the acknowledgment accepted.
async fn reopen_stream(
    entry: &Arc<ServerEntry>,
    cancellation: &CancellationToken,
    mending: Mending,
) -> Result<BTreeSet<String>> {
    // The wait for the lock is the caller's to stop too: whoever holds it is inside a bounded
    // request, and a tool call stopped by the user should not sit that bound out first.
    let mut pump = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
        pump = entry.subscriptions.pump.lock() => pump,
    };
    let subscription = open_stream(entry, cancellation, mending).await?;
    tracing::debug!(
        "MCP server '{server_name}' acknowledged a notification stream for {filter:?}",
        server_name = entry.server_name,
        filter = subscription.acknowledged()
    );
    let accepted = subscription
        .acknowledged()
        .resource_subscriptions
        .clone()
        .unwrap_or_default()
        .into_iter()
        .collect();
    if let Some(previous) = pump.take() {
        previous.stop.cancel();
    }
    *pump = Some(spawn_pump(entry, subscription));
    drop(pump);
    Ok(accepted)
}

/// `subscriptions/listen` for the set as it stands, acknowledged. A rejection as unauthorized on a
/// server whose headers come from `headers_helper` renews the connection, when `mending` allows,
/// and the stream is asked for once more on the new one. The adoption carries the set over too,
/// and the later of the two streams is the one the entry keeps.
async fn open_stream(
    entry: &Arc<ServerEntry>,
    cancellation: &CancellationToken,
    mending: Mending,
) -> Result<Subscription> {
    let failed = |error: rmcp::ServiceError| MekaError::McpConnection {
        server_name: entry.server_name.clone(),
        message: format!("failed to open the notification stream: {error}"),
    };
    bounded(entry, "subscriptions/listen", cancellation, async {
        let connection = entry.live_connection().await?;
        let error = match connection
            .peer
            .listen(listen_filter(&entry.subscriptions.uris()))
            .await
        {
            Ok(subscription) => return Ok(subscription),
            Err(error) => error,
        };
        let renewable = mending == Mending::Reconnect
            && entry.config().headers_helper.is_some()
            && crate::mcp::rejected_as_unauthorized(&error);
        if !renewable
            || !entry
                .renew_rejected(connection.generation, cancellation)
                .await?
        {
            return Err(failed(error));
        }
        drop(connection);
        let renewed = entry.live_connection().await?;
        renewed
            .peer
            .listen(listen_filter(&entry.subscriptions.uris()))
            .await
            .map_err(failed)
    })
    .await
}

/// The stream's filter: every list-changed kind, which the acknowledgment trims to what the
/// server advertises, and the URIs the model asked for.
fn listen_filter(uris: &[String]) -> SubscriptionFilter {
    let mut filter = SubscriptionFilter::builder()
        .tools_list_changed()
        .prompts_list_changed()
        .resources_list_changed();
    if !uris.is_empty() {
        filter = filter.resource_subscriptions(uris.iter().cloned());
    }
    filter.build()
}

fn spawn_pump(entry: &Arc<ServerEntry>, subscription: Subscription) -> Pump {
    let stop = CancellationToken::new();
    tokio::spawn(drain(Arc::clone(entry), subscription, stop.clone()));
    Pump { stop }
}

/// How a stream ended, in the words of the line that reports it, and whether the server chose to
/// end it, which is its prerogative and no cause for a warning.
struct Ended {
    what: String,
    by_the_server: bool,
}

impl Ended {
    fn of(end: Option<&SubscriptionEnd>) -> Self {
        let (what, by_the_server) = match end {
            Some(SubscriptionEnd::Lagged { capacity }) => (
                format!("sent notifications faster than meka read them ({capacity} buffered)"),
                false,
            ),
            Some(SubscriptionEnd::Graceful(_)) => {
                ("ended its notification stream".to_string(), true)
            }
            Some(SubscriptionEnd::Cancelled) => {
                ("canceled its notification stream".to_string(), true)
            }
            Some(SubscriptionEnd::Abrupt) => (
                "notification stream ended without a final result".to_string(),
                false,
            ),
            Some(other) => (format!("notification stream ended ({other:?})"), false),
            None => ("notification stream ended".to_string(), false),
        };
        Self {
            what,
            by_the_server,
        }
    }
}

/// Feed one stream's notifications where the handler feeds a legacy server's, until `stop` says a
/// later stream has taken over. A stream that ends while its connection lives, whether the server
/// closed it, meka fell behind on it, or a proxy cut the long request it rides on, is replaced by
/// another for the set as it now stands: the server sends nothing again until asked. One that
/// ended with its connection is left to the next connection, whose adoption opens its own.
async fn drain(entry: Arc<ServerEntry>, mut subscription: Subscription, stop: CancellationToken) {
    let mut backoff = REOPEN_INITIAL_BACKOFF;
    let mut opened = tokio::time::Instant::now();
    let mut announced: Option<String> = None;
    loop {
        let next = tokio::select! {
            biased;
            _ = stop.cancelled() => {
                if let Err(error) = subscription.cancel().await {
                    tracing::debug!(
                        "failed to cancel MCP server '{server_name}' notification stream: {error}",
                        server_name = entry.server_name
                    );
                }
                return;
            }
            next = subscription.next() => next,
        };
        let ended = match next {
            Ok(Some(notification)) => {
                // On a task of its own, as rmcp runs a legacy server's notification handlers: a
                // tool refresh is a request round trip, and a stream read behind one for its
                // length fills and ends the stream.
                tokio::spawn(dispatch(Arc::clone(&entry), notification));
                continue;
            }
            Ok(None) => Ended::of(subscription.end()),
            Err(error) => Ended {
                what: format!("notification stream failed: {error}"),
                by_the_server: false,
            },
        };
        backoff = backoff_before_reopen(backoff, opened.elapsed());
        let Some(reopened) = reopen(&entry, &stop, &mut backoff, &mut announced, &ended).await
        else {
            return;
        };
        subscription = reopened;
        opened = tokio::time::Instant::now();
    }
}

/// Whether a stream the entry's connection carried can be opened again on it: the connection is
/// still the entry's, still open, and still of the era that has streams.
async fn stream_can_be_reopened(entry: &ServerEntry) -> bool {
    is_modern(entry).await
        && entry
            .require_connected()
            .await
            .is_ok_and(|peer| !peer.is_transport_closed())
}

/// Another stream for the set as it now stands, opened after `backoff` for as long as the
/// connection lives and this pump is still the entry's; `backoff` doubles with every attempt,
/// whether it opens a stream or not. `None` when the connection is gone, whose successor opens
/// its own stream on adoption, or a later stream has its own pump.
///
/// The wait comes before the first look at the connection: a stream that ended with its
/// connection is told apart from one the server ended only once the transport has had a moment
/// to close behind it. An end and a refusal are each said once per cause, as a connect failure
/// is: the first end of a kind, the first refusal and a changed one are warnings, and the repeats
/// are not, so a proxy that cuts every stream at its idle timeout is reported once; the stream
/// opening again after a refusal is news. `announced` is the last end warned about.
async fn reopen(
    entry: &Arc<ServerEntry>,
    stop: &CancellationToken,
    backoff: &mut Duration,
    announced: &mut Option<String>,
    ended: &Ended,
) -> Option<Subscription> {
    let server_name = entry.server_name.as_str();
    let mut last_failure: Option<String> = None;
    loop {
        tokio::select! {
            biased;
            _ = stop.cancelled() => return None,
            _ = tokio::time::sleep(*backoff) => {}
        }
        *backoff = next_backoff(*backoff);
        if !stream_can_be_reopened(entry).await {
            tracing::debug!(
                "MCP server '{server_name}' {what}; the next connection opens another stream",
                what = ended.what
            );
            return None;
        }
        if last_failure.is_none() {
            if ended.by_the_server || announced.as_deref() == Some(ended.what.as_str()) {
                tracing::debug!(
                    "MCP server '{server_name}' {what}; opening another stream",
                    what = ended.what
                );
            } else {
                tracing::warn!(
                    "MCP server '{server_name}' {what}; opening another stream",
                    what = ended.what
                );
                *announced = Some(ended.what.clone());
            }
        }
        // Under the pump lock, as a re-issue is: a stream opened for a newer set cancels this
        // pump's token under that lock, so a pump whose token is live here is still the entry's,
        // and the stream it opens is the one the entry drains.
        let pump = entry.subscriptions.pump.lock().await;
        if stop.is_cancelled() {
            return None;
        }
        let opened = open_stream(entry, &CancellationToken::new(), Mending::Reconnect).await;
        drop(pump);
        match opened {
            Ok(subscription) => {
                if last_failure.is_some() {
                    tracing::info!("MCP server '{server_name}' notification stream is open again");
                }
                return Some(subscription);
            }
            Err(error) => {
                let failure = error.to_string();
                if last_failure.as_deref() == Some(failure.as_str()) {
                    tracing::debug!(
                        "MCP server '{server_name}' notification stream still not open: {failure}"
                    );
                } else {
                    tracing::warn!(
                        "failed to open another notification stream to MCP server \
                         '{server_name}': {failure}"
                    );
                }
                last_failure = Some(failure);
            }
        }
    }
}

async fn dispatch(entry: Arc<ServerEntry>, notification: ServerNotification) {
    match notification {
        ServerNotification::ToolListChangedNotification(_) => {
            handler::refresh_tools_after_list_changed(&entry.server_name, &entry.client_context)
                .await;
        }
        ServerNotification::ResourceUpdatedNotification(updated) => {
            handler::record_resource_update(
                &entry.server_name,
                &entry.client_context,
                &updated.params.uri,
            );
        }
        ServerNotification::ResourceListChangedNotification(_) => {
            tracing::debug!(
                "MCP server '{server_name}' sent resources/list_changed",
                server_name = entry.server_name
            );
        }
        ServerNotification::PromptListChangedNotification(_) => {
            tracing::debug!(
                "MCP server '{server_name}' sent prompts/list_changed",
                server_name = entry.server_name
            );
        }
        other => {
            tracing::debug!(
                "MCP server '{server_name}' sent {other:?} on its notification stream; ignored",
                server_name = entry.server_name
            );
        }
    }
}

// `resources/subscribe` and `resources/unsubscribe` are the mechanism of the era before
// `subscriptions/listen`, and a connection of that era has no other: rmcp deprecates the pair in
// favor of `Peer::listen`, which a legacy server never negotiated.
#[allow(
    deprecated,
    reason = "the one mechanism a legacy connection has; see above"
)]
async fn legacy_subscribe(
    entry: &Arc<ServerEntry>,
    uri: &str,
    cancellation: &CancellationToken,
    mending: Mending,
) -> Result<()> {
    bounded(entry, "resources/subscribe", cancellation, async {
        let subscribe = |peer: rmcp::Peer<rmcp::RoleClient>| async move {
            peer.subscribe(SubscribeRequestParams::new(uri.to_string()))
                .await
        };
        let subscribed = match mending {
            Mending::Reconnect => entry.on_a_connection(cancellation, subscribe).await?,
            Mending::Never => {
                let connection = entry.live_connection().await?;
                subscribe(connection.peer.clone()).await
            }
        };
        subscribed.map_err(|error| MekaError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("failed to subscribe to '{uri}': {error}"),
        })
    })
    .await
}

#[allow(
    deprecated,
    reason = "the one mechanism a legacy connection has; see `legacy_subscribe`"
)]
async fn legacy_unsubscribe(
    entry: &Arc<ServerEntry>,
    uri: &str,
    cancellation: &CancellationToken,
) -> Result<()> {
    bounded(entry, "resources/unsubscribe", cancellation, async {
        entry
            .on_a_connection(cancellation, |peer| async move {
                peer.unsubscribe(UnsubscribeRequestParams::new(uri.to_string()))
                    .await
            })
            .await?
            .map_err(|error| MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("failed to unsubscribe from '{uri}': {error}"),
            })
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The filter asks for every list-changed kind whatever the set holds, since the stream is
    /// how a modern server's tool changes reach meka at all, and names the URIs only when there
    /// are some: an empty `resourceSubscriptions` is a request a server may answer with nothing.
    #[test]
    fn the_filter_asks_for_every_change_kind_and_the_uris_it_was_given() {
        let bare = listen_filter(&[]);
        assert_eq!(bare.tools_list_changed, Some(true));
        assert_eq!(bare.prompts_list_changed, Some(true));
        assert_eq!(bare.resources_list_changed, Some(true));
        assert_eq!(bare.resource_subscriptions, None);

        let watching = listen_filter(&["file:///a".to_string(), "file:///b".to_string()]);
        assert_eq!(watching.tools_list_changed, Some(true));
        assert_eq!(
            watching.resource_subscriptions,
            Some(vec!["file:///a".to_string(), "file:///b".to_string()])
        );
    }

    /// The reopen schedule doubles from one second to thirty across streams that end within
    /// thirty seconds of opening, and starts over after one that lived longer.
    #[test]
    fn the_reopen_backoff_doubles_to_its_bound_and_starts_over_after_a_long_lived_stream() {
        assert_eq!(next_backoff(Duration::from_secs(1)), Duration::from_secs(2));
        assert_eq!(
            next_backoff(Duration::from_secs(16)),
            Duration::from_secs(30)
        );
        assert_eq!(
            next_backoff(Duration::from_secs(30)),
            Duration::from_secs(30)
        );
        assert_eq!(
            backoff_before_reopen(Duration::from_secs(8), Duration::from_secs(5)),
            Duration::from_secs(8)
        );
        assert_eq!(
            backoff_before_reopen(Duration::from_secs(8), Duration::from_secs(30)),
            Duration::from_secs(1)
        );
    }

    #[test]
    fn the_set_is_ordered_and_deduplicated() {
        let subscriptions = Subscriptions::default();
        subscriptions.insert("file:///b");
        subscriptions.insert("file:///a");
        subscriptions.insert("file:///b");
        assert_eq!(subscriptions.uris(), vec!["file:///a", "file:///b"]);
        subscriptions.remove("file:///a");
        assert_eq!(subscriptions.uris(), vec!["file:///b"]);
    }

    /// The server's own end of a stream is its prerogative and reads as such; a stream cut
    /// without a final result, or one meka fell behind on, reads as what happened to it, never as
    /// a debug rendering of the enum.
    #[test]
    fn an_end_is_told_in_words_and_the_server_s_own_is_no_warning() {
        let graceful = Ended::of(Some(&SubscriptionEnd::Graceful(
            rmcp::model::SubscriptionsListenResult::complete(rmcp::model::RequestId::Number(1)),
        )));
        assert!(graceful.by_the_server);
        let canceled = Ended::of(Some(&SubscriptionEnd::Cancelled));
        assert!(canceled.by_the_server);
        let abrupt = Ended::of(Some(&SubscriptionEnd::Abrupt));
        assert!(!abrupt.by_the_server);
        assert_eq!(
            abrupt.what,
            "notification stream ended without a final result"
        );
        let lagged = Ended::of(Some(&SubscriptionEnd::Lagged { capacity: 64 }));
        assert!(!lagged.by_the_server);
        assert!(lagged.what.contains("64 buffered"), "{}", lagged.what);
        for ended in [&graceful, &canceled, &abrupt, &lagged] {
            assert!(!ended.what.contains("Some("), "{}", ended.what);
        }
    }
}
