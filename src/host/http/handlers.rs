//! HTTP request handlers, grouped by resource. Each submodule exports `async fn` handlers that
//! `server::run_serve` wires into the axum `Router`.

pub(crate) mod conversation;
pub(crate) mod discovery;
pub(crate) mod info;
pub(crate) mod jobs;
pub(crate) mod messages;
pub(crate) mod responses;
pub(crate) mod sessions;
pub(crate) mod stores;
pub(crate) mod turn;
