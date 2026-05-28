//! HTTP request handlers, grouped by resource. Each submodule exports `async fn` handlers that
//! `server::run_serve` wires into the axum `Router`.

pub mod discovery;
pub mod info;
pub mod messages;
pub mod responses;
pub mod sessions;
pub mod turn;
