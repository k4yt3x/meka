//! Logs `std::sync::Mutex` / `RwLock` poisoning instead of silently recovering. agsh's server
//! code holds these synchronous locks for microsecond-scale critical sections that never
//! `.await` (poisoning requires a panic while holding the guard, which we don't expect), but
//! "shouldn't happen" isn't "won't happen". If a panic does poison a lock the recovered guard
//! still works — `into_inner()` exposes the inner value — but the underlying panic is
//! invisible to operators. These helpers log a single high-priority line on each recovery so
//! the root cause shows up in observability tooling.

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Acquire a [`Mutex`] guard, logging at `error!` level if the lock was poisoned. `site` is a
/// short `&'static str` identifying the call site (e.g., `"http_frontend::stream"`) so logs
/// pinpoint *which* lock was affected without exposing private types.
pub fn lock<'a, T>(mutex: &'a Mutex<T>, site: &'static str) -> MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!("mutex poisoned at {}; recovering inner value", site);
            poisoned.into_inner()
        }
    }
}

/// Acquire an [`RwLock`] read guard with poisoning recovery + logging.
pub fn read<'a, T>(rwlock: &'a RwLock<T>, site: &'static str) -> RwLockReadGuard<'a, T> {
    match rwlock.read() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!("rwlock poisoned (read) at {}; recovering inner value", site);
            poisoned.into_inner()
        }
    }
}

/// Acquire an [`RwLock`] write guard with poisoning recovery + logging.
pub fn write<'a, T>(rwlock: &'a RwLock<T>, site: &'static str) -> RwLockWriteGuard<'a, T> {
    match rwlock.write() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!(
                "rwlock poisoned (write) at {}; recovering inner value",
                site
            );
            poisoned.into_inner()
        }
    }
}
