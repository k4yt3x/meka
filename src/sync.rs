//! Locks that outlive a panic. A holder that panicked poisons a `std` lock, and refusing every
//! later reader would turn one bug into a dead process; the value is recovered and the site is
//! logged so the panic that caused it can be found. Every `std` lock in the tree is taken through
//! here, so the policy exists once.

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Lock `mutex`, recovering the value if a previous holder panicked.
#[track_caller]
pub(crate) fn lock<'a, T>(mutex: &'a Mutex<T>) -> MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!(
                "mutex poisoned at {caller}; recovering the value",
                caller = std::panic::Location::caller()
            );
            poisoned.into_inner()
        }
    }
}

/// Take `rwlock` for reading, recovering the value if a previous holder panicked.
#[track_caller]
pub(crate) fn read<'a, T>(rwlock: &'a RwLock<T>) -> RwLockReadGuard<'a, T> {
    match rwlock.read() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!(
                "rwlock poisoned (read) at {caller}; recovering the value",
                caller = std::panic::Location::caller()
            );
            poisoned.into_inner()
        }
    }
}

/// Take `rwlock` for writing, recovering the value if a previous holder panicked.
#[track_caller]
pub(crate) fn write<'a, T>(rwlock: &'a RwLock<T>) -> RwLockWriteGuard<'a, T> {
    match rwlock.write() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!(
                "rwlock poisoned (write) at {caller}; recovering the value",
                caller = std::panic::Location::caller()
            );
            poisoned.into_inner()
        }
    }
}
