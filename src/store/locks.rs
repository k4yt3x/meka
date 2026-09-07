//! The store's own file locks: the per-session claim, the lock directory beside the database,
//! and the sweep that removes lock files nothing holds.

use super::*;

/// Removes an in-memory database's temp lock directory on drop.
///
/// On-disk databases keep a `locks/` directory beside the database file, which must outlive the
/// process; in-memory ones get a per-`open()` temp directory instead, so that concurrent tests
/// can't sweep each other's lock files through [`Store::prune_orphan_lock_files`]. Without this
/// guard each `open()` leaves an empty directory in the system temp dir forever.
pub(super) struct EphemeralLockDir(pub(super) PathBuf);
impl Drop for EphemeralLockDir {
    fn drop(&mut self) {
        // Any `.lock` files still inside belong to this store's own sessions; a `FileLock`
        // that outlives the store keeps working, because unlinking an open file on Unix doesn't
        // invalidate the descriptor the `flock` is held on.
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            tracing::debug!(
                "failed to remove ephemeral lock dir '{path}': {error}",
                path = self.0.display(),
            );
        }
    }
}
/// Where a session's lock lives while a host and its agent both need to reach it.
///
/// The lock has to be taken the instant the row exists, and for a session the agent creates that
/// instant is inside `Agent::run_turn_retaining`, seconds or minutes before the host gets control
/// back. Claiming it afterwards leaves a fresh session unlocked for the whole of its first turn, so
/// a second `meka` invocation attaches to it and interleaves its own messages into the same
/// conversation.
///
/// The host still owns the lifetime: the lock outlives any one turn, `/fork` replaces it, and it is
/// dropped last on the way out so the session stays held until the process really ends. Sharing a
/// slot rather than moving the lock either way is what lets both of those be true at once.
///
/// `std::sync::Mutex` rather than tokio's: every access is a move in or out with nothing awaited in
/// between, and a blocking lock keeps the drop order at process exit obvious.
pub(crate) type SessionLockSlot = std::sync::Arc<std::sync::Mutex<Option<FileLock>>>;
/// File stem of the process-wide schema lock, which lives beside the per-session ones.
///
/// Named rather than spelled inline at both sites, because the prune below has to exclude exactly
/// this file and a literal in two places is how that pairing gets broken.
pub(super) const SCHEMA_LOCK_STEM: &str = "schema";
/// File-stem prefix of the per-account credential locks, which also live in the lock directory.
///
/// Excluded from the prune for the same reason as [`SCHEMA_LOCK_STEM`]: a hashed account name does
/// not parse as a UUID, and relying on that accident is how the next lock shaped like one inherits
/// the hazard.
pub(super) const ACCOUNT_LOCK_PREFIX: &str = "account-";

impl Store {
    /// Acquire an exclusive OS file lock on the session. Returns a [`FileLock`] handle whose
    /// lifetime owns the lock; drop it (or let the process exit) to release.
    ///
    /// The session must already exist in the database. Returns [`MekaError::SessionLocked`] if
    /// another live process holds the lock.
    pub(crate) fn lock_session(&self, session_id: Uuid) -> Result<FileLock> {
        let path = self.lock_dir.join(format!("{session_id}.lock"));
        try_lock_file(&path)?.ok_or(MekaError::SessionLocked(session_id))
    }

    /// Best-effort removal of `<lock_dir>/<uuid>.lock` files whose session no longer exists.
    /// `lock_session` creates these files and never deletes them: the OS releases the *lock* on
    /// process exit and the empty file remains.
    ///
    /// Housekeeping only: never fails the caller. A database failure is a `warn!`; a per-file
    /// unlink failure (a root-owned file left by a container run) is expected and a `debug!`.
    ///
    /// **Nothing is unlinked without first taking its lock.** A missing row does not prove the file
    /// is garbage: this `SELECT` can finish before a session's row commits while the `read_dir`
    /// below runs after that session's lock file exists, and [`Self::create_session_locked`] takes
    /// the lock *before* the insert. Unlinking does not release a held `flock`, so a wrong guess
    /// leaves two processes holding locks on different inodes, neither able to see the other, both
    /// writing turns into one session. On Windows `LockFileEx` will not let an open file be
    /// unlinked, so the same mistake makes `remove_file` fail instead; proving the file is unheld
    /// is the right answer on both. `session_exists` runs first because it is the cheaper question
    /// and the common answer is "the row is there, leave it alone".
    ///
    /// One window is accepted rather than closed. [`try_lock_file`] opens and then locks, two
    /// syscalls with nothing between, and a sweep that `read_dir`s inside that gap can take a
    /// creator's lock and unlink its file; the creator then commits a row with nothing on disk to
    /// claim it, and the next sweep re-creates the path, locks it, and deletes the row mid-turn. It
    /// needs the sweep running at that instant, which happens only at [`Store::open`] and after a
    /// delete, with a `session_exists` round trip between the `read_dir` and the `flock`.
    pub(super) async fn prune_orphan_lock_files(&self) {
        let live_ids: std::collections::HashSet<String> = match self
            .connection
            .call(|connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare("SELECT id FROM sessions")?;
                let ids = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<std::collections::HashSet<String>>>()?;
                Ok(ids)
            })
            .await
        {
            Ok(ids) => ids,
            Err(error) => {
                tracing::warn!("lock-file prune: failed to list sessions: {error}");
                return;
            }
        };

        let entries = match std::fs::read_dir(&self.lock_dir) {
            Ok(entries) => entries,
            Err(error) => {
                tracing::debug!(
                    "lock-file prune: failed to read {lock_dir}: {error}",
                    lock_dir = self.lock_dir.display()
                );
                return;
            }
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    tracing::debug!(
                        "lock-file prune: failed to read an entry of {lock_dir}: {error}",
                        lock_dir = self.lock_dir.display()
                    );
                    continue;
                }
            };
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("lock") {
                continue;
            }
            // Only touch files whose stem is a UUID; never delete an unrelated file someone
            // dropped into the lock directory.
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            // `schema.lock` shares this directory and is not a session's. It survives the UUID
            // check below by accident ("schema" does not parse as one), and the accident is
            // worth not relying on: unlinking a file does not release a held `flock`, so a swept
            // schema lock would let the next process create a different inode, and both would enter
            // `memory::store`'s FTS-trigger reconciliation believing they held it, each deciding
            // from a `sqlite_master` read the other is free to invalidate before the decision is
            // acted on. Named explicitly so a future lock called anything UUID-shaped does not
            // quietly inherit the hazard.
            if stem == SCHEMA_LOCK_STEM || stem.starts_with(ACCOUNT_LOCK_PREFIX) {
                continue;
            }
            let Ok(id) = Uuid::parse_str(stem) else {
                continue;
            };
            if live_ids.contains(stem) {
                continue;
            }
            // Re-read, because the snapshot was taken before `read_dir` and a row committed since
            // would not be in it. Before the lock rather than after, so a creator that has opened
            // this file and not yet locked it is not raced for it. Cheap: only a genuinely
            // orphaned file gets this far, and `unwrap_or(true)` keeps a failed read on the
            // sparing side.
            if self.session_exists(id).await.unwrap_or(true) {
                continue;
            }
            // The claim, not a guess about one. A file whose lock is held belongs to a live
            // process whatever the `sessions` table says about its row.
            let Ok(claim) = self.lock_session(id) else {
                continue;
            };
            // Released before the unlink so the descriptor this process holds is not the one being
            // removed from under it.
            drop(claim);
            if let Err(error) = std::fs::remove_file(&path) {
                tracing::debug!(
                    "lock-file prune: failed to remove {path}: {error}",
                    path = path.display()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every in-memory `open()` mints a temp lock dir; unremoved, a few hundred `cargo test` runs
    /// leave thousands of stray directories in the system temp dir.
    #[tokio::test]
    async fn in_memory_lock_dir_is_removed_when_the_last_clone_drops() {
        let store = Store::for_test().await;
        let lock_dir = store.lock_dir.clone();
        assert!(lock_dir.exists(), "open() should have created the lock dir");

        // A clone still holding it must keep the directory alive: sub-agents and tool builders
        // clone the store, and a lock taken through one of them has to keep working.
        let clone = store.clone();
        drop(store);
        assert!(
            lock_dir.exists(),
            "the dir must outlive the first clone to drop"
        );

        drop(clone);
        assert!(
            !lock_dir.exists(),
            "the last clone dropping should remove '{}'",
            lock_dir.display()
        );
    }

    /// The on-disk counterpart must survive: its `locks/` directory lives beside the database and
    /// outlives the process.
    #[tokio::test]
    async fn on_disk_lock_dir_survives_the_manager() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("meka.db");
        let store = Store::open(Some(&db_path), &Default::default())
            .await
            .expect("open on-disk database");
        let lock_dir = store.lock_dir.clone();
        drop(store);
        assert!(
            lock_dir.exists(),
            "an on-disk lock dir must not be swept away with the store"
        );
    }

    #[tokio::test]
    async fn session_locking_acquire_and_release() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");

        let lock = store
            .lock_session(session_id)
            .expect("failed to lock session");

        // While the lock handle is alive, a second attempt must fail.
        match store.lock_session(session_id) {
            Err(MekaError::SessionLocked(id)) => assert_eq!(id, session_id),
            other => panic!("expected SessionLocked, got {:?}", other.map(|_| "Ok(_)")),
        }

        // Dropping the handle releases the OS lock; re-acquisition succeeds.
        drop(lock);
        let _lock2 = store
            .lock_session(session_id)
            .expect("failed to re-acquire session lock after drop");
    }

    #[tokio::test]
    async fn the_prune_removes_orphan_lock_files_and_leaves_live_and_stray_ones() {
        let store = Store::for_test().await;
        let live = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let live_lock = store.lock_dir.join(format!("{live}.lock"));
        let orphan_lock = store.lock_dir.join(format!("{}.lock", Uuid::new_v4()));
        let stray = store.lock_dir.join("not-a-uuid.lock");
        std::fs::write(&live_lock, "").expect("write live lock");
        std::fs::write(&orphan_lock, "").expect("write orphan lock");
        std::fs::write(&stray, "").expect("write stray file");

        store.prune_orphan_lock_files().await;

        assert!(live_lock.exists(), "live session's lock file must be kept");
        assert!(!orphan_lock.exists(), "orphan lock file must be removed");
        assert!(stray.exists(), "non-UUID file must be left untouched");
    }

    /// The prune's own question, asked of the lock rather than inferred from a snapshot.
    ///
    /// The old rule -- unlink any file whose id is not in the live set -- was justified by a
    /// session's row being committed before its lock file is acquired. That constrains the
    /// creator, not the sweeper: the `SELECT` can finish before a row commits while the `read_dir`
    /// runs after that session's lock file exists. Against a running `meka serve`, 21 of 401 live
    /// sessions lost their lock file, after which a second process attached to a session `serve`
    /// still held and wrote a whole turn into it.
    ///
    /// The id here is deliberately absent from `sessions`, which is exactly what a row-only rule
    /// would call an orphan. Holding the lock is the only thing standing between it and deletion.
    #[tokio::test]
    async fn prune_spares_a_lock_file_that_is_held() {
        let store = Store::for_test().await;
        let unrecorded = Uuid::new_v4();
        let path = store.lock_dir.join(format!("{unrecorded}.lock"));
        let held = store.lock_session(unrecorded).expect("take the lock");

        store.prune_orphan_lock_files().await;
        assert!(
            path.exists(),
            "a file whose lock is held belongs to a live process, whatever the sessions table says"
        );

        // And the same file, once nobody holds it, is the garbage the sweep exists for.
        drop(held);
        store.prune_orphan_lock_files().await;
        assert!(!path.exists(), "an unheld orphan is still swept");
    }

    #[tokio::test]
    async fn open_prunes_orphan_lock_files() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("meka.db");
        let lock_dir = temp_dir.path().join("locks");
        std::fs::create_dir_all(&lock_dir).expect("create locks dir");
        let orphan = lock_dir.join(format!("{}.lock", Uuid::new_v4()));
        std::fs::write(&orphan, "").expect("write orphan");

        // A fresh DB has no sessions, so the planted file is an orphan.
        let _manager = Store::open(Some(&db_path), &Default::default())
            .await
            .expect("open should succeed");
        assert!(
            !orphan.exists(),
            "open() must prune pre-existing orphan lock files"
        );
    }

    /// An in-memory store takes its lock directory with it when the last handle drops.
    ///
    /// On-disk stores keep `locks/` beside the database, where it belongs; an in-memory one gets a
    /// per-`open()` temp directory so concurrent tests cannot sweep each other's lock files. That
    /// isolation is only free if something removes it, and nothing else ever will -- a developer's
    /// temp directory otherwise collects one empty directory per store opened, forever.
    #[tokio::test]
    async fn an_in_memory_store_removes_its_temp_lock_directory() {
        let lock_dir = {
            let store = Store::for_test().await;
            let lock_dir = store.lock_dir.clone();
            assert!(
                lock_dir.is_dir(),
                "the store locks somewhere while it is open: {}",
                lock_dir.display()
            );
            // A handle taken from the store keeps the directory alive, which is the whole reason
            // the guard is behind an `Arc` rather than owned outright.
            let token_store = store.token_store();
            drop(store);
            assert!(
                lock_dir.is_dir(),
                "a live `TokenStore` still locks under it: {}",
                lock_dir.display()
            );
            drop(token_store);
            lock_dir
        };
        assert!(
            !lock_dir.exists(),
            "the last handle is gone, so nothing is left behind: {}",
            lock_dir.display()
        );
    }
}
