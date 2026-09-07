//! The copy taken before a migration, and the pruning of older copies.

use super::*;

/// Copy the store aside before [`migrations::apply`] touches it, returning where it went.
///
/// `VACUUM INTO` rather than a file copy: it is atomic against concurrent writers, produces a
/// defragmented image rather than one that may span a live WAL, and carries `user_version` across,
/// so restoring the copy yields a store that identifies itself as the version it was and migrates
/// once when next opened. Lives here rather than in [`migrations`] because it is a question about
/// the store *file* -- a path, a mode -- and because that module is deliberately kept clear of
/// meka's own code.
///
/// Kept rather than cleaned up on success. It is the user's undo, and deleting it the moment the
/// migration works removes the safety net exactly when they might still want it.
///
/// Written to a staging name and renamed into place, so the name the docs tell people to restore
/// either does not exist or is a complete copy. `VACUUM INTO` does not unlink its output when a
/// write fails partway: measured under a write limit, it left a file with a zeroed page-1 header,
/// which SQLite then opens cleanly as an *empty database* that passes `integrity_check`. The retry
/// stepped past it to `.bak.1`, so the file wearing the documented name was the empty one. Staging
/// plus rename removes that whole class rather than special-casing it.
pub(super) fn back_up_before_migrating(
    connection: &rusqlite::Connection,
    database_path: &Path,
    from: u32,
) -> Result<Option<PathBuf>> {
    // Nothing to preserve: an in-memory store is created empty by this very process.
    if database_path == Path::new(":memory:") {
        return Ok(None);
    }
    let target = free_backup_path(database_path, from)?;
    let staging = staging_path(&target);
    // `create_new` rather than `create`: it fails rather than following a symlink or truncating
    // something already there, and it is what makes the mode below a guarantee instead of a hope.
    // The mode matters because this file holds every credential and every conversation the store
    // does, and SQLite would otherwise create it at `0644 & ~umask` and leave it that way for as
    // long as the copy takes. `Store::open` pre-touches the main database at `0600` for
    // exactly this reason; the backup gets the same treatment.
    let mut options = std::fs::OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(&staging).map_err(|error| {
        MekaError::Database(format!(
            "failed to create the pre-migration backup at '{}': {}. Nothing has been changed",
            staging.display(),
            error
        ))
    })?;

    // Bound rather than interpolated, so a data directory containing a quote is not a broken
    // statement. `VACUUM INTO` accepts a parameter here, and writes into the empty file just made.
    let Some(staging_text) = staging.to_str() else {
        remove_partial_backup(&staging);
        return Err(MekaError::Database(format!(
            "cannot write the pre-migration backup to '{}' because the path is not valid UTF-8. \
             Nothing has been changed. Move the store somewhere it is, or set MEKA_DATA_DIR",
            staging.display()
        )));
    };
    if let Err(error) = connection.execute("VACUUM INTO ?1", rusqlite::params![staging_text]) {
        remove_partial_backup(&staging);
        return Err(MekaError::Database(format!(
            "failed to back the store up to '{}' before migrating: {}. Nothing has been changed",
            target.display(),
            error
        )));
    }
    // Belt-and-braces on platforms where the mode above is a no-op, and against a umask that
    // somehow widened it.
    restrict_permissions(&staging, 0o600);
    std::fs::rename(&staging, &target).map_err(|error| {
        remove_partial_backup(&staging);
        MekaError::Database(format!(
            "failed to move the pre-migration backup into place at '{target}': {error}. Nothing \
             has been changed",
            target = target.display(),
        ))
    })?;
    Ok(Some(target))
}
/// Remove the copies an earlier migration left, now that a fresher one is in place.
///
/// **Production reaches this only with a path [`back_up_before_migrating`] returned**, after the
/// rename has put a complete copy at that name; pruning before the copy landed would leave no
/// backup at all when the copy then fails (a path that is not valid UTF-8 is refused at
/// `staging.to_str()`).
///
/// One backup rather than one per release: the store holds every conversation and every memory, so
/// a copy per schema-changing upgrade is a full duplicate accumulating forever in a directory
/// nobody opens. The cost is that the copy kept is of the store *after* the migration before this
/// one, so a step that silently mis-converts and is followed by another schema-changing upgrade
/// takes the only pre-conversion copy with it. `upgrading.md` says the same to users.
///
/// Only names this module could have produced are touched; see [`is_backup_name`], which matches
/// what [`free_backup_path`] emits rather than "some digits". A `meka.db.mine.bak` someone parked
/// beside the store is not meka's to delete. Names are compared through
/// [`std::ffi::OsStr::as_encoded_bytes`] rather than `to_string_lossy`, because lossy conversion
/// maps distinct names onto one string; every byte tested is ASCII, which cannot occur inside a
/// multi-byte sequence, so splitting on it is sound on both platforms.
///
/// Every failure is a `warn!` and nothing more, so a filesystem that will not let a file go cannot
/// turn a migration that worked into a refusal to start.
pub(super) fn prune_older_backups(database_path: &Path, keep: &Path) {
    let (Some(directory), Some(store_name), Some(keep_name)) = (
        database_path.parent(),
        database_path.file_name(),
        keep.file_name(),
    ) else {
        return;
    };
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(
                "failed to list '{directory}' to remove older pre-migration backups: {error}",
                directory = directory.display()
            );
            return;
        }
    };
    for entry in entries {
        // Not `.flatten()`. An entry that cannot be read is a question this cannot answer, and
        // skipping it is right -- but silently is not, because the answer it stands in for is
        // "there may be a superseded copy still on disk".
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(
                    "failed to read an entry of '{directory}' while removing older pre-migration backups: \
                     {error}",
                    directory = directory.display()
                );
                continue;
            }
        };
        let name = entry.file_name();
        if name == keep_name || !is_backup_name(store_name, &name) {
            continue;
        }
        let path = entry.path();
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!(
                "removed the superseded pre-migration backup '{path}'",
                path = path.display()
            ),
            Err(error) => tracing::warn!(
                "failed to remove the superseded pre-migration backup '{path}': {error}",
                path = path.display()
            ),
        }
    }
}
/// Whether `candidate` is a name [`free_backup_path`] could have produced for `store_name`.
///
/// `<store>.v<version>.bak`, optionally then `.<suffix>`. Both numbers are matched against what
/// that function can actually emit rather than against "some digits", because the two differ and
/// the difference is deletions: a `version` it renders through `Display` from a `u32` it is only
/// ever called with above zero, and a `suffix` from `1..1000`. So `.v0.`, `.v01.`, `.bak.0` and
/// `.bak.1000` are all shapes meka cannot write, and a `meka.db.v1.bak.0` is somebody's own
/// zero-indexed archive rather than a copy this module left.
///
/// **One shape, tracking the writer.** If [`free_backup_path`] ever names copies differently, this
/// follows it and matches only the new name. Keeping the old one alongside would be a reader
/// tolerating what an older meka wrote, which is the arrangement `migrations` exists to make
/// unnecessary, and there is no ledger for files sitting beside the store to convert them with. The
/// honest price of such a rename is that copies already on disk stop being recognized and are left
/// where they are, which errs toward keeping a file rather than deleting one.
///
/// A `.partial` is deliberately not a match. It is litter as well, but [`free_backup_path`] reads
/// its presence to step past a name it would otherwise reuse, and removing completed copies is the
/// whole of what this is for.
pub(super) fn is_backup_name(store_name: &std::ffi::OsStr, candidate: &std::ffi::OsStr) -> bool {
    let store = store_name.as_encoded_bytes();
    let candidate = candidate.as_encoded_bytes();
    let Some(rest) = candidate.strip_prefix(store) else {
        return false;
    };
    let Some(rest) = rest.strip_prefix(b".v") else {
        return false;
    };
    let (version, rest) = split_digits(rest);
    // `from > 0` is guarded at the one call site, and `Display` never pads, so a leading zero and a
    // bare `0` are both names this module cannot have written.
    if !matches!(parse_number(version), Some(1..=u32::MAX)) {
        return false;
    }
    let Some(rest) = rest.strip_prefix(b".bak") else {
        return false;
    };
    if rest.is_empty() {
        return true;
    }
    let Some(rest) = rest.strip_prefix(b".") else {
        return false;
    };
    let (suffix, tail) = split_digits(rest);
    tail.is_empty() && matches!(parse_number(suffix), Some(1..MAX_BACKUP_SUFFIX))
}
/// The exclusive end of [`free_backup_path`]'s suffix range, named so the loop there and the match
/// in [`is_backup_name`] cannot drift apart.
pub(super) const MAX_BACKUP_SUFFIX: u32 = 1000;
/// Parse a run of ASCII digits exactly as `free_backup_path` rendered one, or `None`.
///
/// Rejects what `Display` cannot produce (nothing at all, a leading zero) and what a `u32` cannot
/// hold, so an absurdly long run of digits is a non-match rather than a wrap or a panic.
pub(super) fn parse_number(digits: &[u8]) -> Option<u32> {
    if digits.is_empty() || (digits.len() > 1 && digits[0] == b'0') {
        return None;
    }
    std::str::from_utf8(digits).ok()?.parse().ok()
}
/// Split a leading run of ASCII digits from the rest, for [`is_backup_name`]. Either half may be
/// empty; the caller decides which of those are acceptable.
pub(super) fn split_digits(bytes: &[u8]) -> (&[u8], &[u8]) {
    let end = bytes
        .iter()
        .position(|byte| !byte.is_ascii_digit())
        .unwrap_or(bytes.len());
    bytes.split_at(end)
}
/// Clear away a backup that never finished. Best-effort by design: the caller is already returning
/// an error that stops the migration, and failing to tidy up must not replace that error with a
/// less useful one. Leaving it behind is safe either way, because only a completed copy is ever
/// renamed onto the name the docs name.
pub(super) fn remove_partial_backup(staging: &Path) {
    if let Err(error) = std::fs::remove_file(staging) {
        tracing::debug!(
            "failed to remove the incomplete backup '{staging}': {error}",
            staging = staging.display()
        );
    }
}
/// `<store>.v<from>.bak`, or the first numbered variant whose name *and staging sibling* are both
/// unused.
///
/// Built through `OsString` rather than `format!` on a `String`, so a data directory whose name is
/// not valid UTF-8 still produces a path that names the file the user actually has.
///
/// Never reuses a name that exists. An earlier backup is the record of an earlier attempt, and a
/// migration that replaced it would destroy the copy taken before whatever failure made a second
/// attempt necessary.
///
/// **Both halves of the pair have to be free, and that is the whole of a bug worth remembering.**
/// The copy is staged at `<name>.partial` and renamed into place, so an abnormal exit between the
/// two leaves a `.partial` behind. Checking only the target then chose that same name again, and
/// `create_new` failed `EEXIST` on every subsequent start: measured, one `kill -9` during the
/// upgrade of a 90 MB store, then three consecutive runs all refusing with `failed to create the
/// pre-migration backup … File exists` and the store still at its old version. A single interrupted
/// upgrade wedged meka permanently, which is the opposite of what staging was introduced to
/// achieve.
///
/// Exhaustion is an error rather than a fallback to the unsuffixed name. Returning the occupied
/// base is not safe here: the write is staged to `.partial` and finished with `std::fs::rename`,
/// which refuses nothing, where `VACUUM INTO` would have refused a non-empty target. So the old
/// fallback silently overwrote the *oldest* backup, the one most likely to matter.
pub(super) fn free_backup_path(database_path: &Path, from: u32) -> Result<PathBuf> {
    let base = {
        let mut name = database_path.as_os_str().to_os_string();
        name.push(format!(".v{from}.bak"));
        PathBuf::from(name)
    };
    if is_free_pair(&base) {
        return Ok(base);
    }
    for suffix in 1..MAX_BACKUP_SUFFIX {
        let mut name = base.as_os_str().to_os_string();
        name.push(format!(".{suffix}"));
        let candidate = PathBuf::from(name);
        if is_free_pair(&candidate) {
            return Ok(candidate);
        }
    }
    Err(MekaError::Database(format!(
        "cannot name a pre-migration backup: '{}' and its {} numbered variants are all taken. \
         Nothing has been changed. Move or delete the old copies beside the store",
        base.display(),
        MAX_BACKUP_SUFFIX - 1
    )))
}
/// Whether both the backup name and the staging name it implies are unused.
pub(super) fn is_free_pair(target: &Path) -> bool {
    let staging = staging_path(target);
    is_free(target) && is_free(&staging)
}
/// Where a copy is written before it is renamed onto `target`.
///
/// One function so the caller and [`is_free_pair`] cannot disagree about the name, which is exactly
/// how the wedge above happened: the check looked at one path and the create at another.
pub(super) fn staging_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_os_string();
    name.push(".partial");
    PathBuf::from(name)
}
/// Whether nothing at all sits at this path, symlinks included.
///
/// `Path::exists` is the wrong question here: it follows symlinks, so it answers `false` for a
/// *dangling* one and this would hand back a name that is really a redirection.
/// `symlink_metadata` asks about the link rather than through it.
///
/// Staging already keeps `VACUUM INTO` off any such link, since it only ever touches `.partial` and
/// `rename` replaces a symlink rather than following it. This check is for the name: one occupied
/// by a link is still occupied, and handing it back would mean the log naming a file that is not
/// the one written.
pub(super) fn is_free(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_err()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A name already taken is stepped over rather than reused, so an earlier backup survives a
    /// second attempt.
    #[test]
    fn a_backup_name_already_in_use_is_not_reused() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        assert_eq!(
            free_backup_path(&database_path, 1).expect("a free name"),
            temp_dir.path().join("meka.db.v1.bak"),
            "the unsuffixed name when nothing is there"
        );

        std::fs::write(
            temp_dir.path().join("meka.db.v1.bak"),
            b"an earlier attempt",
        )
        .expect("plant");
        assert_eq!(
            free_backup_path(&database_path, 1).expect("a free name"),
            temp_dir.path().join("meka.db.v1.bak.1")
        );
        std::fs::write(temp_dir.path().join("meka.db.v1.bak.1"), b"and another").expect("plant");
        assert_eq!(
            free_backup_path(&database_path, 1).expect("a free name"),
            temp_dir.path().join("meka.db.v1.bak.2")
        );
    }

    /// `Path::exists` follows symlinks and so reports `false` for a dangling one, which would hand
    /// back a name that is really a redirection out of the data directory. The whole credential
    /// store went with it when this was measured.
    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_does_not_count_as_a_free_backup_name() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        let taken = temp_dir.path().join("meka.db.v1.bak");
        std::os::unix::fs::symlink(temp_dir.path().join("nowhere"), &taken).expect("symlink");
        assert!(
            !taken.exists(),
            "the premise: the link dangles, so `exists` says no"
        );

        assert_eq!(
            free_backup_path(&database_path, 1).expect("a free name"),
            temp_dir.path().join("meka.db.v1.bak.1"),
            "a dangling symlink occupies the name as surely as a file does"
        );
    }

    /// Debris from an interrupted upgrade must not wedge the next one.
    ///
    /// The copy is staged at `<name>.partial`, so a crash between creating it and renaming it into
    /// place leaves that file behind. Checking only the target name then reused it, `create_new`
    /// failed `EEXIST`, and *every* later start refused with the store still unmigrated: one Ctrl-C
    /// during an upgrade would brick meka permanently.
    #[test]
    fn a_staging_file_left_by_a_crash_does_not_block_the_next_attempt() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        std::fs::write(
            temp_dir.path().join("meka.db.v1.bak.partial"),
            b"an interrupted copy",
        )
        .expect("plant the debris");

        assert_eq!(
            free_backup_path(&database_path, 1).expect("a usable name is still found"),
            temp_dir.path().join("meka.db.v1.bak.1"),
            "the pair is taken, so the next pair is used rather than failing forever"
        );
    }

    /// Running out of names is an error, not a fallback to the occupied one.
    ///
    /// Returning the base name would be safe only if the write still went through `VACUUM INTO`,
    /// which refuses a non-empty target. Staging moved the write to `.partial` and the final step
    /// to `std::fs::rename`, which refuses nothing, so a fallthrough to the base name would
    /// silently overwrite the *oldest* backup: the one most likely to be the one that mattered.
    #[test]
    fn running_out_of_backup_names_refuses_rather_than_overwriting_the_oldest() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        std::fs::write(
            temp_dir.path().join("meka.db.v1.bak"),
            b"the irreplaceable one",
        )
        .expect("plant");
        for suffix in 1..1000 {
            std::fs::write(
                temp_dir.path().join(format!("meka.db.v1.bak.{suffix}")),
                b"x",
            )
            .expect("plant");
        }

        let error = free_backup_path(&database_path, 1).expect_err("no name is available");
        assert!(
            error.to_string().contains("Nothing has been changed"),
            "{error}"
        );
        assert_eq!(
            std::fs::read(temp_dir.path().join("meka.db.v1.bak")).expect("still there"),
            b"the irreplaceable one",
            "the oldest copy must survive"
        );
    }

    /// Best-effort, and both halves matter: it clears a copy that never finished, and it stays
    /// quiet when there is nothing to clear, because the caller is already returning the error
    /// that stopped the migration.
    #[test]
    fn an_incomplete_backup_is_cleared_and_a_missing_one_is_not_an_error() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let staging = temp_dir.path().join("meka.db.v1.bak.partial");
        std::fs::write(&staging, b"half a copy").expect("plant");
        remove_partial_backup(&staging);
        assert!(!staging.exists(), "the incomplete copy is gone");
        remove_partial_backup(&staging);
    }

    /// When no backup can be taken, the migration refuses and the store is left where it was.
    ///
    /// Reached here by exhausting every name, which is the one way to make the copy impossible that
    /// a test can force deterministically. A read-only data directory does not work, because
    /// `Store::open` calls `restrict_permissions(parent, 0o700)` on the way in and widens
    /// it back. Occupying the staging name with a directory does not work either, because
    /// `free_backup_path` treats a taken `.partial` as a taken pair and steps past it.
    #[tokio::test]
    async fn a_migration_that_cannot_be_backed_up_refuses_and_changes_nothing() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::store::migrations::baseline_for_test())
                .expect("a store with work pending");
        }
        std::fs::write(temp_dir.path().join("meka.db.v1.bak"), b"the oldest copy").expect("plant");
        for suffix in 1..1000 {
            std::fs::write(
                temp_dir.path().join(format!("meka.db.v1.bak.{suffix}")),
                b"x",
            )
            .expect("plant");
        }

        let error = Store::open(Some(&database_path), &Default::default())
            .await
            .err()
            .expect("the migration refuses rather than proceeding without a backup");
        assert!(
            error.to_string().contains("Nothing has been changed"),
            "{error}"
        );

        let connection = rusqlite::Connection::open(&database_path).expect("reopen");
        let version: i64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .expect("version");
        assert_eq!(version, 0, "no migration ran");
        let old_column: i64 = connection
            .query_row(
                "SELECT count(*) FROM pragma_table_info('scheduled_jobs') WHERE name = 'gate_command'",
                [],
                |row| row.get(0),
            )
            .expect("count");
        assert_eq!(old_column, 1, "the conversion did not start");
        assert_eq!(
            std::fs::read(temp_dir.path().join("meka.db.v1.bak")).expect("still there"),
            b"the oldest copy",
            "and the existing copies are untouched"
        );
    }

    /// A store carried forward keeps a copy of what it was, without the user having asked.
    ///
    /// The instruction the retired script's documentation had to give ("take a backup first") is
    /// the one a user skips, and it is only needed on the one run that might go wrong. Doing it
    /// here means it happened.
    #[tokio::test]
    async fn an_upgraded_store_leaves_a_copy_of_what_it_was() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            // A store as an older meka left one: the baseline shape, and nothing in `user_version`.
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::store::migrations::baseline_for_test())
                .expect("the baseline builds");
        }

        let store = Store::open(Some(&database_path), &Default::default())
            .await
            .expect("the store migrates on open");
        drop(store);

        let backup = temp_dir.path().join("meka.db.v1.bak");
        assert!(
            backup.exists(),
            "the pre-migration copy should be beside the store"
        );
        let copy = rusqlite::Connection::open(&backup).expect("the copy opens");
        let version: i64 = copy
            .query_row("SELECT * FROM pragma_user_version", [], |row| row.get(0))
            .expect("the copy has a version");
        assert_eq!(
            version, 0,
            "the copy must identify itself as what it was, so restoring it migrates once rather \
             than being mistaken for a current store"
        );
    }

    /// [`parse_number`]'s own contract, which its callers only partly exercise.
    ///
    /// Both call sites reject zero themselves, so nothing downstream can tell `Some(0)` from `None`
    /// here, and a mutation sweep found exactly that: widening the leading-zero guard's
    /// `digits.len() > 1` to `>= 1`, which makes a bare `0` a non-match, survived the entire suite.
    /// It is worth pinning anyway, because this function's job is to read back what `Display` wrote
    /// and `Display` writes zero as `0`. A guard that is only correct because of who happens to
    /// call it is the kind that drifts.
    #[test]
    fn a_number_is_parsed_exactly_as_display_would_have_written_it() {
        assert_eq!(parse_number(b"0"), Some(0), "`Display` writes zero as `0`");
        assert_eq!(parse_number(b"1"), Some(1));
        assert_eq!(parse_number(b"999"), Some(999));
        assert_eq!(
            parse_number(u32::MAX.to_string().as_bytes()),
            Some(u32::MAX)
        );
        assert_eq!(parse_number(b""), None, "no digits at all is not a number");
        assert_eq!(parse_number(b"01"), None, "`Display` never pads");
        assert_eq!(parse_number(b"007"), None);
        assert_eq!(
            parse_number(b"4294967296"),
            None,
            "one past `u32::MAX` is a non-match rather than a wrap"
        );
        assert_eq!(parse_number(b"99999999999999999999"), None);
    }

    /// Exactly which names this module will delete, stated as a table.
    ///
    /// The cheapest guard on the whole change, and the one that matters most: everything else
    /// decides *when* to prune, and this decides *what*. A match that is too generous deletes a
    /// file meka did not write, which is the one outcome the feature must never have.
    ///
    /// `meka.db.vault.bak` earns its row. It is the near-miss the digit requirement exists for:
    /// drop that requirement and `.v` followed by anything at all becomes a backup.
    #[test]
    fn only_names_this_module_writes_are_recognized_as_backups() {
        let store = std::ffi::OsStr::new("meka.db");
        let matches = |name: &str| is_backup_name(store, std::ffi::OsStr::new(name));

        for name in ["meka.db.v1.bak", "meka.db.v42.bak", "meka.db.v1.bak.7"] {
            assert!(matches(name), "{name} is a name free_backup_path builds");
        }
        for name in [
            "meka.db",
            "meka.db-wal",
            "meka.db-shm",
            "meka.db.mine.bak",
            // `.v` then a word, not a version.
            "meka.db.vault.bak",
            // Staging is deliberately spared: `free_backup_path` reads it to step past a name.
            "meka.db.v1.bak.partial",
            // A version or suffix has to be digits, and the suffix has to end the name.
            "meka.db.v.bak",
            "meka.db.v1.bak.",
            "meka.db.v1.bak.x",
            "meka.db.v1.bak.1.2",
            // Digits are not enough: these are shapes `free_backup_path` cannot emit, so a file
            // wearing one belongs to whoever made it. `.bak.0` is the near-miss that matters,
            // being how a person numbering their own archive from zero would name it.
            "meka.db.v1.bak.0",
            "meka.db.v0.bak",
            "meka.db.v01.bak",
            "meka.db.v1.bak.01",
            "meka.db.v1.bak.1000",
            "meka.db.v99999999999999999999.bak",
            // Another store's backup in a shared directory.
            "other.db.v1.bak",
            "notes.txt",
        ] {
            assert!(!matches(name), "{name} is not meka's to delete");
        }
    }

    /// One copy survives an upgrade, not one per release.
    ///
    /// The planted names are what a store carried through two earlier releases looks like, and they
    /// also force the interesting interaction: `free_backup_path` steps past both to `.bak.2`, so
    /// the copy being kept is *not* the name the older ones wear. A prune keyed to the name rather
    /// than to the path would take the wrong file.
    #[tokio::test]
    async fn only_the_newest_pre_migration_copy_survives() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::store::migrations::baseline_for_test())
                .expect("a store with work pending");
        }
        std::fs::write(temp_dir.path().join("meka.db.v1.bak"), b"older").expect("plant");
        std::fs::write(temp_dir.path().join("meka.db.v1.bak.1"), b"less old").expect("plant");

        drop(
            Store::open(Some(&database_path), &Default::default())
                .await
                .expect("the store migrates on open"),
        );

        let kept = temp_dir.path().join("meka.db.v1.bak.2");
        assert!(kept.exists(), "the copy just taken is the one that stays");
        assert!(
            !temp_dir.path().join("meka.db.v1.bak").exists()
                && !temp_dir.path().join("meka.db.v1.bak.1").exists(),
            "and the copies it supersedes are gone"
        );
        let copy = rusqlite::Connection::open(&kept).expect("the survivor opens");
        let version: i64 = copy
            .query_row("SELECT * FROM pragma_user_version", [], |row| row.get(0))
            .expect("the copy has a version");
        assert_eq!(version, 0, "and it is the real copy, not a planted file");
    }

    /// Everything else beside the store is left exactly as it was.
    ///
    /// A data directory is the user's, and a migration that tidies it is a migration that deletes
    /// something one day. The `.partial` is here because sparing it is load-bearing rather than
    /// incidental: `free_backup_path` treats a taken staging name as a taken pair, which is what
    /// makes it step to `.bak.1` below.
    #[tokio::test]
    async fn a_file_meka_did_not_write_is_left_alone() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::store::migrations::baseline_for_test())
                .expect("a store with work pending");
        }
        let bystanders = [
            "meka.db.mine.bak",
            // `.v` with no version at all: the near-miss that a match without the digit
            // requirement swallows, which is why it is here and not only in the table test.
            "meka.db.v.bak",
            "meka.db.vault.bak",
            "other.db.v1.bak",
            "notes.txt",
            "meka.db.v1.bak.partial",
        ];
        for name in bystanders {
            std::fs::write(temp_dir.path().join(name), name.as_bytes()).expect("plant");
        }

        drop(
            Store::open(Some(&database_path), &Default::default())
                .await
                .expect("the store migrates on open"),
        );

        for name in bystanders {
            let path = temp_dir.path().join(name);
            assert_eq!(
                std::fs::read(&path).expect("still there").as_slice(),
                name.as_bytes(),
                "{name} is not meka's to delete"
            );
        }
        assert!(
            temp_dir.path().join("meka.db.v1.bak.1").exists(),
            "and the copy stepped past the occupied staging name, as it always did"
        );
    }

    /// A prune that cannot delete is a warning, not a failed upgrade.
    ///
    /// Driven directly rather than through `Store::open`, because open widens the data
    /// directory back to `0700` on its way in (`restrict_permissions`), so a read-only parent
    /// cannot be staged through it. The property under test belongs to the helper anyway: it must
    /// return, having done what it could, whatever the filesystem says.
    #[cfg(unix)]
    #[test]
    fn a_prune_that_cannot_delete_still_returns() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let directory = temp_dir.path().join("data");
        std::fs::create_dir(&directory).expect("create");
        let database_path = directory.join("meka.db");
        std::fs::write(&database_path, b"store").expect("plant");
        let keep = directory.join("meka.db.v2.bak");
        std::fs::write(&keep, b"newest").expect("plant");
        let doomed = directory.join("meka.db.v1.bak");
        std::fs::write(&doomed, b"older").expect("plant");

        let original = std::fs::metadata(&directory).expect("stat").permissions();
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o500))
            .expect("make it read-only");
        prune_older_backups(&database_path, &keep);
        std::fs::set_permissions(&directory, original).expect("restore so the tempdir can go");

        assert!(
            doomed.exists(),
            "it failed to delete, which is the point: it must not have panicked or unwound either"
        );
    }

    /// Nothing is pruned unless a fresh copy actually landed.
    ///
    /// The failure this needs is a narrow one: it has to happen *after* `free_backup_path` has
    /// chosen a name and *before* the rename, since anything earlier or later cannot tell a
    /// prune-then-copy ordering from the copy-then-prune one. A store path that is not valid UTF-8
    /// is exactly that: `back_up_before_migrating` gets as far as `staging.to_str()` and refuses
    /// there, with a name already picked and no copy written.
    ///
    /// Control flow is what enforces the ordering; this is what notices a rearrangement of it.
    ///
    /// Linux, not `unix`: macOS validates filenames as UTF-8 in the kernel and answers `EILSEQ` to
    /// a `create` for this name, so the store path this needs cannot be built there. The ordering
    /// is still guarded on every platform by
    /// [`a_failed_migration_keeps_the_copy_it_would_have_superseded`], which reaches the same
    /// refusal through an unreadable config.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn nothing_is_pruned_when_no_fresh_copy_landed() {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};

        let temp_dir = tempfile::tempdir().expect("tempdir");
        // A directory whose name is not valid UTF-8, so every path under it is not either.
        let mut name = OsString::from_vec(b"not-utf8-\xff".to_vec());
        name.push("");
        let directory = temp_dir.path().join(name);
        std::fs::create_dir(&directory).expect("create");
        let database_path = directory.join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::store::migrations::baseline_for_test())
                .expect("a store with work pending");
        }
        let planted = directory.join("meka.db.v1.bak");
        std::fs::write(&planted, b"the only copy").expect("plant");

        let error = Store::open(Some(&database_path), &Default::default())
            .await
            .err()
            .expect("a backup that cannot be written refuses the migration");
        assert!(
            error.to_string().contains("not valid UTF-8"),
            "the refusal should be the one this test aims at, not some other: {error}"
        );
        assert_eq!(
            std::fs::read(&planted).expect("still there"),
            b"the only copy",
            "a prune that ran before the copy landed would have left the user with none"
        );
    }

    /// A migration that fails leaves the copy it was going to supersede where it is.
    ///
    /// The sibling of [`nothing_is_pruned_when_no_fresh_copy_landed`], guarding the other end of
    /// the same ordering. That one proves nothing is pruned before a copy lands; this one proves
    /// nothing is pruned before the migration those copies exist to undo has actually committed.
    ///
    /// The failure driven here is the one the 0.43-to-0.44 upgrade actually hits: a `config.toml`
    /// that will not parse leaves `sessions_name_their_provider` with a carried-forward session it
    /// cannot name a profile for, so it refuses and rolls back. `apply` restores the store, but a
    /// deleted file is not part of that transaction, so a prune that ran first would already have
    /// taken the copy `upgrading.md` tells the user to fall back on -- and a retry, which is what
    /// the guide's own instructions produce, would take the next one.
    #[tokio::test]
    async fn a_failed_migration_keeps_the_copy_it_would_have_superseded() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::store::migrations::baseline_for_test())
                .expect("a store with work pending");
            connection
                .execute(
                    "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s', 'now', 'now')",
                    [],
                )
                .expect("a session that has to be given a profile");
        }
        let planted = temp_dir.path().join("meka.db.v1.bak");
        std::fs::write(&planted, b"the only copy").expect("plant");

        let error = Store::open(
            Some(&database_path),
            &crate::store::migrations::Context::on_unreadable_config(),
        )
        .await
        .err()
        .expect("a session that cannot be given a profile refuses the migration");
        assert!(
            error.to_string().contains("carried-forward"),
            "the refusal should be the one this test aims at, not some other: {error}"
        );
        assert_eq!(
            std::fs::read(&planted).expect("still there"),
            b"the only copy",
            "a prune that ran before the migration committed would have destroyed the copy the \
             failed upgrade tells the user to fall back on"
        );
    }

    /// A directory that is not there at all is not an error, for the same reason.
    #[test]
    fn a_prune_with_nowhere_to_look_is_not_an_error() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let missing = temp_dir.path().join("gone").join("meka.db");
        prune_older_backups(
            &missing,
            &temp_dir.path().join("gone").join("meka.db.v1.bak"),
        );
    }

    /// A backup protects data that already exists, so neither a first run nor a subsequent one
    /// leaves a copy: the first has nothing to lose, and the second has nothing to do. Copying on
    /// every start would double the store on disk and make opening meka cost more as the
    /// conversation grows.
    #[tokio::test]
    async fn a_store_with_nothing_to_preserve_is_not_backed_up() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        drop(
            Store::open(Some(&database_path), &Default::default())
                .await
                .expect("first open"),
        );
        drop(
            Store::open(Some(&database_path), &Default::default())
                .await
                .expect("second open"),
        );

        let copies: Vec<_> = std::fs::read_dir(temp_dir.path())
            .expect("readable")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".bak"))
            .collect();
        assert!(copies.is_empty(), "expected no backups, found {copies:?}");
    }
}
