//! Where meka keeps things: the configuration and data directories, the file each holds, and the
//! `~` expansion every path setting goes through. A leaf, so any layer may ask without reaching
//! into `config`.

use std::path::PathBuf;

/// Expand a leading `~` or `~/` in a user-supplied path to the home directory.
///
/// Returns `None` only when a tilde needs the home directory but it cannot be determined, which the
/// caller reports rather than silently treating `~` as a relative directory of that name.
///
/// Shared by the screen's `/cd` (and its path completer) and by `[skills] additional_directories`,
/// so a tilde means the same thing wherever a user types a path. Deliberately does *not* expand
/// `~user`: that needs a password-database lookup, and no config surface here has ever accepted the
/// form.
pub(crate) fn expand_user_path(target: &str) -> Option<PathBuf> {
    if target.is_empty() || target == "~" {
        dirs::home_dir()
    } else if let Some(rest) = target
        .strip_prefix('~')
        .and_then(|rest| rest.strip_prefix(std::path::is_separator))
    {
        // `is_separator`, not a literal `"~/"`. On Windows both `/` and `\` separate, and `~\` is
        // the spelling a user reaches for there because that is what PowerShell's own tilde
        // expansion produces. Matched on `~/` only, `~\projects` stays a literal relative
        // directory of that name.
        dirs::home_dir().map(|home| home.join(rest))
    } else {
        Some(PathBuf::from(target))
    }
}
/// Returns the meka config directory (the directory that contains `config.toml` and `skills/`).
/// Honors the `MEKA_CONFIG_DIR` env var, used by tests for per-run isolation and by power users
/// who want a non-standard location, before falling back to the platform-native
/// `dirs::config_dir().join("meka")`. The env-var route is the only reliable way to isolate state
/// on macOS and Windows, where `dirs::config_dir()` doesn't honor `XDG_CONFIG_HOME`.
///
/// `None` for a relative override as well as for a platform with no config directory: the user
/// asked for another directory, and the platform's would be the real one, so a relative value
/// fails closed. [`refuse_relative_overrides`] says so at startup.
pub(crate) fn meka_config_dir() -> Option<PathBuf> {
    match override_of(std::env::var_os("MEKA_CONFIG_DIR")) {
        Override::Absolute(path) => Some(path),
        Override::Relative(_) => None,
        Override::Unset => dirs::config_dir().map(|directory| directory.join("meka")),
    }
}

/// What a directory override holds. An empty value reads as unset, as an empty `XDG_*` does: a
/// shell profile's `MEKA_CONFIG_DIR=` or a systemd unit that exports one blank asks for nothing.
enum Override {
    Unset,
    Absolute(PathBuf),
    Relative(PathBuf),
}

fn override_of(value: Option<std::ffi::OsString>) -> Override {
    match value {
        None => Override::Unset,
        Some(value) if value.is_empty() => Override::Unset,
        Some(value) => {
            let path = PathBuf::from(value);
            if path.is_absolute() {
                Override::Absolute(path)
            } else {
                Override::Relative(path)
            }
        }
    }
}

/// Refuse a relative `MEKA_CONFIG_DIR` or `MEKA_DATA_DIR`, before anything reads either directory.
///
/// Falling back to the platform directory would act on the real configuration and the real store,
/// credentials and all, when the user asked for another place; resolving it against the working
/// directory would give each launch directory its own store. Neither is what was meant.
pub(crate) fn refuse_relative_overrides() -> Result<(), String> {
    for (variable, value) in [
        ("MEKA_CONFIG_DIR", std::env::var_os("MEKA_CONFIG_DIR")),
        ("MEKA_DATA_DIR", std::env::var_os("MEKA_DATA_DIR")),
    ] {
        if let Override::Relative(path) = override_of(value) {
            return Err(format!(
                "`{variable}` must be an absolute path, not '{}'",
                path.display()
            ));
        }
    }
    Ok(())
}

/// The directory `meka.db` lives in: the `MEKA_DATA_DIR` override, else the platform data
/// directory's `meka` subdirectory. `None` when the platform names no data directory at all, and
/// for a relative override, for the reason [`meka_config_dir`] gives.
///
/// `dirs::data_dir()` honors `XDG_DATA_HOME` on Linux, returns `~/Library/Application Support` on
/// macOS, and `%APPDATA%` on Windows. No silent fallback: writing the session store to a
/// wrong-for-the-platform path is worse than asking the user to set `MEKA_DATA_DIR`.
pub(crate) fn meka_data_dir() -> Option<PathBuf> {
    match override_of(std::env::var_os("MEKA_DATA_DIR")) {
        Override::Absolute(path) => Some(path),
        Override::Relative(_) => None,
        Override::Unset => dirs::data_dir().map(|base| base.join("meka")),
    }
}
/// `config.toml`'s path, under [`meka_config_dir`].
pub(crate) fn config_file_path() -> Option<PathBuf> {
    meka_config_dir().map(|dir| dir.join("config.toml"))
}

/// The skills directory meka itself manages, under [`meka_config_dir`].
pub(crate) fn skills_dir() -> Option<PathBuf> {
    crate::paths::meka_config_dir().map(|dir| dir.join("skills"))
}
/// The roots to scan, in precedence order: meka's own first, then `additional_directories` as
/// given.
///
/// meka's own root leads because it is the store the user curates *through meka*, and because it is
/// the only one anything writes to; a skill there should not be shadowed by a copy another client
/// installed.
pub(crate) fn skill_roots(extra: &[PathBuf]) -> Vec<PathBuf> {
    skills_dir()
        .into_iter()
        .chain(extra.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A relative override is refused, and no directory is resolved from it meanwhile, neither
    /// the platform's nor one under the working directory; an empty one is unset.
    #[test]
    fn a_relative_directory_override_is_refused_and_resolves_to_nothing() {
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        // SAFETY: guarded above.
        unsafe { std::env::set_var("MEKA_DATA_DIR", "relative/data") };
        let refused = refuse_relative_overrides();
        let resolved = meka_data_dir();
        // SAFETY: guarded above.
        unsafe { std::env::set_var("MEKA_DATA_DIR", "") };
        let unset = meka_data_dir();
        // SAFETY: guarded above.
        unsafe { std::env::remove_var("MEKA_DATA_DIR") };

        let refused = refused.expect_err("a relative override is refused");
        assert!(refused.contains("`MEKA_DATA_DIR`"), "{refused}");
        assert_eq!(resolved, None, "and resolves to no directory meanwhile");
        assert_eq!(
            unset,
            dirs::data_dir().map(|base| base.join("meka")),
            "an empty one is unset"
        );
    }
}
