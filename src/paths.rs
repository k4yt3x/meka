//! Where meka keeps things: the configuration and data directories, the file each holds, and the
//! `~` expansion every path setting goes through. A leaf, so any layer may ask without reaching
//! into `config`.

use std::path::PathBuf;

/// Expand a leading `~` or `~/` in a user-supplied path to the home directory.
///
/// Returns `None` only when a tilde needs the home directory but it cannot be determined, which the
/// caller reports rather than silently treating `~` as a relative directory of that name.
///
/// Shared by the REPL's `/cd` (and its path completer) and by `[skills] extra_paths`, so a tilde
/// means the same thing wherever a user types a path. Deliberately does *not* expand `~user`: that
/// needs a password-database lookup, and no config surface here has ever accepted the form.
pub(crate) fn expand_user_path(target: &str) -> Option<PathBuf> {
    if target.is_empty() || target == "~" {
        dirs::home_dir()
    } else if let Some(rest) = target
        .strip_prefix('~')
        .and_then(|rest| rest.strip_prefix(std::path::is_separator))
    {
        // `is_separator`, not a literal `"~/"`. On Windows both `/` and `\` separate, and `~\` is
        // the spelling a user reaches for there because that is what PowerShell's own tilde
        // expansion produces. Matching only `~/` left `~\projects` unexpanded, so it became a
        // literal relative directory of that name and `/cd` reported the tilde back at the user as
        // though it were a folder. `[skills] extra_paths` took the same spelling and resolved to a
        // root that never existed.
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
pub(crate) fn meka_config_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MEKA_CONFIG_DIR") {
        let path = PathBuf::from(dir);
        // An empty value reads as unset rather than as the current directory. `MEKA_CONFIG_DIR=`
        // in a shell profile, or an exported-but-unset variable in a systemd unit, made meka load
        // `./config.toml` from whatever directory it happened to start in -- and then spawn that
        // file's MCP servers. A relative value has the same shape of problem: which config you get
        // depends on your cwd. `MEKA_DATA_DIR` refuses both for the same reasons, and more
        // sharply: `meka.db` holds every provider credential.
        if path.as_os_str().is_empty() {
            // `warn!`, not `debug!`: an override the user set and meka did not honor is exactly
            // the recoverable-fallback case, and the environment-variable documentation says a
            // warning is what happens.
            tracing::warn!("MEKA_CONFIG_DIR is empty; using the platform config directory");
        } else if !path.is_absolute() {
            tracing::warn!(
                "MEKA_CONFIG_DIR '{path}' is not an absolute path; ignoring it and using the platform \
                 config directory",
                path = path.display()
            );
        } else {
            return Some(path);
        }
    }
    dirs::config_dir().map(|directory| directory.join("meka"))
}
/// The directory `meka.db` lives in: the `MEKA_DATA_DIR` override, else the platform data
/// directory's `meka` subdirectory. `None` only when the platform names no data directory at all.
///
/// `dirs::data_dir()` honors `XDG_DATA_HOME` on Linux, returns `~/Library/Application Support` on
/// macOS, and `%APPDATA%` on Windows. No silent fallback: writing the session store to a
/// wrong-for-the-platform path is worse than asking the user to set `MEKA_DATA_DIR`.
pub(crate) fn meka_data_dir() -> Option<PathBuf> {
    data_dir_override().or_else(|| dirs::data_dir().map(|base| base.join("meka")))
}
/// Where `execute_command` spools a command's output when it overflows the inline result.
///
/// `MEKA_DATA_DIR` first, so a run isolated to a scratch directory keeps its captures there too
/// rather than dropping them in the real user's cache; otherwise the platform cache directory,
/// since a capture is reproducible from the command that made it; otherwise the temp directory.
pub(crate) fn command_output_dir() -> PathBuf {
    data_dir_override()
        .map(|path| path.join("command-output"))
        .or_else(|| dirs::cache_dir().map(|directory| directory.join("meka")))
        .unwrap_or_else(std::env::temp_dir)
}
pub(crate) fn config_file_path() -> Option<PathBuf> {
    meka_config_dir().map(|dir| dir.join("config.toml"))
}

pub(crate) fn skills_dir() -> Option<PathBuf> {
    crate::paths::meka_config_dir().map(|dir| dir.join("skills"))
}
/// The roots to scan, in precedence order: meka's own first, then `extra_paths` as given.
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

/// `MEKA_DATA_DIR` when it is set to an absolute path, else `None`.
///
/// Absolute only, matching [`meka_config_dir`]. A relative value means the database, which holds
/// every provider credential, lands under whatever directory meka happened to start in, so `meka`
/// in one project and `meka` in another silently get different credential stores, and neither is
/// the one the user set up. Ignored rather than fatal, for the same reason the config directory
/// ignores it: the platform default is always a usable answer.
fn data_dir_override() -> Option<PathBuf> {
    let value = std::env::var_os("MEKA_DATA_DIR")?;
    if value.is_empty() {
        return None;
    }
    let path = PathBuf::from(value);
    if path.is_absolute() {
        return Some(path);
    }
    tracing::warn!(
        "MEKA_DATA_DIR '{path}' is not an absolute path; ignoring it and using the platform data \
         directory",
        path = path.display()
    );
    None
}
