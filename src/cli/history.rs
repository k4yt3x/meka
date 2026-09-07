//! `meka history`: the REPL's input history.

use crate::store::Store;

pub(crate) fn run_history_subcommand(
    store: &Store,
    action: &crate::cli::HistoryAction,
) -> anyhow::Result<()> {
    // Capacity only gates the write/prune path, so `0` is fine for read/clear. The table is the
    // ledger's, and the store this handle came from has already applied it.
    let history = crate::store::history::HistoryStore::open(store.database_path(), 0)?;
    match action {
        crate::cli::HistoryAction::List { limit, format } => {
            let entries = history.recent(*limit as usize)?;
            if *format == crate::cli::OutputFormat::Json {
                crate::cli::write_json_listing("history", &entries)?;
            } else if entries.is_empty() {
                crate::streams::write_stderr_line("No history.");
            } else {
                for entry in entries {
                    crate::render::write_stdout_line(&entry)?;
                }
            }
        }
        crate::cli::HistoryAction::Clear => {
            let removed = history.clear_all()?;
            let noun = if removed == 1 { "entry" } else { "entries" };
            tracing::info!("cleared {removed} input history {noun}");
        }
    }
    Ok(())
}
