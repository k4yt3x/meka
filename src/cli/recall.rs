//! `meka recall`: the screen's recall.

use crate::store::Store;

pub(crate) fn run_recall_subcommand(
    store: &Store,
    action: &crate::cli::RecallAction,
) -> anyhow::Result<()> {
    // Capacity only gates the write/prune path, so `0` is fine for read/clear. The table is the
    // ledger's, and the store this handle came from has already applied it.
    let recall = crate::store::recall::RecallStore::open(store.database_path(), 0)?;
    match action {
        crate::cli::RecallAction::List { limit, format } => {
            let entries = recall.recent(*limit as usize)?;
            if *format == crate::cli::OutputFormat::Json {
                crate::cli::write_json_listing("recall", &entries)?;
            } else if entries.is_empty() {
                crate::streams::write_stderr_line("No recall entries.");
            } else {
                for entry in entries {
                    crate::render::write_stdout_line(&entry)?;
                }
            }
        }
        crate::cli::RecallAction::Clear => {
            let removed = recall.clear_all()?;
            let noun = if removed == 1 { "entry" } else { "entries" };
            tracing::info!("cleared {removed} recall {noun}");
        }
    }
    Ok(())
}
