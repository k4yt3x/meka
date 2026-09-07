//! `meka instructions`: where the standing instructions come from, and their text.

/// `meka instructions`: what the model is being told, and where that came from.
///
/// Four tiers feed one value and the conventional path appears in no config file, so the question
/// is otherwise only answerable by reading the source.
pub(crate) fn run_instructions_subcommand(
    action: &crate::cli::InstructionsAction,
) -> anyhow::Result<()> {
    match action {
        crate::cli::InstructionsAction::Show => {
            // `--instructions` belongs to a run, not to this query, so it is deliberately not
            // consulted here; `None` resolves the persistent tiers only.
            match crate::instructions::resolve_for_display()? {
                Some(found) => {
                    // Source to stderr, text to stdout: the text is the data you asked for, so
                    // `2>/dev/null` leaves something pipeable.
                    crate::streams::write_stderr_line(format!("Source: {}", found.source));
                    crate::streams::write_stderr_line("");
                    crate::render::write_stdout_line(&found.text)?;
                }
                None => crate::streams::write_stderr_line(format!(
                    "No instructions configured; write them to {}.",
                    display_path(crate::instructions::instructions_file()),
                )),
            }
        }
        crate::cli::InstructionsAction::Path => {
            for path in [
                crate::instructions::instructions_dir(),
                crate::instructions::instructions_file(),
            ]
            .into_iter()
            .flatten()
            {
                let state = if path.exists() { "present" } else { "absent" };
                crate::render::write_stdout_line(format!("{}\t{}", path.display(), state))?;
            }
        }
    }
    Ok(())
}
pub(crate) fn display_path(path: Option<std::path::PathBuf>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| "<no config directory>".to_string())
}
