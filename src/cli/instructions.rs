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
            let rows: Vec<Vec<String>> = [
                crate::instructions::instructions_dir(),
                crate::instructions::instructions_file(),
            ]
            .into_iter()
            .flatten()
            .map(|path| {
                vec![
                    path.display().to_string(),
                    if path.exists() { "yes" } else { "no" }.to_string(),
                ]
            })
            .collect();
            if rows.is_empty() {
                crate::streams::write_stderr_line("No config directory.");
                return Ok(());
            }
            // The path is what you open, so it is shown in full.
            crate::render::write_stdout(crate::text::format_table(
                &[
                    crate::text::Column::content("Path"),
                    crate::text::Column::content("Exists"),
                ],
                &rows,
            ))?;
        }
    }
    Ok(())
}
pub(crate) fn display_path(path: Option<std::path::PathBuf>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| "<no config directory>".to_string())
}
