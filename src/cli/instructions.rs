//! `meka instructions`: where the standing instructions come from, and their text.

use std::path::PathBuf;

/// `meka instructions`: what the model is being told, and where that came from.
///
/// Four tiers feed one value and the conventional path appears in no config file, so the question
/// is otherwise only answerable by reading the source. The files `[instructions].files` names are
/// read against the directory this command runs in, which stands in for a session opened there.
pub(crate) fn run_instructions_subcommand(
    action: &crate::cli::InstructionsAction,
) -> anyhow::Result<()> {
    match action {
        crate::cli::InstructionsAction::Show { format } => {
            // `--instructions` belongs to a run, not to this query, so it is deliberately not
            // consulted here; `None` resolves the persistent tiers only.
            let standing = crate::instructions::resolve_for_display()?;
            let listed =
                crate::instructions::read_files(&listed_entries()?, &std::env::current_dir()?);
            if *format == crate::cli::OutputFormat::Json {
                // Absent fields are left out, so a configuration with no instructions is `{}`.
                #[derive(serde::Serialize)]
                struct InstructionsView {
                    #[serde(skip_serializing_if = "Option::is_none")]
                    source: Option<String>,
                    #[serde(skip_serializing_if = "Option::is_none")]
                    files: Option<String>,
                    #[serde(skip_serializing_if = "Option::is_none")]
                    content: Option<String>,
                }
                let source = standing.as_ref().map(|found| found.source.to_string());
                let files = listed.as_ref().map(|found| found.source.to_string());
                let content = crate::instructions::join(standing.map(|found| found.text), listed);
                crate::cli::write_json(&InstructionsView {
                    source,
                    files,
                    content,
                })?;
                return Ok(());
            }
            if standing.is_none() && listed.is_none() {
                crate::streams::write_stderr_line(format!(
                    "No instructions configured; write them to {}.",
                    display_path(crate::instructions::instructions_file()),
                ));
                return Ok(());
            }
            // Sources to stderr, text to stdout: the text is the data you asked for, so
            // `2>/dev/null` leaves something pipeable.
            if let Some(found) = &standing {
                crate::streams::write_stderr_line(format!("Source: {}", found.source));
            }
            if let Some(found) = &listed {
                crate::streams::write_stderr_line(format!("Files: {}", found.source));
            }
            crate::streams::write_stderr_line("");
            if let Some(text) = crate::instructions::join(standing.map(|found| found.text), listed)
            {
                crate::render::write_stdout_line(&text)?;
            }
        }
        crate::cli::InstructionsAction::Path { format } => {
            let current = std::env::current_dir()?;
            let listed = listed_entries()?
                .into_iter()
                .map(|entry| crate::instructions::listed_path(&entry, &current));
            let entries: Vec<(String, bool)> = [
                crate::instructions::instructions_dir(),
                crate::instructions::instructions_file(),
            ]
            .into_iter()
            .flatten()
            .chain(listed)
            .map(|path| (path.display().to_string(), path.exists()))
            .collect();
            if *format == crate::cli::OutputFormat::Json {
                #[derive(serde::Serialize)]
                struct PathView<'a> {
                    path: &'a str,
                    exists: bool,
                }
                let views: Vec<PathView<'_>> = entries
                    .iter()
                    .map(|(path, exists)| PathView {
                        path,
                        exists: *exists,
                    })
                    .collect();
                crate::cli::write_json_listing("paths", &views)?;
                return Ok(());
            }
            if entries.is_empty() {
                crate::streams::write_stderr_line("No config directory.");
                return Ok(());
            }
            let rows: Vec<Vec<String>> = entries
                .into_iter()
                .map(|(path, exists)| vec![path, if exists { "yes" } else { "no" }.to_string()])
                .collect();
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

/// `[instructions].files` as the config file has it, resolved the way a run resolves it.
fn listed_entries() -> anyhow::Result<Vec<PathBuf>> {
    let config = crate::config::load_config_file_or_err()?;
    Ok(crate::config::resolve_instruction_files(
        config
            .instructions
            .unwrap_or_default()
            .files
            .as_deref()
            .unwrap_or_default(),
    ))
}

pub(crate) fn display_path(path: Option<std::path::PathBuf>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| "<no config directory>".to_string())
}
