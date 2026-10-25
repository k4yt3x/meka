//! `meka project`: the projects a session can be on, as rows of the store.
//!
//! A project is a memory tier beside the global one and nothing more, so the store is where it
//! lives: `add` writes the row, `rename` moves it with every session and memory on it in one
//! transaction, and `remove` refuses while any row still names it.

use crate::{cli::ProjectAction, store::Store, text::Precision};

const PROJECT_COLUMNS: [crate::text::Column; 4] = [
    crate::text::Column::capped("Name", crate::text::NAME_WIDTH),
    crate::text::Column::content("Created"),
    crate::text::Column::content("Memories"),
    crate::text::Column::content("Sessions"),
];

pub(crate) async fn run(action: &ProjectAction, store: &Store) -> anyhow::Result<()> {
    match action {
        ProjectAction::Add { name } => {
            store.add_project(name).await?;
            tracing::info!("added project '{name}'");
        }
        ProjectAction::List { format } => run_list(store, *format).await?,
        ProjectAction::Rename { name, new_name } => {
            let (sessions, memories) = store.rename_project(name, new_name).await?;
            tracing::info!(
                "renamed project '{name}' to '{new_name}'; {sessions} session(s) and {memories} \
                 memor(ies) follow"
            );
        }
        ProjectAction::Remove { name } => {
            store.remove_project(name).await?;
            tracing::info!("removed project '{name}'");
        }
    }
    Ok(())
}

async fn run_list(store: &Store, format: crate::cli::OutputFormat) -> anyhow::Result<()> {
    let views: Vec<crate::view::ProjectView> = store
        .list_projects()
        .await?
        .into_iter()
        .map(crate::view::ProjectView::from)
        .collect();
    match format {
        crate::cli::OutputFormat::Json => crate::cli::write_json_listing("projects", &views)?,
        crate::cli::OutputFormat::Plain if views.is_empty() => {
            crate::streams::write_stderr_line("No projects.");
        }
        crate::cli::OutputFormat::Plain => {
            let rows: Vec<Vec<String>> = views
                .iter()
                .map(|view| {
                    vec![
                        view.name.clone(),
                        super::session::format_stored_timestamp(
                            &view.created_at,
                            Precision::Minutes,
                        ),
                        view.memories.to_string(),
                        view.sessions.to_string(),
                    ]
                })
                .collect();
            crate::render::write_stdout(crate::text::format_table(&PROJECT_COLUMNS, &rows))?;
        }
    }
    Ok(())
}
