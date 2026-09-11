//! `meka tool`: the built-in tool catalog as the model would see it.

use crate::{
    config::ResolvedConfig, permission::SharedPermission, store::Store, tools::ToolRegistry,
};

/// The columns of both tool tables, `meka tool list` and `meka mcp tools`: they answer the same
/// questions about a tool and are read side by side. The name is what the reader retypes into
/// `[tools]`, a per-tool `permission` or `--eager-load-tool`, and no command prints it in full
/// elsewhere, so it is never cut.
pub(super) const TOOL_TABLE_COLUMNS: [crate::text::Column; 5] = [
    crate::text::Column::content("Name"),
    crate::text::Column::content("Permission"),
    crate::text::Column::content("Source"),
    crate::text::Column::content("Status"),
    crate::text::Column::remainder("Description"),
];

/// Handle `meka tool <action>`.
pub(crate) fn run_tool_subcommand(
    store: &Store,
    action: &crate::cli::ToolAction,
    cli_args: &crate::cli::Cli,
) -> anyhow::Result<()> {
    match action {
        crate::cli::ToolAction::List { format } => {
            let config = ResolvedConfig::resolve(cli_args.overrides());
            // The table's permission and status columns come from config, so rendering it off
            // defaults would misreport every tool the user has overridden.
            config.require_readable_config()?;
            let filter = crate::config::BuiltinToolFilter::from_config(
                config.builtin_allowed_tools.clone(),
                config.builtin_disabled_tools.clone(),
                config.builtin_tool_permissions.clone(),
            );
            crate::tools::warn_on_stale_builtin_tool_config(&filter);

            // Build with no filter so the catalog carries every tool's hardcoded level; overlay
            // the real filter for status/source.
            let store = store.clone();
            // Built for its profile names only: the listing shows the `profile` choices a session
            // would offer, and nothing here resolves one.
            let providers = std::sync::Arc::new(crate::provider::ProviderRegistry::new(
                &config,
                store.token_store(),
            ));
            let shared_permission =
                SharedPermission::new(config.permission, config.enabled_permissions);
            let materials = crate::session::SessionMaterials {
                // Nothing here runs a shell, so the sandbox is not probed: the listing shows the
                // tools a session would have, at the levels they declare.
                core: crate::session::CoreMaterials::from_config(
                    &config,
                    crate::config::BuiltinToolFilter::default(),
                    &crate::sandbox::resolve_backend(config.sandbox_backend, false),
                ),
                // `meka tool list` only prints the catalog, so neither store's metadata is read
                // and the filesystem walk is skipped. The switches still have to be honored: this
                // listing exists to show what a real session would have.
                skills: if config.skills_enabled {
                    crate::skills::SkillCache::for_root(None)
                } else {
                    crate::skills::SkillCache::disabled()
                },
                skills_agent_managed: config.skills_agent_managed,
                memories: if config.memory_enabled {
                    crate::store::memory::MemoryStore::detached()
                } else {
                    crate::store::memory::MemoryStore::disabled()
                },
                store,
                providers,
                mcp_manager: None,
                session_stats: std::sync::Arc::new(crate::stats::SessionStats::default()),
                schedule: config.schedule.clone(),
                background: config.background.clone(),
                subagents: config.subagents.clone(),
                // At least one, so the `agent_*` family is registered and can be listed; whether a
                // real session would have it is asked below, against the configured depth.
                subagent_max_depth: config.subagent_max_depth.max(1),
            };
            let cells = crate::session::SessionCells::new(
                shared_permission,
                crate::workspace::SharedCwd::new(std::path::PathBuf::from(".")),
                crate::workspace::SharedRoots::default(),
                crate::provider::PublishedProfile::unbound(),
                std::sync::Arc::new(crate::frontend::SilentFrontend),
            );
            let reference = ToolRegistry::build_default(
                &materials,
                &cells,
                &crate::session::AgentOptions::from_config(&config, false, None, None),
            )?;

            let mut catalog = reference.tool_catalog();
            catalog.sort_by(|left, right| left.0.cmp(&right.0));
            // Both conditions that remove the whole family, asked the way `build_default` asks
            // them. A per-name `[tools]` entry is a separate question, applied below alongside
            // every other tool's.
            let agent_family_registered =
                crate::tools::subagent::agent_tools_registered(&filter, config.subagent_max_depth);

            let views: Vec<crate::view::ConfiguredToolView> = catalog
                .into_iter()
                .map(|(name, description, required, deferred)| {
                    let override_entry = filter.permission_overrides.get(&name).copied();
                    let enabled = filter.admits(&name)
                        && (agent_family_registered
                            || !crate::tools::subagent::AGENT_TOOL_NAMES.contains(&name.as_str()));
                    crate::view::ConfiguredToolView {
                        tool: crate::view::ToolView::new(
                            name,
                            description,
                            override_entry.unwrap_or(required),
                            deferred,
                        ),
                        permission_source: if override_entry.is_some() {
                            "override"
                        } else {
                            "builtin"
                        },
                        enabled,
                    }
                })
                .collect();
            if *format == crate::cli::OutputFormat::Json {
                crate::cli::write_json_listing("tools", &views)?;
                return Ok(());
            }
            let rows: Vec<Vec<String>> = views
                .iter()
                .map(|view| {
                    vec![
                        view.tool.name.clone(),
                        view.tool.required_permission.to_string(),
                        view.permission_source.to_string(),
                        if view.enabled {
                            if view.tool.deferred {
                                "deferred"
                            } else {
                                "enabled"
                            }
                        } else {
                            "disabled"
                        }
                        .to_string(),
                        crate::text::prose_cell(&view.tool.description),
                    ]
                })
                .collect();
            crate::render::write_stdout(crate::text::format_table(&TOOL_TABLE_COLUMNS, &rows))?;
        }
    }
    Ok(())
}
