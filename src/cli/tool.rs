//! `meka tool`: the built-in tool catalog as the model would see it, and any one built-in's schema.

use crate::{
    config::ResolvedConfig, error::MekaError, permission::SharedPermission, store::Store,
    tools::ToolRegistry, view::ToolDefinitionView,
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
    let config = ResolvedConfig::resolve(cli_args.overrides());
    // The permission and status columns come from config, so rendering them off defaults would
    // misreport every tool the user has overridden.
    config.require_readable_config()?;
    match action {
        crate::cli::ToolAction::List { format } => {
            let views = ReferenceTools::build(store, &config)?.catalog();
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
                        status_word(view.enabled, view.tool.deferred).to_string(),
                        crate::text::prose_cell(&view.tool.description),
                    ]
                })
                .collect();
            crate::render::write_stdout(crate::text::format_table(&TOOL_TABLE_COLUMNS, &rows))?;
        }
        crate::cli::ToolAction::Show { name, format } => {
            let shown = ReferenceTools::build(store, &config)?.show(name)?;
            if *format == crate::cli::OutputFormat::Json {
                crate::cli::write_json(&shown)?;
                return Ok(());
            }
            crate::render::write_stdout(render_shown(&shown)?)?;
        }
    }
    Ok(())
}

/// The one word the tables give a tool's standing with the config.
fn status_word(enabled: bool, deferred: bool) -> &'static str {
    match (enabled, deferred) {
        (false, _) => "disabled",
        (true, true) => "deferred",
        (true, false) => "enabled",
    }
}

/// The registry a session on this config would build, with every built-in registered whatever
/// `[tools]` says, beside what decides each tool's standing: the filter, and the two conditions
/// that remove a family whole.
struct ReferenceTools {
    registry: ToolRegistry,
    filter: crate::config::BuiltinToolFilter,
    agent_family_registered: bool,
    web_fetch_registered: bool,
}

impl ReferenceTools {
    fn build(store: &Store, config: &ResolvedConfig) -> anyhow::Result<Self> {
        let filter = crate::config::BuiltinToolFilter::from_config(
            config.builtin_allowed_tools.clone(),
            config.builtin_disabled_tools.clone(),
            config.builtin_tool_permissions.clone(),
        );
        crate::tools::warn_on_stale_builtin_tool_config(&filter);

        // Built with no filter so the catalog carries every tool's hardcoded level; the real
        // filter is overlaid for status and source.
        let store = store.clone();
        // Built for its profile names only: the listing shows the `profile` choices a session
        // would offer, and nothing here resolves one.
        let providers = std::sync::Arc::new(crate::provider::ProviderRegistry::new(
            config,
            store.token_store(),
        ));
        let shared_permission =
            SharedPermission::new(config.permission, config.enabled_permissions);
        // Probed, though nothing here runs a shell: `shell_execute` declares `read` only where the
        // sandbox is on and usable, so a listing built with the sandbox off would print
        // `unrestricted` on every machine whose sessions run the tool at `read`. The warning is
        // the one a session start gives, and it explains an `unrestricted` row.
        let sandbox = crate::sandbox::resolve_backend(config.sandbox_backend, config.sandbox);
        crate::sandbox::warn_if_sandbox_issues(
            &crate::sandbox::SandboxState::new(
                config.sandbox,
                config.confine_reads,
                config.network,
                &sandbox,
            ),
            crate::sandbox::WarnContext::ToolListing,
        );
        let materials = crate::session::SessionMaterials {
            core: crate::session::CoreMaterials::from_config(
                config,
                crate::config::BuiltinToolFilter::default(),
                &sandbox,
            ),
            // Only the catalog is read, so neither store's metadata is read and the filesystem
            // walk is skipped. The switches still have to be honored: this listing exists to show
            // what a real session would have.
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
            // At least one, so the `agent_*` family is registered and can be listed; whether a
            // real session would have it is asked below, against the configured depth.
            subagents: crate::config::ResolvedSubagentsConfig {
                max_depth: config.subagents.max_depth.max(1),
                ..config.subagents.clone()
            },
        };
        let cells = crate::session::SessionCells::new(
            shared_permission,
            crate::workspace::SharedCwd::new(std::path::PathBuf::from(".")),
            crate::workspace::SharedRoots::default(),
            crate::provider::PublishedProfile::unbound(),
            std::sync::Arc::new(crate::frontend::SilentFrontend),
        );
        let registry = ToolRegistry::build_default(
            &materials,
            &cells,
            &crate::session::AgentOptions::from_config(config, false, None, None),
        )?;
        // Both conditions that remove the whole family, asked the way `build_default` asks them.
        // A per-name `[tools]` entry is a separate question, applied per tool.
        let agent_family_registered =
            crate::tools::subagent::agent_tools_registered(&filter, config.subagents.max_depth);
        // Withheld from every session on `none`, asked the same way.
        let web_fetch_registered = crate::tools::web::web_fetch_registered(config.network);
        Ok(Self {
            registry,
            filter,
            agent_family_registered,
            web_fetch_registered,
        })
    }

    /// Every built-in with its standing, sorted by name.
    fn catalog(&self) -> Vec<crate::view::ConfiguredToolView> {
        let mut catalog = self.registry.tool_catalog();
        catalog.sort_by(|left, right| left.0.cmp(&right.0));
        catalog
            .into_iter()
            .map(|entry| self.configured(entry))
            .collect()
    }

    /// One catalog entry with what the config does to it.
    fn configured(
        &self,
        (name, description, required, deferred): (
            String,
            String,
            crate::permission::Permission,
            bool,
        ),
    ) -> crate::view::ConfiguredToolView {
        let override_entry = self.filter.permission_overrides.get(&name).copied();
        let enabled = self.filter.admits(&name)
            && (self.agent_family_registered
                || !crate::tools::subagent::AGENT_TOOL_NAMES.contains(&name.as_str()))
            && (self.web_fetch_registered || name != "web_fetch");
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
    }

    /// The built-in `name` with its schema, or the refusal naming the nearest built-in.
    fn show(&self, name: &str) -> anyhow::Result<ToolDefinitionView> {
        let catalog = self.registry.tool_catalog();
        let entry = catalog.iter().find(|entry| entry.0 == name).cloned();
        let definition = self.registry.get(name).map(|tool| tool.definition());
        let (Some(entry), Some(definition)) = (entry, definition) else {
            let hint =
                crate::tools::did_you_mean_hint(name, catalog.iter().map(|entry| entry.0.as_str()));
            let remedy = if hint.is_empty() {
                "; list them with `meka tool list`".to_string()
            } else {
                format!(".{hint}")
            };
            return Err(MekaError::Usage(format!("no tool named '{name}'{remedy}")).into());
        };
        Ok(ToolDefinitionView {
            tool: self.configured(entry),
            parameters: definition.parameters,
        })
    }
}

/// The plain rendering of `meka tool show`: the fields, then the description and the schema
/// indented under a bare heading, as `schedule show` lays out a prompt, so a line of the
/// description cannot read as another field.
fn render_shown(shown: &ToolDefinitionView) -> serde_json::Result<String> {
    let tool = &shown.tool;
    let fields = [
        ("name", tool.tool.name.clone()),
        ("permission", tool.tool.required_permission.to_string()),
        ("source", tool.permission_source.to_string()),
        (
            "status",
            status_word(tool.enabled, tool.tool.deferred).to_string(),
        ),
        ("description", String::new()),
    ];
    let mut out = crate::text::format_fields(&fields);
    for line in tool.tool.description.lines() {
        // A blank line stays blank rather than ending in the indent.
        if !line.trim().is_empty() {
            out.push_str("  ");
            out.push_str(&crate::text::sanitize_to_line(line, usize::MAX));
        }
        out.push('\n');
    }
    out.push_str("parameters:\n");
    for line in serde_json::to_string_pretty(&shown.parameters)?.lines() {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    Ok(out)
}
