use std::path::PathBuf;

use serde::Deserialize;

use crate::cli::Cli;
use crate::permission::Permission;

#[derive(Debug, Deserialize, Default)]
pub struct ConfigFile {
    pub provider: Option<ProviderConfig>,
    pub display: Option<DisplayConfig>,
    pub web: Option<WebConfig>,
    pub shell: Option<ShellConfig>,
    pub session: Option<SessionConfig>,
}

#[derive(Debug, Deserialize, Default)]
pub struct DisplayConfig {
    pub newline_before_prompt: Option<bool>,
    pub newline_after_prompt: Option<bool>,
    pub show_session_id: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
pub struct WebConfig {
    pub user_agent: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ShellConfig {
    pub sandbox: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
pub struct SessionConfig {
    pub context_messages: Option<usize>,
    pub retention_days: Option<u64>,
    pub max_storage_bytes: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProviderConfig {
    pub name: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug)]
pub struct ResolvedConfig {
    pub provider_name: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub permission: Permission,
    pub streaming: bool,
    pub session_id: Option<uuid::Uuid>,
    pub continue_last: bool,
    pub prompt: Option<String>,
    pub newline_before_prompt: bool,
    pub newline_after_prompt: bool,
    pub show_session_id: bool,
    pub user_agent: String,
    pub sandbox: bool,
    pub context_messages: Option<usize>,
    pub retention_days: Option<u64>,
    pub max_storage_bytes: Option<u64>,
}

fn config_file_path() -> Option<PathBuf> {
    dirs::config_dir().map(|directory| directory.join("agsh").join("config.toml"))
}

fn load_config_file() -> ConfigFile {
    let Some(path) = config_file_path() else {
        return ConfigFile::default();
    };

    match std::fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str(&contents) {
            Ok(config) => config,
            Err(error) => {
                tracing::warn!("failed to parse config file {}: {}", path.display(), error);
                ConfigFile::default()
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ConfigFile::default(),
        Err(error) => {
            tracing::warn!("failed to read config file {}: {}", path.display(), error);
            ConfigFile::default()
        }
    }
}

impl ResolvedConfig {
    pub fn from_cli(cli: &Cli) -> Self {
        let config_file = load_config_file();
        let file_provider = config_file.provider.unwrap_or_default();
        let file_display = config_file.display.unwrap_or_default();
        let file_web = config_file.web.unwrap_or_default();
        let file_shell = config_file.shell.unwrap_or_default();
        let file_session = config_file.session.unwrap_or_default();

        let provider_name = cli
            .provider
            .clone()
            .or_else(|| std::env::var("AGSH_PROVIDER").ok())
            .or_else(|| file_provider.name.clone());

        let model = cli
            .model
            .clone()
            .or_else(|| std::env::var("AGSH_MODEL").ok())
            .or_else(|| file_provider.model.clone());

        let api_key = resolve_api_key(provider_name.as_deref(), &file_provider);

        let base_url = cli
            .base_url
            .clone()
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .or_else(|| file_provider.base_url.clone());

        let permission = cli
            .permission
            .or_else(|| {
                std::env::var("AGSH_PERMISSION")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(Permission::Read);

        Self {
            provider_name,
            model,
            api_key,
            base_url,
            permission,
            streaming: !cli.no_stream,
            session_id: cli.session_id,
            continue_last: cli.continue_last,
            prompt: cli.prompt.clone(),
            newline_before_prompt: file_display.newline_before_prompt.unwrap_or(true),
            newline_after_prompt: file_display.newline_after_prompt.unwrap_or(true),
            show_session_id: file_display.show_session_id.unwrap_or(false),
            user_agent: file_web
                .user_agent
                .unwrap_or_else(|| "Mozilla/5.0 (compatible; agsh/0.1)".to_string()),
            sandbox: file_shell.sandbox.unwrap_or(true),
            context_messages: file_session.context_messages.or(Some(200)),
            retention_days: file_session.retention_days.or(Some(90)),
            max_storage_bytes: file_session.max_storage_bytes.or(Some(52_428_800)),
        }
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        if self.provider_name.is_none() {
            return Err(crate::error::AgshError::Config(
                "no provider configured. Set --provider, AGSH_PROVIDER env var, \
                 or provider.name in config file (~/.config/agsh/config.toml)"
                    .to_string(),
            ));
        }
        if self.model.is_none() {
            return Err(crate::error::AgshError::Config(
                "no model configured. Set --model, AGSH_MODEL env var, \
                 or provider.model in config file (~/.config/agsh/config.toml)"
                    .to_string(),
            ));
        }
        if self.api_key.is_none() {
            return Err(crate::error::AgshError::Config(
                "no API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY env var, \
                 or provider.api_key in config file (~/.config/agsh/config.toml)"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

fn resolve_api_key(provider_name: Option<&str>, file_provider: &ProviderConfig) -> Option<String> {
    match provider_name {
        Some("anthropic") => std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .or_else(|| file_provider.api_key.clone()),
        Some("openai") | _ => std::env::var("OPENAI_API_KEY")
            .ok()
            .or_else(|| file_provider.api_key.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_file_deserialization() {
        let toml_str = r#"
[provider]
name = "openai"
model = "gpt-4o"
api_key = "sk-test"
base_url = "https://api.openai.com/v1"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let provider = config.provider.expect("provider should be present");
        assert_eq!(provider.name.as_deref(), Some("openai"));
        assert_eq!(provider.model.as_deref(), Some("gpt-4o"));
        assert_eq!(provider.api_key.as_deref(), Some("sk-test"));
        assert_eq!(
            provider.base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );
    }

    #[test]
    fn test_empty_config_file() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.provider.is_none());
    }

    #[test]
    fn test_partial_config_file() {
        let toml_str = r#"
[provider]
name = "anthropic"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let provider = config.provider.expect("provider should be present");
        assert_eq!(provider.name.as_deref(), Some("anthropic"));
        assert!(provider.model.is_none());
        assert!(provider.api_key.is_none());
        assert!(provider.base_url.is_none());
    }

    #[test]
    fn test_session_config_deserialization() {
        let toml_str = r#"
[session]
context_messages = 100
retention_days = 90
max_storage_bytes = 52428800
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let session = config.session.expect("session should be present");
        assert_eq!(session.context_messages, Some(100));
        assert_eq!(session.retention_days, Some(90));
        assert_eq!(session.max_storage_bytes, Some(52428800));
    }

    #[test]
    fn test_session_config_partial() {
        let toml_str = r#"
[session]
context_messages = 50
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let session = config.session.expect("session should be present");
        assert_eq!(session.context_messages, Some(50));
        assert!(session.retention_days.is_none());
        assert!(session.max_storage_bytes.is_none());
    }

    #[test]
    fn test_session_defaults_applied() {
        let file_session = SessionConfig::default();
        let context_messages = file_session.context_messages.or(Some(200));
        let retention_days = file_session.retention_days.or(Some(90));
        let max_storage_bytes = file_session.max_storage_bytes.or(Some(52_428_800));

        assert_eq!(context_messages, Some(200));
        assert_eq!(retention_days, Some(90));
        assert_eq!(max_storage_bytes, Some(52_428_800));
    }

    #[test]
    fn test_session_config_overrides_defaults() {
        let toml_str = r#"
[session]
context_messages = 50
retention_days = 30
max_storage_bytes = 10485760
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let file_session = config.session.unwrap_or_default();
        let context_messages = file_session.context_messages.or(Some(200));
        let retention_days = file_session.retention_days.or(Some(90));
        let max_storage_bytes = file_session.max_storage_bytes.or(Some(52_428_800));

        assert_eq!(context_messages, Some(50));
        assert_eq!(retention_days, Some(30));
        assert_eq!(max_storage_bytes, Some(10_485_760));
    }
}
