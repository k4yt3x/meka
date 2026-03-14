use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::ToolDefinition;

use super::util::require_str;
use super::{Tool, ToolOutput};

pub(super) struct ExecuteCommandTool {
    pub sandbox_capability: crate::sandbox::SandboxCapability,
    pub shared_permission: crate::permission::SharedPermission,
    pub sandbox_enabled: bool,
}

#[async_trait]
impl Tool for ExecuteCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "execute_command".to_string(),
            description: "Execute a shell command and return its output. In read mode, \
                the command runs in a read-only sandbox where filesystem writes are blocked."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Timeout in milliseconds. Defaults to 30000 (30 seconds)."
                    }
                },
                "required": ["command"]
            }),
        }
    }

    fn required_permission(&self) -> Permission {
        if self.sandbox_enabled
            && !matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::Unavailable
            )
        {
            Permission::Read
        } else {
            Permission::Write
        }
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let command = require_str(&input, "command", "execute_command")?;
        let timeout_ms = input["timeout_ms"].as_u64().unwrap_or(30000);
        let permission = self.shared_permission.get();
        let sandboxed = self.sandbox_enabled && permission < Permission::Write;

        if sandboxed
            && matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::Unavailable
            )
        {
            return Ok(ToolOutput {
                content: "Shell command execution in read mode is not available on this \
                    platform because filesystem sandboxing is not supported. Switch to \
                    write mode (Shift+Tab) to execute commands without sandboxing."
                    .to_string(),
                is_error: true,
            });
        }

        #[cfg(windows)]
        let mut command_builder = {
            let mut cmd = tokio::process::Command::new("powershell");
            cmd.arg("-Command").arg(&command);
            cmd
        };

        #[cfg(target_os = "macos")]
        let mut command_builder = if sandboxed
            && matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::SandboxExec
            ) {
            let mut cmd = tokio::process::Command::new("sandbox-exec");
            cmd.arg("-p")
                .arg(crate::sandbox::SANDBOX_PROFILE_READONLY)
                .arg("sh")
                .arg("-c")
                .arg(&command);
            cmd
        } else {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(&command);
            cmd
        };

        #[cfg(target_os = "linux")]
        let mut command_builder = {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(&command);
            if sandboxed
                && let crate::sandbox::SandboxCapability::Landlock { abi_version } =
                    self.sandbox_capability
            {
                unsafe {
                    cmd.pre_exec(move || {
                        crate::sandbox::apply_landlock_readonly(abi_version)
                            .map_err(std::io::Error::from_raw_os_error)
                    });
                }
            }
            cmd
        };

        let mut child = command_builder
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "execute_command".to_string(),
                message: format!("failed to spawn command: {}", error),
            })?;

        let timeout_duration = std::time::Duration::from_millis(timeout_ms);

        // wait_with_output() consumes the child, so use wait() + manual
        // stdout/stderr reading instead to allow kill on cancellation.
        tokio::select! {
            _ = cancellation.cancelled() => {
                if let Err(error) = child.kill().await {
                    tracing::debug!("failed to kill child process: {}", error);
                }
                Err(AgshError::Interrupted)
            }
            _ = tokio::time::sleep(timeout_duration) => {
                if let Err(error) = child.kill().await {
                    tracing::debug!("failed to kill child process: {}", error);
                }
                Ok(ToolOutput {
                    content: format!("Command timed out after {}ms", timeout_ms),
                    is_error: true,
                })
            }
            status = child.wait() => {
                let status = status.map_err(|error| AgshError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message: format!("failed to wait for command: {}", error),
                })?;

                let exit_code = status.code().unwrap_or(-1);

                let mut stdout_content = String::new();
                let mut stderr_content = String::new();

                if let Some(mut stdout) = child.stdout.take() {
                    use tokio::io::AsyncReadExt;
                    if let Err(error) = stdout.read_to_string(&mut stdout_content).await {
                        tracing::debug!("failed to read stdout: {}", error);
                    }
                }
                if let Some(mut stderr) = child.stderr.take() {
                    use tokio::io::AsyncReadExt;
                    if let Err(error) = stderr.read_to_string(&mut stderr_content).await {
                        tracing::debug!("failed to read stderr: {}", error);
                    }
                }

                let mut result_text = String::new();
                if !stdout_content.is_empty() {
                    result_text.push_str(&stdout_content);
                }
                if !stderr_content.is_empty() {
                    if !result_text.is_empty() {
                        result_text.push_str("\n--- stderr ---\n");
                    }
                    result_text.push_str(&stderr_content);
                }
                if exit_code != 0 {
                    result_text.push_str(&format!("\nExit code: {}", exit_code));
                }

                Ok(ToolOutput {
                    content: if result_text.is_empty() {
                        "(no output)".to_string()
                    } else {
                        result_text
                    },
                    is_error: exit_code != 0,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_shared_permission() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(Permission::Write)
    }

    #[tokio::test]
    async fn test_execute_command() {
        let tool = ExecuteCommandTool {
            sandbox_capability: crate::sandbox::detect(),
            shared_permission: test_shared_permission(),
            sandbox_enabled: true,
        };
        let result = tool
            .execute(
                serde_json::json!({"command": "echo hello"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert_eq!(result.content.trim(), "hello");
    }

    #[tokio::test]
    async fn test_execute_command_failure() {
        let tool = ExecuteCommandTool {
            sandbox_capability: crate::sandbox::detect(),
            shared_permission: test_shared_permission(),
            sandbox_enabled: true,
        };
        let result = tool
            .execute(
                serde_json::json!({"command": "false"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
    }
}
