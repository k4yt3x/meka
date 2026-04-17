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
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
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
        let sandboxed = self.sandbox_enabled && permission != Permission::Write;

        if sandboxed
            && matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::Unavailable
            )
        {
            return Ok(ToolOutput::text(
                "Shell command execution in read mode is not available on this \
                    platform because filesystem sandboxing is not supported. Switch to \
                    write mode (Shift+Tab) to execute commands without sandboxing."
                    .to_string(),
                true,
            ));
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
                Ok(ToolOutput::text(
                    format!("Command timed out after {}ms", timeout_ms),
                    true,
                ))
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

                // No output-length truncation here: the agent layer's
                // `persist_oversized_results` auto-persists any oversized
                // result to the scratchpad losslessly. Truncating here would
                // corrupt binary-in-base64 pipelines (see #1 in the trial
                // feedback).
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

                Ok(ToolOutput::text(
                    if result_text.is_empty() {
                        "(no output)".to_string()
                    } else {
                        result_text
                    },
                    exit_code != 0,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ContentBlock;

    fn text_content(output: &ToolOutput) -> String {
        ContentBlock::tool_result_text_content(&output.content)
    }

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
        assert_eq!(text_content(&result).trim(), "hello");
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

    #[tokio::test]
    async fn test_execute_command_large_output_not_truncated() {
        // Output well over the old 30 KB cap — the tool must return it in
        // full. The agent layer handles oversize downstream.
        let tool = ExecuteCommandTool {
            sandbox_capability: crate::sandbox::detect(),
            shared_permission: test_shared_permission(),
            sandbox_enabled: true,
        };
        let result = tool
            .execute(
                // 50 000 "x" characters + newline.
                serde_json::json!({"command": "printf 'x%.0s' {1..50000}"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            !text.contains("(output truncated"),
            "no truncation marker expected, got: {:.200}...",
            text
        );
        assert!(
            text.trim().len() >= 50_000,
            "expected >= 50 000 chars, got {}",
            text.trim().len()
        );
    }
}
