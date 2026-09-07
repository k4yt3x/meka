//! `session/prompt`: content conversion, local slash commands, and the turn itself.

use super::*;

/// Append one prompt content block to `prompt_text`, inserting a newline separator between blocks.
pub(super) fn push_prompt_block(prompt_text: &mut String, block: &str) {
    if !prompt_text.is_empty() {
        prompt_text.push('\n');
    }
    prompt_text.push_str(block);
}
/// Append a ` mime="..."` attribute to a resource/resource_link tag when one is present.
pub(super) fn push_mime_attr(tag: &mut String, mime: &Option<String>) {
    if let Some(mime) = mime {
        tag.push_str(&format!(" mime=\"{mime}\""));
    }
}
/// Render an ACP embedded resource (an @-mention's inlined contents) as a `<resource>` tag for the
/// prompt body. Text resources inline their contents; binary (blob) resources emit a self-closing
/// marker without the (potentially huge) payload, so the model still learns the reference exists.
///
/// A distinct `<resource>` tag (not `<context>`) is deliberate: `<context>` is what the agent's own
/// per-turn block is called, and a prompt must not read as one. A `<resource>` tag round-trips
/// through history replay exactly like `<resource_link>` does.
pub(super) fn format_embedded_resource(embedded: &EmbeddedResource) -> String {
    match &embedded.resource {
        EmbeddedResourceResource::TextResourceContents(text) => {
            let mut tag = format!("<resource uri=\"{}\"", text.uri);
            push_mime_attr(&mut tag, &text.mime_type);
            tag.push('>');
            tag.push_str(&text.text);
            tag.push_str("</resource>");
            tag
        }
        EmbeddedResourceResource::BlobResourceContents(blob) => {
            let mut tag = format!("<resource uri=\"{}\"", blob.uri);
            push_mime_attr(&mut tag, &blob.mime_type);
            tag.push_str(" encoding=\"base64\"/>");
            tag
        }
        // `EmbeddedResourceResource` is `#[non_exhaustive]`; a future variant we can't introspect
        // still gets a bare marker so the prompt stays well-formed.
        _ => "<resource/>".to_string(),
    }
}
/// Decode an ACP image content block into meka's internal [`crate::image::ImageSource`] via the
/// shared client-image pipeline, so ACP and the HTTP API enforce the same limits.
///
/// Off the runtime, for the reason `read_file` and `fetch_url` document at their own call sites:
/// the pipeline base64-decodes and then decodes the image to verify it, which is tens of
/// milliseconds of pure CPU on a multi-megapixel screenshot, and on the runtime it blocks every
/// other task on that worker. The editor pasting one attachment must not stall an unrelated
/// session's stream.
pub(super) async fn decode_acp_image(
    image: &ImageContent,
) -> Result<crate::image::ImageSource, String> {
    let data = image.data.clone();
    let mime_type = image.mime_type.clone();
    tokio::task::spawn_blocking(move || crate::image::decode_base64_image(&data, &mime_type))
        .await
        .map_err(|error| format!("image decode task failed: {error}"))?
}

/// If `prompt_text` is a [`crate::host::COMMANDS`] invocation, render its output; otherwise `None`
/// so the prompt falls through to skill resolution / the model. Text after the command name is
/// ignored (these commands take no arguments).
pub(super) async fn try_local_command(
    prompt_text: &str,
    agent: &crate::agent::Agent,
    conversation_len: usize,
    permission: &crate::permission::SharedPermission,
    shared: &crate::host::SharedDeps,
) -> Option<String> {
    let (name, _extra) = split_acp_slash(prompt_text)?;
    match name.as_str() {
        "status" => Some(build_status_text(
            agent,
            conversation_len,
            permission,
            shared,
        )),
        "mcp" => Some(build_mcp_list_text(shared).await),
        "usage" => Some(build_usage_text(agent).await),
        _ => None,
    }
}
/// Plain-text `/usage` output for ACP clients, reusing the shared `render::format_account_usage`.
pub(super) async fn build_usage_text(agent: &crate::agent::Agent) -> String {
    match agent.fetch_usage().await {
        Ok(Some(usage)) => crate::render::format_account_usage(&usage),
        Ok(None) => "Account usage isn't available for this backend.".to_string(),
        Err(error) => format!("Error fetching usage: {error}"),
    }
}
/// Plain-text `/status` output: the block the REPL prints, from [`crate::host::format_status`],
/// under a heading and after the permission level, which an ACP client may not otherwise surface.
pub(super) fn build_status_text(
    agent: &crate::agent::Agent,
    conversation_len: usize,
    permission: &crate::permission::SharedPermission,
    shared: &crate::host::SharedDeps,
) -> String {
    format!(
        "Session status\n  Permission:      {}\n{}",
        level_display_name(permission.get()),
        crate::host::format_status(agent, &shared.providers, conversation_len)
    )
    .trim_end()
    .to_string()
}
/// Plain-text `/mcp` output: each configured MCP server and its live connection state.
pub(super) async fn build_mcp_list_text(shared: &crate::host::SharedDeps) -> String {
    let Some(manager) = shared.mcp_manager.as_ref() else {
        return "No MCP servers configured.".to_string();
    };
    let names = manager.server_names();
    if names.is_empty() {
        return "No MCP servers configured.".to_string();
    }
    let mut out = String::from("MCP servers\n");
    for name in names {
        let status = match manager.server_entry(&name) {
            Some(entry) => match entry.state().await {
                crate::mcp::ServerState::Failed { error, .. } => {
                    format!("failed: {}", error.lines().next().unwrap_or("").trim())
                }
                other => other.label().to_string(),
            },
            None => "unknown".to_string(),
        };
        out.push_str(&format!("  {name}: {status}\n"));
    }
    out.truncate(out.trim_end().len());
    out
}
/// Emit a `session/update: available_commands_update` listing the [`crate::host::COMMANDS`] marked
/// `for_editors`, followed by every installed skill, each as an [`AvailableCommand`].
/// Editor clients render these as slash commands in their prompt input; picking one inserts
/// `/<name> `. Skills whose name collides with a built-in command are dropped so the palette has no
/// duplicates.
///
/// `SkillCache::current` is mtime-cached, so calling this at the top of every prompt is cheap (one
/// `read_dir`, no parsing on the warm path).
pub(super) async fn emit_available_commands(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    skills: &Arc<SkillCache>,
) {
    let snapshot = skills.current().await;
    let mut commands: Vec<AvailableCommand> = crate::host::COMMANDS
        .iter()
        .filter(|command| command.for_editors)
        .map(|command| AvailableCommand::new(command.name, command.help))
        .collect();
    commands.extend(
        snapshot
            .skills
            .iter()
            .filter(|skill| {
                !crate::host::COMMANDS
                    .iter()
                    .any(|command| command.for_editors && command.name == skill.name)
            })
            .map(|skill| {
                // Sanitized, like every other place a skill description is shown. The store hands
                // back the file's bytes now, so this is a render boundary: a description carrying a
                // bidi override or a control character would otherwise reach the editor's command
                // palette over JSON-RPC and be drawn by whatever renders it.
                AvailableCommand::new(
                    skill.name.clone(),
                    crate::memory::render_description_for_model(&skill.description),
                )
                .input(AvailableCommandInput::Unstructured(
                    UnstructuredCommandInput::new("additional context (optional)"),
                ))
            }),
    );
    send_session_update(
        connection,
        session_id,
        SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(commands)),
    );
}
/// Outcome of running a slash-command parse against an ACP `session/prompt`'s text. Carries enough
/// detail for the prompt handler to either continue with the resolved text or surface a JSON-RPC
/// error explaining what went wrong.
#[derive(Debug)]
pub(super) enum SlashInvocationError {
    /// The already-composed reason from [`crate::skills::SkillIndex::unavailable`], so this path
    /// distinguishes a name nobody wrote from a `SKILL.md` that will not parse rather than calling
    /// both "unknown skill".
    SkillNotFound(String),
    SkillLoadFailed {
        name: String,
        source: String,
    },
}
impl std::fmt::Display for SlashInvocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlashInvocationError::SkillNotFound(reason) => write!(f, "{reason}"),
            SlashInvocationError::SkillLoadFailed { name, source } => {
                write!(f, "failed to load skill '{name}': {source}")
            }
        }
    }
}
/// Split an ACP prompt that looks like `/<name> [extra]` into the command name and the remainder.
/// Returns `None` if the input isn't in that shape, i.e. doesn't start with `/`, has only
/// whitespace after the slash, or contains a newline before the first whitespace (heuristic: a real
/// slash command never spans lines, but pasted content might).
pub(super) fn split_acp_slash(prompt_text: &str) -> Option<(String, String)> {
    let rest = prompt_text.strip_prefix('/')?;
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        return None;
    }
    Some(match rest.split_once(char::is_whitespace) {
        Some((name, extra)) => (name.to_string(), extra.trim().to_string()),
        None => (rest.to_string(), String::new()),
    })
}
/// Intercept `/<skill-name> [extra]` invocations in an ACP prompt's
/// text. Returns the text the agent should actually run with:
///
/// - Non-slash input: returned unchanged.
/// - Slash followed by a name that isn't a syntactically valid skill identifier (e.g. a pasted path
///   like `/etc/hosts` or a `//` comment): returned unchanged so the model can see it.
/// - `/<skill-name>` matching an installed skill: returns `extra\n\n{body}` where `body` is
///   [`crate::skills::load_skill_body`]'s output (the skill's base-directory header followed by its
///   body verbatim). Empty `extra` collapses to just `body`. Same composition the REPL's
///   `SlashCommand::SkillInvoke` handler uses; named rather than cited by line, because a line
///   number is a cross-reference that rots on the next edit and this one already had.
/// - `/<name>` with a syntactically valid skill name but no installed skill of that name:
///   `SkillNotFound`. The caller sends the original text to the model rather than failing the
///   prompt, because `/usr local lib` parses the same way.
pub(super) async fn slash_to_prompt_text(
    prompt_text: String,
    skills: &Arc<SkillCache>,
) -> Result<String, SlashInvocationError> {
    let Some((name, extra)) = split_acp_slash(&prompt_text) else {
        return Ok(prompt_text);
    };
    // Anything that doesn't even look like a skill identifier was never going to match. Pass
    // through so pasted paths and code comments reach the model unchanged.
    //
    // Narrower than the rule the delete doors apply, on purpose: those decide whether a name is
    // safe to act on, and this decides whether the user meant a skill at all. A skill whose name
    // predates the spec still resolves (`/My_Skill`), but a line like `/v1.2 of the API` stays
    // prose rather than becoming "no such skill".
    if !crate::skills::looks_like_skill_invocation(&name) {
        return Ok(prompt_text);
    }
    let snapshot = skills.current().await;
    let Some(skill) = snapshot.find(&name) else {
        return Err(SlashInvocationError::SkillNotFound(
            snapshot.unavailable(&name),
        ));
    };
    let body = crate::skills::load_skill_body(skill)
        .await
        .map_err(|source| SlashInvocationError::SkillLoadFailed {
            name: name.clone(),
            source,
        })?;
    Ok(if extra.is_empty() {
        body
    } else {
        format!("{extra}\n\n{body}")
    })
}
/// Body of the `session/prompt` spawn. Extracted so the closure stays thin. Owns `responder` and
/// replies exactly once.
///
/// Lock ordering: take the outer `sessions` read lock briefly to clone the per-session
/// `Arc<Mutex<SessionRuntime>>`, drop it, then hold *only* the per-session mutex for the duration
/// of the turn. Cancel and other sessions remain unblocked.
#[allow(
    clippy::significant_drop_tightening,
    reason = "the runtime guard is the turn's exclusivity and lives to the end on purpose"
)]
pub(super) async fn run_prompt_turn(
    state: Arc<ServerState>,
    request: PromptRequest,
    responder: agent_client_protocol::Responder<PromptResponse>,
    admitted: Option<crate::host::TurnGuard>,
) -> Result<(), agent_client_protocol::Error> {
    // Accept `text` + `resource_link` (the ACP baseline) + embedded `resource` and `image`. Whether
    // this session may carry an image is asked below, once its profile has been applied; other
    // content variants are refused here.
    let mut prompt_text = String::new();
    let mut images: Vec<crate::image::ImageSource> = Vec::new();
    for block in &request.prompt {
        match block {
            ContentBlock::Text(text) => {
                push_prompt_block(&mut prompt_text, &text.text);
            }
            // `ResourceLink` is part of the ACP baseline that every agent MUST support (alongside
            // `Text`). meka doesn't fetch the resource server-side; the model sees the reference as
            // a structured tag carrying the link's name, uri, and (optional) description so it can
            // decide what to do with it.
            ContentBlock::ResourceLink(link) => {
                let mut tag =
                    format!("<resource_link name=\"{}\" uri=\"{}\"", link.name, link.uri,);
                push_mime_attr(&mut tag, &link.mime_type);
                tag.push('>');
                if let Some(description) = &link.description {
                    tag.push_str(description);
                }
                tag.push_str("</resource_link>");
                push_prompt_block(&mut prompt_text, &tag);
            }
            // `Resource` carries an @-mention's inlined contents (the `embedded_context`
            // capability). meka surfaces it to the model as a `<resource>` tag rather than fetching
            // anything server-side, mirroring `ResourceLink`.
            ContentBlock::Resource(embedded) => {
                push_prompt_block(&mut prompt_text, &format_embedded_resource(embedded));
            }
            // Decoded here for its payload alone, through the shared image pipeline so the size
            // cap and format conversion match tool-result images. Whether this session admits an
            // image at all is a question for its profile, asked below once that has been applied.
            ContentBlock::Image(image) => match decode_acp_image(image).await {
                Ok(source) => images.push(source),
                Err(message) => {
                    return responder.respond_with_error(invalid_params_error(format!(
                        "invalid image content block: {message}"
                    )));
                }
            },
            _ => {
                return responder.respond_with_error(invalid_params_error(
                    "meka acp accepts `text`, `resource_link`, `resource`, and (when the \
                     profile has vision enabled) `image` content blocks in `prompt`; `audio` is \
                     not supported",
                ));
            }
        }
    }
    let has_images = !images.is_empty();

    // Ahead of the session being taken, so a prompt with nothing in it never holds the runtime
    // mutex or publishes a cancel token: a client that pipelines a real prompt behind it must find
    // the session free, not busy. A prompt of nothing but whitespace costs a provider round-trip to
    // produce nothing, and a client that dropped its content on the floor should hear about it.
    // The words are expanded below once a slash command has been resolved.
    let input = match crate::agent::TurnInput::from_parts(prompt_text.clone(), images) {
        Ok(input) => input,
        Err(error) => return responder.respond_with_error(acp_error_for(&error, false)),
    };

    // Look up the target session by id under the outer read lock, clone the entry (cheap, two
    // `Arc`s), drop the outer guard. From here on, only the per-session runtime mutex is held; the
    // sibling cancellation cell is accessible to the cancel handler throughout the turn.
    let session_id_str = request.session_id.0.as_ref().to_string();
    let session_uuid = match parse_session_id(&session_id_str) {
        Ok(uuid) => uuid,
        Err(error) => return responder.respond_with_error(error),
    };
    let entry = {
        let sessions = state.sessions.read().await;
        match sessions.get(&session_id_str) {
            Some(entry) => {
                entry.touch();
                entry.clone()
            }
            None => {
                return responder.respond_with_error(acp_error_for(
                    &MekaError::SessionNotFound(session_uuid),
                    false,
                ));
            }
        }
    };

    // Acquire the runtime mutex non-blocking. If another prompt is already in flight for this
    // session, reject explicitly: ACP models one prompt at a time per session and silent queueing
    // also enables a race against the sibling cancellation cell (the second prompt would overwrite
    // the first's token before the first finishes, so `session/cancel` would target the wrong
    // turn). The lock guard is held for the entire turn so the token written below cannot be
    // overwritten by a sibling request, and per-session pre-work serializes naturally.
    let mut conversation = match entry.conversation.try_lock() {
        Ok(guard) => guard,
        // `InvalidParams`, which is what clients already handle for a second prompt on a session
        // with one in flight; JSON-RPC has no busy code, and `InternalError` would read as a fault
        // in the server. The holder may be a scheduled turn or a background delivery rather than a
        // prompt, and a retry resolves either.
        Err(_) => {
            return responder.respond_with_error(acp_error_for(
                &MekaError::TurnInFlight { doing: "prompt" },
                false,
            ));
        }
    };

    // Under the lock and before the first round, so a switch made while the previous turn held this
    // mutex takes effect on this one. Refused rather than run on the old profile: the client asked
    // for a specific account, and answering from another one silently is the failure the whole
    // per-session binding exists to prevent.
    if let Err(error) = apply_recorded_profile(&state, &entry.agent, entry.id).await {
        return responder.respond_with_error(build_failure_error(
            "cannot run this turn on the profile this session is recorded against",
            &error,
            state.shared.relay_provider_errors(),
        ));
    }
    // Once the binding is applied, so the question is asked of the profile this turn runs on: a
    // `session/set_config_option` switch reaches the agent on the line above and nowhere earlier,
    // and a session moved onto a text-only profile must refuse the attachment even on a connection
    // whose `initialize` advertised `image`.
    if has_images && !entry.accepts_images() {
        return responder.respond_with_error(invalid_params_error(
            "image content blocks require a profile with vision enabled; set `vision = true` under \
             `[profiles.<name>]` or send text only",
        ));
    }

    // Install a fresh cancellation token inside the locked scope so the cancel handler (which reads
    // the sibling cell) always sees the token for the turn currently using the runtime. The guard
    // lives to the end of this function, which is the end of the turn.
    let cancellation = CancellationToken::new();
    // Admitted when the prompt was dispatched, so a `session/cancel` sent straight after it stops
    // this turn rather than the next one the user submits. A turn no dispatcher admitted (a
    // scheduled fire, a resumed one) is admitted here, with no window to close.
    let _published = entry.cancel.publish(
        cancellation.clone(),
        admitted
            .as_ref()
            .map(|guard| guard.admission)
            .unwrap_or_else(|| entry.cancel.admit()),
    );

    // Refresh the slash-command palette before the prompt body resolves. This uses the per-session
    // frontend so the notification routes to the right ACP connection.
    let frontend = Arc::clone(&entry.frontend);
    emit_available_commands(
        &frontend.connection,
        &frontend.session_id,
        &state.shared.skills,
    )
    .await;

    // Local slash commands (`/status`, `/mcp`, `/usage`) render text and end the turn with no model
    // call. Checked before skill resolution so they aren't misread as unknown skills.
    if let Some(output) = try_local_command(
        &prompt_text,
        &entry.agent,
        conversation.len(),
        &entry.cells().permission,
        &state.shared,
    )
    .await
    {
        // `agent_message_chunk` is rendered as Markdown, where a bare newline is a soft break (it
        // collapses to a space) and small indents are stripped. Wrap the preformatted table in a
        // fenced code block so the column alignment and line breaks survive, matching how
        // `execute_command` output is rendered.
        let body = format!("```\n{output}\n```");
        send_session_update(
            &frontend.connection,
            &frontend.session_id,
            SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                agent_client_protocol::schema::v1::TextContent::new(body),
            ))),
        );
        // Reported here too, even though a local command consumes no tokens: `usage` is
        // session-cumulative, so omitting it would make a client's running total flicker away on
        // every `/status` turn.
        return responder
            .respond(PromptResponse::new(StopReason::EndTurn).usage(session_usage(&entry.agent)));
    }

    let original_prompt_text = prompt_text.clone();
    let prompt_text = match slash_to_prompt_text(prompt_text, &state.shared.skills).await {
        Ok(text) => text,
        Err(SlashInvocationError::SkillNotFound(name)) => {
            // `slash_to_prompt_text` only returns `SkillNotFound` for strings whose first token is
            // a syntactically-valid skill name. That's deliberately a narrow filter, but it still
            // false-positives on pasted text like `/usr local lib` (parses as name=`usr`,
            // extra=`local lib`). Treat "no such skill" as "this wasn't a skill invocation after
            // all" and feed the original text to the model. It can respond with "I don't know that
            // command" if the user really meant `/<name>`. The alternative (hard-error) breaks
            // paste UX for any string starting with `/word`.
            tracing::debug!(
                "session/prompt: '/{name}' didn't match a registered skill; passing through"
            );
            original_prompt_text
        }
        Err(SlashInvocationError::SkillLoadFailed { name, source }) => {
            // The skill name was valid and matched an installed skill; the failure is a server-side
            // problem reading the body (disk I/O, permission, etc.). JSON-RPC `InternalError` is
            // the correct classification; `InvalidParams` would mislead the client into thinking
            // the user's request was malformed. The reason names a path, so it stays in the log.
            tracing::warn!("failed to load skill '{name}': {source}");
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to load skill '{name}'"),
            ));
        }
    };

    let agent = &entry.agent;
    let session_uuid = &entry.id;
    let messages = &mut *conversation;
    let input = input.with_words(prompt_text);
    // An outcome that did not warrant a turn of its own rides on this one, ahead of the editor's
    // prompt rather than as a message of its own: see `background::render_outcomes_riding`.
    let outcomes = if state.shared.config.background.enabled {
        crate::host::claim_undelivered_outcomes(agent, &state.shared.store, *session_uuid).await
    } else {
        Vec::new()
    };
    if !outcomes.is_empty() {
        // Shown to the editor as well as sent, as the scheduled-fire path does: the model answers
        // about this text, and a transcript without it reads as an answer to a question nobody
        // asked. Pushed before the prompt, which is the order the model receives them in.
        entry
            .frontend
            .push_out_of_band_prompt(&crate::background::render_outcomes(&outcomes));
    }
    let input = input.riding(outcomes);
    // Clone the cancellation token so we can probe `is_cancelled()` after the call returns. The
    // spec mandates that any cancel arriving during a turn must surface as `StopReason::Cancelled`,
    // even when the cancellation manifests as a provider / tool error rather than the clean
    // `MekaError::Interrupted` path.
    let cancel_probe = cancellation.clone();
    let result = agent.run_turn(messages, input, cancellation).await;

    let stop_reason = match result {
        Ok(crate::agent::TurnOutcome::EndTurn) => StopReason::EndTurn,
        Ok(crate::agent::TurnOutcome::MaxTokens) => StopReason::MaxTokens,
        Ok(crate::agent::TurnOutcome::Refusal(_)) => StopReason::Refusal,
        Err(MekaError::Interrupted) => StopReason::Cancelled,
        Err(error) => {
            if cancel_probe.is_cancelled() {
                StopReason::Cancelled
            } else {
                // Through the one classifier, which decides what of a failed turn an editor may
                // read. Formatted here instead, every variant's `Display` went out verbatim: an
                // upstream body naming the operator's account regardless of
                // `relay_provider_errors`, an MCP connector's spawn command line, and a
                // `Database` error naming the store's path.
                return responder.respond_with_error(acp_error_for(
                    &error,
                    state.shared.relay_provider_errors(),
                ));
            }
        }
    };

    // The first user message defines the session title; push it once now that the turn has run and
    // that message is in the conversation.
    maybe_emit_session_title(
        &frontend.connection,
        &frontend.session_id,
        &entry.title_sent,
        messages,
    );

    responder.respond(PromptResponse::new(stop_reason).usage(session_usage(agent)))
}
/// Session-cumulative token counts for `session/prompt`'s response. Complements the per-turn
/// `usage_update` notification, which carries the context gauge rather than these totals.
pub(super) fn session_usage(agent: &Agent) -> Usage {
    let snapshot = agent.session_stats_snapshot();
    let mut usage = Usage::new(
        snapshot
            .total_input_tokens()
            .saturating_add(snapshot.output_tokens),
        snapshot.input_tokens,
        snapshot.output_tokens,
    );
    usage.cached_read_tokens = Some(snapshot.cache_read_input_tokens);
    usage.cached_write_tokens = Some(snapshot.cache_creation_input_tokens);
    // `thought_tokens` is left at its `None` default: meka doesn't meter reasoning separately from
    // output, and reporting a made-up split would be worse than reporting none.
    usage
}
