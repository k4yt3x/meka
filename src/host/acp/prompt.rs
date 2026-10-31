//! `session/prompt`: content conversion, local slash commands, and the turn itself.

use super::*;

/// What an editor's prompt says as one string, its parts' text a newline apart, for the local
/// commands to recognize: a command with a resource attached is not one.
pub(super) fn sent_text(parts: &[crate::conversation::ContentBlock]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            crate::conversation::ContentBlock::Text { text }
            | crate::conversation::ContentBlock::Resource { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}
/// The name a resource is shown by when the editor sent its contents without one: the last
/// segment of its URI, which for a file is its name.
fn resource_name(uri: &str) -> String {
    uri.trim_end_matches('/')
        .rsplit('/')
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or(uri)
        .to_string()
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
/// An ACP embedded resource as the block the turn's message carries: the tag
/// [`format_embedded_resource`] renders, beside the URI and the name an editor shows it by.
pub(super) fn embedded_resource_block(
    embedded: &EmbeddedResource,
) -> crate::conversation::ContentBlock {
    let uri = match &embedded.resource {
        EmbeddedResourceResource::TextResourceContents(text) => text.uri.clone(),
        EmbeddedResourceResource::BlobResourceContents(blob) => blob.uri.clone(),
        _ => String::new(),
    };
    // A kind this build cannot read names nothing, and is shown by what it is.
    let name = if uri.is_empty() {
        "resource".to_string()
    } else {
        resource_name(&uri)
    };
    crate::conversation::ContentBlock::Resource {
        name,
        uri,
        text: format_embedded_resource(embedded),
    }
}
/// An ACP resource link as the block the turn's message carries: a `<resource_link>` tag
/// carrying the link's name, URI and description, since meka does not fetch it.
pub(super) fn resource_link_block(link: &ResourceLink) -> crate::conversation::ContentBlock {
    let mut tag = format!("<resource_link name=\"{}\" uri=\"{}\"", link.name, link.uri);
    push_mime_attr(&mut tag, &link.mime_type);
    tag.push('>');
    if let Some(description) = &link.description {
        tag.push_str(description);
    }
    tag.push_str("</resource_link>");
    crate::conversation::ContentBlock::Resource {
        uri: link.uri.clone(),
        name: link.name.clone(),
        text: tag,
    }
}
/// Decode an ACP image content block into meka's internal [`crate::image::ImageSource`] via the
/// shared client-image pipeline, so ACP and the HTTP API enforce the same limits.
///
/// Off the runtime, for the reason `file_read` and `web_fetch` document at their own call sites:
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
        .map_err(|error| format!("failed to decode the image: {error}"))?
}

/// If `prompt_text` is a [`crate::host::COMMANDS`] invocation, render its output; otherwise `None`
/// so the prompt falls through to skill resolution / the model. These commands take no argument,
/// so one followed by words is prose for the model, as `/usage looks wrong, why?` is.
pub(super) async fn try_local_command(
    prompt_text: &str,
    agent: &crate::agent::Agent,
    conversation_len: usize,
    permission: &crate::permission::SharedPermission,
    shared: &crate::host::SharedDeps,
) -> Option<String> {
    let (name, extra) = split_acp_slash(prompt_text)?;
    if !crate::text::is_blank(&extra) {
        return None;
    }
    match name.as_str() {
        "status" => Some(build_status_text(agent, conversation_len, permission, shared).await),
        "mcp" => Some(build_mcp_list_text(shared).await),
        "usage" => Some(build_usage_text(agent, shared.relay_provider_errors()).await),
        _ => None,
    }
}
/// Plain-text `/usage` output for ACP clients, reusing the shared `render::format_account_usage`.
/// A failure is worded by `acp_error_for`, the one mapping for what this wire may carry.
pub(super) async fn build_usage_text(
    agent: &crate::agent::Agent,
    relay_provider_errors: bool,
) -> String {
    match agent.fetch_usage().await {
        Ok(Some(usage)) => crate::render::format_account_usage(&usage),
        Ok(None) => "Account usage is not available for this backend.".to_string(),
        Err(error) => format!(
            "failed to fetch usage: {}",
            super::acp_error_text(&error, relay_provider_errors)
        ),
    }
}
/// Plain-text `/status` output: the block the screen prints, from [`crate::host::format_status`],
/// under a heading and after the permission level, which an ACP client may not otherwise surface.
pub(super) async fn build_status_text(
    agent: &crate::agent::Agent,
    conversation_len: usize,
    permission: &crate::permission::SharedPermission,
    shared: &crate::host::SharedDeps,
) -> String {
    format!(
        "Session status\n  Permission:      {}\n{}",
        level_display_name(permission.get()),
        crate::host::format_status(agent, &shared.providers, conversation_len).await
    )
    .trim_end()
    .to_string()
}
/// Plain-text `/mcp` output: each configured MCP server and its live connection state.
pub(super) async fn build_mcp_list_text(shared: &crate::host::SharedDeps) -> String {
    let Some(manager) = shared.mcp_manager.as_ref() else {
        return "No MCP servers.".to_string();
    };
    let names = manager.server_names();
    if names.is_empty() {
        return "No MCP servers.".to_string();
    }
    let mut out = String::from("MCP servers\n");
    for name in names {
        let status = match manager.server_entry(&name) {
            Some(entry) => match entry.state().await {
                // The name travels and the reason does not: a connector's reason has carried a
                // spawn command line, which is for the terminal `meka acp` runs in.
                crate::mcp::ServerState::Failed { .. } => {
                    "failed; the reason is in the meka log".to_string()
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
/// with an `editor_help`, followed by every installed skill, each as an [`AvailableCommand`].
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
        .filter_map(|command| {
            command
                .editor_help
                .map(|help| AvailableCommand::new(command.name, help))
        })
        .collect();
    commands.extend(
        snapshot
            .skills
            .iter()
            .filter(|skill| {
                !crate::host::COMMANDS
                    .iter()
                    .any(|command| command.editor_help.is_some() && command.name == skill.name)
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
/// Intercept a `/<skill-name> [extra]` invocation that opens an ACP prompt. Returns the one block
/// that takes the place of every part, or `None` when they stay as they are:
///
/// - Non-slash input: `None`.
/// - Slash followed by a name that isn't a syntactically valid skill identifier (e.g. a pasted path
///   like `/etc/hosts` or a `//` comment): `None`, so the model sees it.
/// - `/<skill-name>` matching an installed skill: the invocation of all that follows the name, the
///   words and the resources attached among them, so the model reads the whole request ahead of the
///   skill's body as [`crate::skills::invocation`] composes it, and a history shows the words with
///   each resource as its mention.
/// - `/<name>` with a syntactically valid skill name but no installed skill of that name:
///   `SkillNotFound`. The caller sends the original text to the model rather than failing the
///   prompt, because `/usr local lib` parses the same way.
pub(super) async fn slash_to_invocation(
    parts: &[crate::conversation::ContentBlock],
    skills: &Arc<SkillCache>,
) -> Result<Option<crate::conversation::ContentBlock>, SlashInvocationError> {
    let Some((crate::conversation::ContentBlock::Text { text: opening }, rest)) =
        parts.split_first()
    else {
        return Ok(None);
    };
    let Some((name, extra)) = split_acp_slash(opening) else {
        return Ok(None);
    };
    // Anything that doesn't even look like a skill identifier was never going to match. Pass
    // through so pasted paths and code comments reach the model unchanged.
    //
    // Narrower than the rule the delete doors apply, on purpose: those decide whether a name is
    // safe to act on, and this decides whether the user meant a skill at all. A skill whose name
    // predates the spec still resolves (`/My_Skill`), but a line like `/v1.2 of the API` stays
    // prose rather than becoming "no such skill".
    if !crate::skills::looks_like_skill_invocation(&name) {
        return Ok(None);
    }
    // A local command with words after it, which `try_local_command` left for the model.
    if crate::host::COMMANDS
        .iter()
        .any(|command| command.editor_help.is_some() && command.name == name)
    {
        return Ok(None);
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
    // The words after the name as typed, so a mention reads where it was attached among them.
    let typed = opening
        .strip_prefix('/')
        .and_then(|rest| rest.strip_prefix(name.as_str()))
        .unwrap_or(&extra)
        .trim_start();
    let (mut directive, mut words) = (typed.to_string(), typed.to_string());
    // Two text blocks side by side are two lines, as the prompt's joined words read them; one
    // beside a mention runs into it, as the editor laid them out.
    let mut after_text = true;
    for part in rest {
        match part {
            crate::conversation::ContentBlock::Text { text } => {
                if after_text {
                    directive.push('\n');
                    words.push('\n');
                }
                directive.push_str(text);
                words.push_str(text);
                after_text = true;
            }
            crate::conversation::ContentBlock::Resource { name, text, .. } => {
                directive.push_str(&format!("@{name}"));
                words.push_str(&format!("\n{text}\n"));
                after_text = false;
            }
            _ => {}
        }
    }
    // The words are cut at their ends here alone: a file attached first or last leaves the line
    // breaks that set its contents apart at the edge of the words.
    Ok(Some(crate::skills::invocation_with_words(
        &name,
        directive.trim().to_string(),
        words.trim().to_string(),
        body,
    )))
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
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    // Accept `text` + `resource_link` (the ACP baseline) + embedded `resource` and `image`. Whether
    // this session may carry an image is asked below, once its profile has been applied; other
    // content variants are refused here. The words and the resources keep the editor's order,
    // since words around a mention read by where it is.
    let mut parts: Vec<crate::conversation::ContentBlock> = Vec::new();
    let mut images: Vec<crate::image::ImageSource> = Vec::new();
    for block in &request.prompt {
        match block {
            ContentBlock::Text(text) => parts.push(crate::conversation::ContentBlock::Text {
                text: text.text.clone(),
            }),
            // `ResourceLink` is part of the ACP baseline that every agent MUST support (alongside
            // `Text`). meka doesn't fetch the resource server-side; the model sees the reference as
            // a structured tag so it can decide what to do with it.
            ContentBlock::ResourceLink(link) => parts.push(resource_link_block(link)),
            // `Resource` carries an @-mention's inlined contents (the `embedded_context`
            // capability). meka surfaces it to the model as a `<resource>` tag rather than fetching
            // anything server-side, mirroring `ResourceLink`.
            ContentBlock::Resource(embedded) => parts.push(embedded_resource_block(embedded)),
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
    // Blank text ahead of everything else is nothing typed, for every reader below alike: a local
    // command and a skill are recognized by what the prompt opens with.
    let typed_from = parts
        .iter()
        .position(|part| {
            !matches!(part, crate::conversation::ContentBlock::Text { text } if crate::text::is_blank(text))
        })
        .unwrap_or(parts.len());
    parts.drain(..typed_from);

    // Ahead of the session being taken, so a prompt with nothing in it never holds the runtime
    // mutex or publishes a cancel token: a client that pipelines a real prompt behind it must find
    // the session free, not busy. A prompt of nothing but whitespace costs a provider round-trip to
    // produce nothing, and a client that dropped its content on the floor should hear about it.
    // The words are expanded below once a slash command has been resolved.
    let prompt_text = sent_text(&parts);
    let input = match crate::agent::TurnInput::sent(parts.clone(), images) {
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
    let entry = match state
        .resident_or_reopened(&cx, &session_id_str, session_uuid)
        .await
    {
        Some(Ok(entry)) => entry,
        Some(Err(error)) => return responder.respond_with_error(error),
        None => {
            return responder.respond_with_error(acp_error_for(
                &MekaError::SessionNotFound(session_uuid),
                false,
            ));
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

    // Under the lock and before the first round, so a switch made while the previous turn held
    // this mutex takes effect on this one. Refused rather than run on the old profile: the client
    // asked for a specific account, and answering from another one silently is the failure the
    // whole per-session binding exists to prevent.
    if let Err(error) =
        crate::host::apply_recorded_bindings(&state.shared, &entry.agent, entry.id).await
    {
        return responder.respond_with_error(build_failure_error(
            "cannot run this turn on the profile and project this session is recorded against",
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
             `[profiles.<name>]`",
        ));
    }

    // Install a fresh cancellation token inside the locked scope so the cancel handler (which reads
    // the sibling cell) always sees the token for the turn currently using the runtime. The guard
    // lives to the end of this function, which is the end of the turn.
    let cancellation = crate::agent::TurnCancellation::new();
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
        // `shell_execute` output is rendered.
        let body = format!("```\n{output}\n```");
        send_session_update(
            &frontend.connection,
            &frontend.session_id,
            SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                agent_client_protocol::schema::v1::TextContent::new(body),
            ))),
        );
        // A local command spends nothing, and says so: `usage` is the turn's own.
        let none = entry.agent.session_stats_snapshot();
        return responder
            .respond(PromptResponse::new(StopReason::EndTurn).usage(turn_usage(&none, &none)));
    }

    // An invocation opens the prompt or is not one, and takes the place of everything sent with it.
    match slash_to_invocation(&parts, &state.shared.skills).await {
        Ok(Some(invocation)) => parts = vec![invocation],
        Ok(None) => {}
        Err(SlashInvocationError::SkillNotFound(name)) => {
            // `slash_to_invocation` only returns `SkillNotFound` for strings whose first token is
            // a syntactically-valid skill name. That's deliberately a narrow filter, but it still
            // false-positives on pasted text like `/usr local lib` (parses as name=`usr`,
            // extra=`local lib`). Treat "no such skill" as "this wasn't a skill invocation after
            // all" and feed the original text to the model. It can respond with "I don't know that
            // command" if the user really meant `/<name>`. The alternative (hard-error) breaks
            // paste UX for any string starting with `/word`.
            tracing::debug!(
                "session/prompt: '/{name}' didn't match a registered skill; passing through"
            );
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
    let input = input.expanded(parts);
    // An outcome that did not warrant a turn of its own rides on this one, ahead of the editor's
    // prompt rather than as a message of its own: see `background::render_outcomes_riding`.
    let outcomes = if state.shared.config.background.enabled {
        crate::host::claim_undelivered_outcomes(agent, &state.shared.store, *session_uuid).await
    } else {
        Vec::new()
    };
    let input = input.riding(outcomes);
    // Clone the cancellation token so we can probe `is_canceled()` after the call returns. The
    // spec mandates that any cancel arriving during a turn must surface as `StopReason::Cancelled`,
    // even when the cancellation manifests as a provider / tool error rather than the clean
    // `MekaError::Interrupted` path.
    let cancel_probe = cancellation.clone();
    let before = agent.session_stats_snapshot();
    let result = agent.run_turn(messages, input, cancellation).await;
    // On both sides of the turn, as every other turn path touches: the idle sweep measures from
    // the last touch, and a long turn touched at its start alone would read as idle at its end.
    entry.touch();

    let stop_reason = match result {
        Ok(crate::agent::TurnOutcome::EndTurn) => StopReason::EndTurn,
        Ok(crate::agent::TurnOutcome::MaxTokens) => StopReason::MaxTokens,
        Ok(crate::agent::TurnOutcome::Refusal(_)) => StopReason::Refusal,
        Err(MekaError::Interrupted) => StopReason::Cancelled,
        Err(error) => {
            if cancel_probe.is_canceled() {
                StopReason::Cancelled
            } else {
                // Through the one classifier, which decides what of a failed turn an editor may
                // read; a `Display` formatted here would carry an upstream body naming the
                // operator's account, an MCP connector's spawn command line, or a `Database`
                // error naming the store's path.
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
        messages.title(),
    );

    let after = agent.session_stats_snapshot();
    responder.respond(PromptResponse::new(stop_reason).usage(turn_usage(&before, &after)))
}
/// What one turn spent, for `session/prompt`'s response, which the protocol defines as the turn's
/// usage: the session's counters after it less before, so a sub-agent's spend and a compaction's
/// inside the turn count toward it. The `usage_update` notification carries the context gauge
/// instead.
pub(super) fn turn_usage(
    before: &crate::stats::SessionStatsSnapshot,
    after: &crate::stats::SessionStatsSnapshot,
) -> Usage {
    let spent = |after: u64, before: u64| after.saturating_sub(before);
    let input = spent(after.input_tokens, before.input_tokens);
    let output = spent(after.output_tokens, before.output_tokens);
    let cache_read = spent(
        after.cache_read_input_tokens,
        before.cache_read_input_tokens,
    );
    let cache_write = spent(
        after.cache_creation_input_tokens,
        before.cache_creation_input_tokens,
    );
    let mut usage = Usage::new(
        input
            .saturating_add(cache_read)
            .saturating_add(cache_write)
            .saturating_add(output),
        input,
        output,
    );
    usage.cached_read_tokens = Some(cache_read);
    usage.cached_write_tokens = Some(cache_write);
    // `thought_tokens` is left at its `None` default: meka doesn't meter reasoning separately from
    // output, and reporting a made-up split would be worse than reporting none.
    usage
}
