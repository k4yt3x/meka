//! The world state as the model sees it: the full rendering of a snapshot on a first turn or after
//! a compaction, and the diff against what the model was last shown otherwise, over the bounded
//! skill, memory, tool, schedule and background indexes.

use super::*;

/// Render `current` for the per-turn `<context>` block, relative to what the model was last told.
///
/// - `previous == Some(same)` → `""`. The steady-state path, and the one that matters: an unchanged
///   session must add nothing per turn, or this would cost more than the cache it protects.
/// - `previous == None` → the full picture. Used on the first turn of a session and again after a
///   compaction, which rewrites the head of the conversation and can summarize the earlier
///   rendering away.
/// - otherwise → only what changed, carrying explicit replacement wording so the model treats the
///   new text as superseding the old rather than adding to it.
pub(crate) fn render_world_state(
    current: &WorldSnapshot,
    previous: Option<&WorldSnapshot>,
) -> String {
    let Some(previous) = previous else {
        return render_world_state_full(current);
    };
    if previous == current {
        return String::new();
    }
    render_world_state_diff(current, previous)
}

/// The whole picture, for a session's first turn and for the turn after a compaction.
pub(super) fn render_world_state_full(current: &WorldSnapshot) -> String {
    let catalog: Vec<ToolCatalogEntry> = current
        .tools
        .iter()
        .map(|(name, (required, deferred, summary))| {
            (name.clone(), summary.clone(), *required, *deferred)
        })
        .collect();
    let active: Vec<&ToolCatalogEntry> = catalog.iter().filter(|(_, _, _, d)| !d).collect();

    // `[Section]` headings rather than markdown ones, matching the rest of the `<context>` block
    // (`[Permission context]`, `[Background]`, `[Scratchpad entries]`).
    let mut sections: Vec<String> = Vec::new();

    if !active.is_empty() {
        let mut out = String::from(
            "[Available tools]\nSchemas are in the API tool catalog. The level beside each name \
             is its permission classification; execution also depends on approvals and \
             confinement. See [Permission context].\n\n",
        );
        for (name, _summary, required, _) in &active {
            out.push_str(&format!("- **{name}** (level `{required}`)\n"));
        }
        sections.push(out);
    }

    let discovery = render_tool_discovery(&catalog, current.loads_by_search);
    if !discovery.is_empty() {
        sections.push(discovery);
    }

    // Skips alone are enough to render the section, for the reason `[Memory]` gives just below.
    if !current.skills.is_empty() || !current.skipped_skills.is_empty() {
        sections.push(render_skill_section(
            "Skills",
            &current.skills,
            &current.skipped_skills,
            current.skill_tools,
        ));
    }

    if !current.memories.is_empty() {
        sections.push(render_memory_section(
            &current.memories,
            current.inline_overflow.as_ref(),
            current.memory_tools,
            current.project.as_deref(),
        ));
    }

    if !current.scheduled.is_empty() {
        sections.push(render_schedule_section(&current.scheduled));
    }

    if !current.mcp_instructions.is_empty() {
        let mut out =
            String::from("[MCP server instructions]\nEach server's own guidance for its tools.\n");
        for (server, body) in &current.mcp_instructions {
            out.push_str(&format!("\n{server}\n{body}\n"));
        }
        sections.push(out);
    }

    sections.join("\n")
}

/// Ceiling on how many jobs the `[Scheduled]` index lists. Low on purpose: this renders on turns
/// that have nothing to do with scheduling, and a handful is enough to stop the model
/// double-booking a reminder. `schedule_list` is there for the rest.
pub(super) const SCHEDULE_INDEX_MAX_ENTRIES: usize = 20;

/// Ceiling on how many jobs the world-state *diff* names when their status changes at once.
///
/// Lower than the index above, because this is one line rather than a section and the whole set
/// flips together: dropping a session to `none` withholds every job it has, each contributing the
/// same sentence. Past this the count carries the fact.
pub(super) const SCHEDULE_STATUS_MAX_ENTRIES: usize = 5;

/// The `[Background]` section: what is still running, and nothing else.
///
/// Rendered fresh every turn from live state rather than living in
/// [`WorldSnapshot`]. The snapshot is a record of what the model has been *told*, diffed so an
/// unchanged picture costs nothing, and it carries an invariant that every difference must produce
/// something to read. Running tasks fit neither half of that: they churn, a departure is already
/// reported by its own outcome turn, and announcing departures here as well would say the same
/// thing twice. Always-current is also simply more useful, since the model can read what is running
/// instead of reconstructing it from arrival notices.
///
/// Deliberately carries no results. An outcome is permanent and belongs in the conversation.
pub(super) fn render_background_section(
    tasks: &[crate::store::background::BackgroundTask],
) -> String {
    let mut out = String::from("[Background]\nTasks still running.\n\n");
    for task in tasks.iter().take(BACKGROUND_INDEX_MAX_ENTRIES) {
        out.push_str(&format!(
            "- **{}**: {}\n",
            task.short_id(),
            elide(&task.label, crate::background::LABEL_MAX_CHARS)
        ));
    }
    let hidden = tasks.len().saturating_sub(BACKGROUND_INDEX_MAX_ENTRIES);
    if hidden > 0 {
        out.push_str(&format!(
            "\n{hidden} more not shown here; use `task_list` to see them.\n"
        ));
    }
    out
}

/// Ceiling on tasks listed in `[Background]`. Well above `[background] max_tasks`'s default, so in
/// practice every running task is shown and this only guards a raised ceiling.
pub(super) const BACKGROUND_INDEX_MAX_ENTRIES: usize = 20;

/// Render the `[Scheduled]` index.
///
/// Deliberately omits next-fire times. They move every time a job fires, and [`WorldSnapshot`] is
/// diffed by equality, so including them would re-render the whole section on most turns of any
/// session with a short interval, paying tokens on every turn to tell the model something it
/// almost never needs. What it does need is that a job exists, so it does not schedule a second
/// copy of one the user already asked for.
pub(super) fn render_schedule_section(jobs: &[ScheduledIndexEntry]) -> String {
    let mut out = String::from(
        "[Scheduled]\nJobs scheduled in this session; `schedule_list` for exact fire times, gates \
         and prompts.\n\n",
    );
    for entry in jobs.iter().take(SCHEDULE_INDEX_MAX_ENTRIES) {
        out.push_str(&format!(
            "- **{}** ({}): {}\n",
            entry.short_id, entry.schedule, entry.summary
        ));
        // A held job is otherwise indistinguishable from a healthy one that has nothing to report,
        // which is the normal resting state of a watcher. Without this the model could cancel a
        // job it had no way of knowing was dead.
        if let Some(reason) = &entry.withheld {
            out.push_str(&format!("  NOT FIRING: {reason}\n"));
        }
    }
    let hidden = jobs.len().saturating_sub(SCHEDULE_INDEX_MAX_ENTRIES);
    if hidden > 0 {
        out.push_str(&format!(
            "\n{hidden} more not shown here; use `schedule_list` to see them.\n"
        ));
    }
    out
}

/// Byte and entry ceilings on the rendered `[Skills]` index. Same values and same reasoning as the
/// `[Memory]` pair below: the content is the same shape (a name and a one-line description), so it
/// gets the same budget.
pub(super) const SKILL_INDEX_MAX_BYTES: usize = 8_192;
pub(super) const SKILL_INDEX_MAX_ENTRIES: usize = 200;

/// Byte and entry ceilings on the rendered `[Tool discovery]` index: the skills and memory pair,
/// for the same reason, an entry being a name and one line. Unlike those two this index has a
/// middle tier: past the byte budget every entry keeps its name and loses its summary, since a
/// name under its server's heading says most of what the summary says at a tenth of the cost, and
/// only past both ceilings does the section start counting what it cannot name.
pub(super) const TOOL_INDEX_MAX_BYTES: usize = 8_192;
pub(super) const TOOL_INDEX_MAX_ENTRIES: usize = 200;

/// Render the `[Skills]` index: the entries that fit, then a count of those that did not.
///
/// `skills` arrives sorted by `(priority, name)` from [`crate::skills`], so the budget takes a
/// prefix and what falls off is genuinely the least important rather than whatever sorted late
/// alphabetically.
///
/// The priority itself is deliberately not rendered; see the field docs on
/// [`crate::skills::Skill::priority`].
pub(super) fn render_skill_section(
    heading: &str,
    skills: &[(String, String)],
    skipped: &[crate::skills::SkippedSkill],
    tools: SkillTools,
) -> String {
    // The usual header promises an index of things to call. With nothing loadable there is no
    // index, and the reader has to be told that before it reads a list of files it cannot open.
    let mut out = format!("[{heading}]\n");
    out.push_str(if skills.is_empty() {
        "No skill is currently loadable.\n"
    } else {
        "`skill_read` loads one by name.\n\n"
    });

    let mut shown = 0;
    for (name, description) in skills.iter().take(SKILL_INDEX_MAX_ENTRIES) {
        let line = format!(
            "- **{}**: {}\n",
            name,
            crate::entry::elide_description_for_index(description)
        );
        // Always emit at least one entry, for the same reason `[Memory]` does: one pathological
        // description longer than the whole budget should still be visible rather than collapsing
        // the section to a bare count.
        if shown > 0 && out.len() + line.len() > SKILL_INDEX_MAX_BYTES {
            break;
        }
        out.push_str(&line);
        shown += 1;
    }

    let hidden = skills.len().saturating_sub(shown);
    if hidden > 0 {
        // The remedy clause only when the model has the tool, exactly as `[Memory]` does. Saying
        // the rest exists is still worth it without one (a silently truncated index reads as
        // "this is everything"), but naming a tool that is not there is not a remedy.
        out.push_str(&format!(
            "\n{} more skill{} not shown here{}\n",
            hidden,
            if hidden == 1 { "" } else { "s" },
            if tools.search {
                format!(
                    "; use `skill_search` to find {} by content.",
                    if hidden == 1 { "it" } else { "them" }
                )
            } else {
                ".".to_string()
            }
        ));
    }
    out.push_str(&render_unreadable_skills(skipped));
    out
}

/// The paragraph naming skill directories that could not be loaded, or an empty string.
///
/// A skill absent from this index is one the model has no reason to ask for, so making `skill_read`
/// honest about a name it is never given closed only half the hole. Somebody drops a procedure into
/// the store, its frontmatter has a typo, and from inside the session that is indistinguishable
/// from a procedure nobody wrote.
///
/// Memory needs no such paragraph: a memory is a database row, so there is no parse to fail and no
/// file to be unreadable. Skills stay on files because a `SKILL.md` is a shared spec other clients
/// read.
pub(super) fn render_unreadable_skills(skipped: &[crate::skills::SkippedSkill]) -> String {
    if skipped.is_empty() {
        return String::new();
    }
    // "could not be loaded" rather than "not in the index above", because there may be no index:
    // when nothing loads, the header above says so instead of listing anything.
    let mut out = format!(
        "\n{} director{} in your skills path could not be loaded, so {} unavailable and cannot be \
         invoked:\n\n",
        skipped.len(),
        if skipped.len() == 1 { "y" } else { "ies" },
        if skipped.len() == 1 {
            "it is"
        } else {
            "they are"
        },
    );
    for entry in skipped.iter().take(SKIP_MAX_ENTRIES) {
        out.push_str(&format!(
            "- **{}**: {}\n",
            elide(&entry.name, SKIP_NAME_MAX_CHARS),
            elide(&entry.reason, SKIP_REASON_MAX_CHARS)
        ));
    }
    let hidden = skipped.len().saturating_sub(SKIP_MAX_ENTRIES);
    if hidden > 0 {
        out.push_str(&format!("\n{hidden} further unloadable director(ies).\n"));
    }
    out.push_str(
        "\nSay so rather than improvising a replacement: whoever wrote these cannot tell them \
         apart from skills you have read, and is likely relying on them.\n",
    );
    out
}

/// Byte ceiling on the rendered `[Memory]` index. Tuning constant rather than config: it trades
/// per-turn tokens against how much of the store the model can see without searching, and neither
/// end of that trade is a user preference worth a config key.
pub(super) const MEMORY_INDEX_MAX_BYTES: usize = 8_192;

/// Ceiling on how many memories the index lists, independent of byte size. Bounds the line count
/// for a store full of terse descriptions, where the byte budget alone would let the section run
/// to hundreds of lines.
pub(super) const MEMORY_INDEX_MAX_ENTRIES: usize = 200;

/// How many distinct tags the histogram names before it stops. Enough to steer a search; this is a
/// signpost, not a census.
pub(super) const MEMORY_TAG_HISTOGRAM_MAX: usize = 6;

/// The `[Memory]` section: the header, the standing band, the index and the footers, over the
/// memories the session sees.
///
/// With a project, the band and the index are grouped by tier, the global tier first and the
/// project's after it, under one budget and one ranking: which entries the index shows is decided
/// over the whole view, as the store admitted them, and the groups are how the shown ones are
/// laid out. A tier with nothing to show keeps its heading, since the header names it. The inline
/// memories the budget demoted are listed once, after both groups.
pub(super) fn render_memory_section(
    memories: &[MemoryIndexEntry],
    overflow: Option<&InlineOverflow>,
    tools: MemoryTools,
    project: Option<&str>,
) -> String {
    let now = std::time::SystemTime::now();
    // `memory_read` is unconditional because the section itself is gated on it. The rest is not:
    // naming a disabled tool is an instruction the model cannot follow, which is exactly what the
    // gate one level up exists to prevent.
    let mut out = String::from("[Memory]\n");
    match project {
        Some(project) => out.push_str(&format!(
            "Durable notes for project '{project}' and global notes, most important first; an \
             instruction given now is newer than any of them. Call `memory_read` with a name to \
             read one."
        )),
        None => out.push_str(
            "Durable notes available to this session, most important first; an instruction \
             given now is newer than any of them. Call `memory_read` with a name to read one.",
        ),
    }
    if tools.write {
        out.push_str(
            " Use `memory_write` for what later sessions need: preferences, constraints, \
             decisions and facts. The scratchpad holds a session's temporary task state and large \
             outputs, and later sessions do not read it.",
        );
    }
    out.push_str("\n\n");
    let demoted: &[String] = overflow.map_or(&[], |overflow| overflow.demoted.as_slice());
    let standing: Vec<&MemoryIndexEntry> = memories
        .iter()
        .filter(|entry| entry.inline_body.is_some() && !entry.expired)
        .collect();
    let inlined: std::collections::HashSet<&str> =
        standing.iter().map(|entry| entry.name.as_str()).collect();

    // Whatever the standing band renders in full is not repeated as a description line below.
    // Listing it twice wastes the budget and reads as a duplicate, which a model treats as
    // evidence the entry was planted. A search-only or an expired memory is listed nowhere and
    // counted in the footer instead.
    let listable: Vec<&MemoryIndexEntry> = memories
        .iter()
        .filter(|entry| {
            entry.load != crate::memory::Load::Search
                && !entry.expired
                && !inlined.contains(entry.name.as_str())
        })
        .collect();

    // Measured against the index's own bytes, not `out.len()`. The standing band has a budget of
    // its own (`[memory] inline_max_bytes`), and charging it here as well would let four ordinary
    // directives cost the index 40% of its entries; the separation is only real if the two are
    // counted separately.
    let lines: Vec<String> = listable
        .iter()
        .take(MEMORY_INDEX_MAX_ENTRIES)
        .map(|entry| index_line(entry, now))
        .collect();
    let mut index_bytes = 0;
    let mut shown = 0;
    for line in &lines {
        // Always emit at least one entry: a single pathological description longer than the whole
        // budget should still be visible rather than collapsing the section to a bare count.
        if shown > 0 && index_bytes + line.len() > MEMORY_INDEX_MAX_BYTES {
            break;
        }
        index_bytes += line.len();
        shown += 1;
    }

    match project {
        None => {
            let band = render_standing_memories(&standing, overflow, tools, now);
            if !band.is_empty() {
                out.push_str(&band);
                out.push('\n');
            }
            for line in lines.iter().take(shown) {
                out.push_str(line);
            }
        }
        Some(project) => {
            let whole: Vec<&MemoryIndexEntry> = standing
                .iter()
                .copied()
                .filter(|entry| !demoted.contains(&entry.name))
                .collect();
            if !whole.is_empty() {
                out.push_str(STANDING_BAND_CONTRACT);
            }
            let project_heading = format!("Project '{project}'");
            let groups = [("Global", ""), (project_heading.as_str(), project)];
            for (position, (heading, tier)) in groups.iter().enumerate() {
                if position > 0 {
                    out.push('\n');
                }
                out.push_str(heading);
                out.push('\n');
                let blocks: String = whole
                    .iter()
                    .filter(|entry| entry.project == *tier)
                    .map(|entry| standing_block(entry, now))
                    .collect();
                let index: String = listable
                    .iter()
                    .zip(&lines)
                    .take(shown)
                    .filter(|(entry, _)| entry.project == *tier)
                    .map(|(_, line)| line.as_str())
                    .collect();
                out.push_str(&blocks);
                if !blocks.is_empty() && !index.is_empty() {
                    out.push('\n');
                }
                out.push_str(&index);
            }
            let by_description: Vec<&MemoryIndexEntry> = standing
                .iter()
                .copied()
                .filter(|entry| demoted.contains(&entry.name))
                .collect();
            if let Some(overflow) = overflow
                && !by_description.is_empty()
            {
                out.push('\n');
                out.push_str(&render_demoted_list(&by_description, overflow, tools, now));
            }
        }
    }

    let hidden = listable.len().saturating_sub(shown);
    if hidden > 0 {
        // A bare count is not a usable signal once it runs to thousands: it says something is
        // missing without saying what, so the model cannot turn it into a query. The tag
        // distribution can be, which is most of what tags are for. The remedy clause only when the
        // model has the tool. Without `memory_search` the honest statement is that the rest exists
        // and this index cannot reach it, which is still worth saying (a silently truncated index
        // reads as "this is everything I know"), but pointing at a tool that is not there is not a
        // remedy.
        out.push_str(&format!(
            "\n{} more {} not shown here{}{}\n",
            hidden,
            if hidden == 1 { "memory" } else { "memories" },
            render_tag_histogram(listable.get(shown..).unwrap_or_default()),
            if tools.search {
                format!(
                    "; use `memory_search` to find {}.",
                    if hidden == 1 { "it" } else { "them" }
                )
            } else {
                ".".to_string()
            },
        ));
    }
    // Stated apart from the overflow above, which is a budget the store outgrew: these are
    // choices and expiries, and an inline rule that has expired has stopped applying without
    // anything else in the context saying so.
    let search_only = memories
        .iter()
        .filter(|entry| entry.load == crate::memory::Load::Search && !entry.expired)
        .count();
    let expired = memories.iter().filter(|entry| entry.expired).count();
    out.push_str(&render_unlisted(search_only, expired, tools));
    out
}

/// One index line: name, priority, age, deadline and description.
fn index_line(entry: &MemoryIndexEntry, now: std::time::SystemTime) -> String {
    format!(
        "- **{}** (p{}, {}{}): {}\n",
        entry.name,
        entry.priority,
        crate::memory::render_age(entry.created, now),
        match entry.expires {
            Some(expires) => format!(", {}", crate::memory::render_expiry(expires, now)),
            None => String::new(),
        },
        crate::entry::elide_description_for_index(&entry.description)
    )
}

/// The sentence the standing band opens with. It states the contract, because without it the
/// band is ambiguous: a model shown a body cannot tell "this is the whole note" from "this is a
/// preview", and hedges that the full stored body "may contain more", which is the kind of
/// provisionality a standing rule must not acquire.
const STANDING_BAND_CONTRACT: &str = "Standing preferences and constraints, subject to current instructions. Bodies are complete.\n\n";

/// "3 search-only and 2 expired memories are not listed; `memory_search` reaches them.", or
/// nothing when there are none.
fn render_unlisted(search_only: usize, expired: usize, tools: MemoryTools) -> String {
    let total = search_only + expired;
    if total == 0 {
        return String::new();
    }
    let kinds = match (search_only, expired) {
        (0, expired) => format!("{expired} expired"),
        (search_only, 0) => format!("{search_only} search-only"),
        (search_only, expired) => format!("{search_only} search-only and {expired} expired"),
    };
    format!(
        "\n{kinds} {} not listed{}\n",
        if total == 1 {
            "memory is"
        } else {
            "memories are"
        },
        if tools.search {
            format!(
                "; `memory_search` reaches {}.",
                if total == 1 { "it" } else { "them" }
            )
        } else {
            ".".to_string()
        }
    )
}

/// The inline band, rendered with its bodies whole, or an empty string when there is none.
///
/// For a standing directive the body *is* the directive, and a one-line description with the text
/// behind a `memory_read` is a rule the model has to choose to look up before it can follow it.
/// An expired one is left out: its rule has stopped applying, and the footer says so.
///
/// A body is shown whole or not at all. The memories `overflow` demoted are listed here by
/// description, under a sentence that says why, rather than left to the index below: an index line
/// does not say which load a memory has, so "listed below" would name nothing the model could
/// find, and a body cut partway would read as the whole rule.
pub(super) fn render_standing_memories(
    standing: &[&MemoryIndexEntry],
    overflow: Option<&InlineOverflow>,
    tools: MemoryTools,
    now: std::time::SystemTime,
) -> String {
    if standing.is_empty() {
        return String::new();
    }
    let demoted: &[String] = overflow.map_or(&[], |overflow| overflow.demoted.as_slice());
    let mut out = String::from(STANDING_BAND_CONTRACT);
    let mut by_description: Vec<&MemoryIndexEntry> = Vec::new();
    for entry in standing {
        if demoted.contains(&entry.name) {
            by_description.push(entry);
            continue;
        }
        out.push_str(&standing_block(entry, now));
    }
    if let Some(overflow) = overflow
        && !by_description.is_empty()
    {
        out.push('\n');
        out.push_str(&render_demoted_list(&by_description, overflow, tools, now));
    }
    out
}

/// One inline memory shown whole: its bullet, then its body indented under it.
fn standing_block(entry: &MemoryIndexEntry, now: std::time::SystemTime) -> String {
    let mut block = standing_line(entry, now);
    // Deliberately *not* `elide`, which collapses whitespace: it exists for one-line index
    // entries, and a standing directive is very often a short list of rules. Flattening
    // "Answer in kind.\nNever apologize." into one run-on line is a legibility loss in exactly
    // the case this band was built for.
    for line in entry.inline_body.as_deref().unwrap_or_default().lines() {
        block.push_str(&format!("  {line}\n"));
    }
    block
}

/// The inline memories the budget demoted, by description under a sentence that says why.
fn render_demoted_list(
    by_description: &[&MemoryIndexEntry],
    overflow: &InlineOverflow,
    tools: MemoryTools,
    now: std::time::SystemTime,
) -> String {
    let mut out = format!(
        "{}, so these are shown by description only{}:\n",
        inline_overflow_lead(overflow),
        inline_overflow_remedy(tools)
    );
    // Bounded like the index, by the same two ceilings: this list is a listing by
    // description, and a store written past the doors can hold thousands of inline rows.
    let mut listed = 0;
    let mut bytes = 0;
    for entry in by_description.iter().take(MEMORY_INDEX_MAX_ENTRIES) {
        let line = standing_line(entry, now);
        if listed > 0 && bytes + line.len() > MEMORY_INDEX_MAX_BYTES {
            break;
        }
        bytes += line.len();
        listed += 1;
        out.push_str(&line);
    }
    // The same tail as the index's own footer, for the same reason: a bare count says
    // something is missing without saying what, and the names are listed nowhere else.
    let unlisted = by_description.len().saturating_sub(listed);
    if unlisted > 0 {
        out.push_str(&format!(
            "and {unlisted} more inline {} not listed here{}{}\n",
            if unlisted == 1 { "memory" } else { "memories" },
            render_tag_histogram(by_description.get(listed..).unwrap_or_default()),
            if tools.search {
                format!(
                    "; use `memory_search` to find {}.",
                    if unlisted == 1 { "it" } else { "them" }
                )
            } else {
                ".".to_string()
            }
        ));
    }
    out
}

/// One inline memory's bullet: name, age, deadline and description, the body's heading.
///
/// The description is elided like every other index line. Descriptions are deliberately not
/// bounded at parse time (see `crate::entry::elide_description_for_index`), so an unbounded one
/// here would let a single memory spend the band on its description alone.
fn standing_line(entry: &MemoryIndexEntry, now: std::time::SystemTime) -> String {
    format!(
        "- **{}** ({}{}): {}\n",
        entry.name,
        crate::memory::render_age(entry.created, now),
        match entry.expires {
            Some(expires) => format!(", {}", crate::memory::render_expiry(expires, now)),
            None => String::new(),
        },
        crate::entry::elide_description_for_index(&entry.description)
    )
}

/// "Inline memories total 41.0 KiB against the 32.0 KiB budget": the fact the band and the diff
/// both open with.
fn inline_overflow_lead(overflow: &InlineOverflow) -> String {
    format!(
        "Inline memories total {} against the {} budget",
        crate::text::format_size(overflow.total_bytes),
        crate::text::format_size(overflow.max_bytes)
    )
}

/// What to do about the memories shown by description only: read one before acting on it, since
/// a rule the model is held to and has not read is the failure the band exists to prevent, and
/// the cleanup, only when the model has the tool to carry it out; without `memory_write` the
/// cleanup is the operator's, and naming a tool the model lacks is an instruction it cannot follow.
/// `memory_read` needs no such guard, because the section itself is gated on it.
fn inline_overflow_remedy(tools: MemoryTools) -> &'static str {
    if tools.write {
        "; read one with `memory_read` before relying on it, and shorten them or set `load: index` \
         on those that are not standing rules"
    } else {
        "; read one with `memory_read` before relying on it"
    }
}

/// The diff's one-line form of the overflow, naming the demoted memories.
fn render_inline_overflow_line(overflow: &InlineOverflow, tools: MemoryTools) -> String {
    let demoted: Vec<&String> = overflow.demoted.iter().collect();
    format!(
        "- {}, so {} {} shown by description only{}.",
        inline_overflow_lead(overflow),
        name_some_of(&demoted),
        if demoted.len() == 1 { "is" } else { "are" },
        inline_overflow_remedy(tools)
    )
}

/// `, most common tags infra (820), people (611)` for the entries the budget could not list, or an
/// empty string when none of them carry a tag.
///
/// Deliberately not "mostly tagged", a claim about coverage that nothing here measures: one tagged
/// memory among 246 would render as "mostly tagged infra (1)", and the six-tag truncation makes
/// the docs' own example (820 + 611 + 405 of 4,910) a 37% minority. The adoption case is the common
/// one, because every existing store passes through "a handful are tagged" on the way to being
/// useful. Naming the tags and their counts asserts nothing the counts contradict: the model can
/// read `(1)` against 246 and draw its own conclusion.
pub(super) fn render_tag_histogram(hidden: &[&MemoryIndexEntry]) -> String {
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for entry in hidden {
        for tag in &entry.tags {
            *counts.entry(tag.as_str()).or_default() += 1;
        }
    }
    if counts.is_empty() {
        return String::new();
    }
    // By count, then name: a histogram that reordered equal counts at random would make the whole
    // section differ between turns and re-render for nothing.
    let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
    ranked.truncate(MEMORY_TAG_HISTOGRAM_MAX);

    let rendered: Vec<String> = ranked
        .iter()
        .map(|(tag, count)| format!("{tag} ({count})"))
        .collect();
    format!(", most common tags {}", rendered.join(", "))
}

/// Ceiling on how many unloadable directories the `[Skills]` section names before it starts
/// counting instead. A handful is enough to act on; the point is to say that something is wrong,
/// not to be the repair log.
pub(super) const SKIP_MAX_ENTRIES: usize = 10;

/// Per-entry cap on a skip reason. These are parser errors, which can run long.
pub(super) const SKIP_REASON_MAX_CHARS: usize = 120;

/// Per-entry cap on a skipped directory's name.
///
/// Both fields go through [`elide`], which is doing more than shortening here: it collapses
/// whitespace, and these are the one part of the section meka did not author. A directory name is
/// whatever the filesystem accepted, which on Unix is any byte but `/` and NUL, so an unelided one
/// could carry newlines straight into the block and break the one-line-per-entry shape the reader
/// and the budget both assume.
pub(super) const SKIP_NAME_MAX_CHARS: usize = 80;

/// Only what moved since the model was last told, phrased so the new text supersedes the old.
pub(super) fn render_world_state_diff(current: &WorldSnapshot, previous: &WorldSnapshot) -> String {
    let mut lines: Vec<String> = Vec::new();

    type Facts = (Permission, bool, String);
    let newly_callable: Vec<(&String, &Facts)> = current
        .tools
        .iter()
        .filter(|(name, (_, deferred, _))| {
            !deferred && !matches!(previous.tools.get(*name), Some((_, false, _)))
        })
        .collect();
    // Described rather than merely named: a deferred tool's schema is withheld, so this one-line
    // summary is all the model has to decide whether the tool is worth a `tool_load` round trip.
    // A late-connecting MCP server can announce eighty tools at once, and eighty bare names are
    // not a catalog. Newly *callable* tools need no summary here because their full schema ships
    // in the API tools array; only the permission, which is meka's own concept, has to be stated.
    let newly_deferred: Vec<(&String, &Facts)> = current
        .tools
        .iter()
        .filter(|(name, (_, deferred, _))| {
            *deferred && !matches!(previous.tools.get(*name), Some((_, true, _)))
        })
        .collect();
    let gone: Vec<&String> = previous
        .tools
        .keys()
        .filter(|name| !current.tools.contains_key(*name))
        .collect();
    // A tool whose name and callability both held still but whose facts moved underneath: an MCP
    // server reconnecting with a reworded description for the same tool. Without this bucket the
    // snapshot would advance while nothing was said, leaving the model working from a stale
    // summary for the rest of the session.
    let restated: Vec<(&String, &Facts)> = current
        .tools
        .iter()
        .filter(|(name, facts)| {
            previous
                .tools
                .get(*name)
                .is_some_and(|before| before != *facts && before.1 == facts.1)
        })
        .collect();

    // Every list under the `[Tool discovery]` ceilings, and by the same rule: described while
    // that fits, names past it, a count past the names. A reconnecting server can reword every
    // tool it has at once, so the redescribed list is cut like the new one.
    let searchable = current.tools.contains_key(TOOL_SEARCH_TOOL);
    let bare = |entries: &[(&String, &Facts)]| -> Vec<String> {
        entries
            .iter()
            .map(|(name, _)| format!("`{name}`"))
            .collect()
    };
    let named = |entries: &[(&String, &Facts)]| -> Vec<String> {
        entries
            .iter()
            .map(|(name, facts)| name_tool(name, facts))
            .collect()
    };
    let described = |entries: &[(&String, &Facts)]| -> Vec<String> {
        entries
            .iter()
            .map(|(name, facts)| describe_tool(name, facts))
            .collect()
    };
    let callable = bare(&newly_callable);
    if let Some(line) = tool_change_line(
        "- Schemas now available: ",
        &callable,
        &callable,
        searchable,
    ) {
        lines.push(line);
    }
    let new_deferred_lead = format!(
        "- New deferred tools (use `{}` for schemas): ",
        load_door(current.loads_by_search)
    );
    if let Some(line) = tool_change_line(
        &new_deferred_lead,
        &described(&newly_deferred),
        &named(&newly_deferred),
        searchable,
    ) {
        lines.push(line);
    }
    let departed: Vec<String> = gone.iter().map(|name| format!("`{name}`")).collect();
    if let Some(line) = tool_change_line(
        "- No longer available, do not call: ",
        &departed,
        &departed,
        searchable,
    ) {
        lines.push(line);
    }
    if let Some(line) = tool_change_line(
        "- Redescribed: ",
        &described(&restated),
        &named(&restated),
        searchable,
    ) {
        lines.push(line);
    }

    // Looked up by name rather than by position: the list is priority-ordered, so re-prioritizing
    // one skill shifts every skill after it, and a positional comparison would announce the whole
    // store as changed when only its ordering did. The rank is not in the index anyway.
    let previous_skills: std::collections::HashMap<&str, &str> = previous
        .skills
        .iter()
        .map(|(name, description)| (name.as_str(), description.as_str()))
        .collect();
    let added_skills: Vec<(String, String)> = current
        .skills
        .iter()
        .filter(|(name, description)| {
            previous_skills.get(name.as_str()) != Some(&description.as_str())
        })
        .cloned()
        .collect();
    let removed_skills: Vec<&String> = previous
        .skills
        .iter()
        .filter(|(name, _)| {
            !current
                .skills
                .iter()
                .any(|(candidate, _)| candidate == name)
        })
        .map(|(name, _)| name)
        .collect();
    if !added_skills.is_empty() {
        // The index renderer, so a bulk update is bounded like the initial listing; its trailing
        // newline is dropped so the lines join like the others.
        lines.push(
            render_skill_section(
                "Skills added or updated",
                &added_skills,
                &[],
                current.skill_tools,
            )
            .trim_end()
            .to_string(),
        );
    }
    if !removed_skills.is_empty() {
        lines.push(format!(
            "- Skills no longer available: {}",
            name_some_of(&removed_skills)
        ));
    }
    // Announced in both directions, for the reason the memory equivalent gives: the snapshot
    // advances whether or not anything is said, so a transition that rendered nothing would record
    // the model as having been told about a file it never heard of.
    if current.skipped_skills != previous.skipped_skills {
        if current.skipped_skills.is_empty() {
            lines.push("- Every skill loads again; none are unreadable any more.".to_string());
        } else {
            let named = current
                .skipped_skills
                .iter()
                .take(SKIP_MAX_ENTRIES)
                .map(|entry| {
                    format!(
                        "{} ({})",
                        elide(&entry.name, SKIP_NAME_MAX_CHARS),
                        elide(&entry.reason, SKIP_REASON_MAX_CHARS)
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            let hidden = current
                .skipped_skills
                .len()
                .saturating_sub(SKIP_MAX_ENTRIES);
            let remainder = if hidden > 0 {
                format!(", and {hidden} more")
            } else {
                String::new()
            };
            lines.push(format!(
                "- Skills that cannot be loaded, so they cannot be invoked: {named}{remainder}"
            ));
        }
    }

    // A move between projects changes which tier the session sees and writes into, so the whole
    // section is restated under a line naming the project rather than announced memory by
    // memory: the model reads a new view, not a batch of edits to the old one. The tools, skills
    // and jobs below diff as on any turn.
    let mut restated_memories: Option<String> = None;
    if current.project != previous.project {
        lines.push(match &current.project {
            Some(project) => format!("- Project is now '{project}'."),
            None => "- No project.".to_string(),
        });
        if !current.memories.is_empty() {
            restated_memories = Some(render_memory_section(
                &current.memories,
                current.inline_overflow.as_ref(),
                current.memory_tools,
                current.project.as_deref(),
            ));
        } else if !previous.memories.is_empty() {
            // Said outright, since the names of the old view stand in the model's context until
            // it is told they are gone, and the per-memory delta below is not drawn across a move.
            lines.push("- No memories are in view.".to_string());
        }
    }
    // Memories move whenever the agent writes one, which is often, so the diff carries the delta
    // rather than re-listing the index. Priority is included because it decides where the entry
    // will sit when the index is next stated in full. A global memory's line says so when the
    // session is on a project, where the tier a change landed in is a fact the model acts on.
    // Across a move the section is restated above, so no memory-by-memory line is drawn.
    let same_project = current.project == previous.project;
    let mark_global = current.project.is_some();
    // Backticked here, as `join_names` does, with the tier outside the quoting: the name is what
    // a `memory_read` takes, and the marker is a fact about it.
    let tiered_name = |entry: &MemoryIndexEntry| {
        if mark_global && entry.project.is_empty() {
            format!("`{}` (global)", entry.name)
        } else {
            format!("`{}`", entry.name)
        }
    };
    let previous_memories: std::collections::HashMap<&str, &MemoryIndexEntry> = previous
        .memories
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect();
    // What the band shows by description only, now and at the last render. A body is restated
    // only for a memory the band shows whole: the demoted ones are shown by description there,
    // and the diff must not tell the model more than the next full render will.
    let demoted_now: &[String] = current
        .inline_overflow
        .as_ref()
        .map_or(&[], |overflow| overflow.demoted.as_slice());
    let demoted_before: &[String] = previous
        .inline_overflow
        .as_ref()
        .map_or(&[], |overflow| overflow.demoted.as_slice());
    // Name alongside line, so an entry the budget cuts can still be named rather than counted,
    // and the bytes the budget charges the entry for: its line without the body.
    let changed_memories: Vec<(String, String, usize)> = current
        .memories
        .iter()
        .filter(|_| same_project)
        .filter(|entry| {
            // Compare only what the model was told, not `created`. Rewriting a memory with
            // identical content is noise to re-announce; the timestamp still rides in the
            // snapshot, because it decides ordering the next time the index renders in full.
            //
            // `inline_body` and `tags` are in the comparison because both are things the model was
            // told: an inline body is rendered in full, so editing one changes what is in
            // force, and the tag histogram is what stands in for everything the budget could not
            // list. A field added to `MemoryIndexEntry` and left out here makes a pair of
            // snapshots differ while rendering nothing, which
            // `world_state_diff_never_advances_silently` exists to catch.
            previous_memories
                .get(entry.name.as_str())
                .is_none_or(|before| {
                    before.priority != entry.priority
                        || before.description != entry.description
                        || before.inline_body != entry.inline_body
                        || before.tags != entry.tags
                        || before.load != entry.load
                        || before.expires != entry.expires
                        || before.expired != entry.expired
                        || before.project != entry.project
                })
        })
        // An expiry passing is announced on its own line below, as an expiry rather than an
        // update: the memory's text is unchanged, and what the model has to hear is that the
        // rule has stopped applying.
        .filter(|entry| {
            !(entry.expired
                && previous_memories
                    .get(entry.name.as_str())
                    .is_some_and(|before| !before.expired))
        })
        // The line carries what *changed*, not just that something did. A full `[Memory]` render
        // only happens when there is no previous snapshot, so for the rest of a session this delta
        // is the only channel: naming a rewritten inline memory without restating its body
        // leaves the superseded directive as the only rule text in the window, and the model goes
        // on following it. Tags are stated for the weaker version of the same reason: otherwise
        // a tags-only edit emits a line byte-identical to the index entry already in context,
        // which is a change announcement carrying no change.
        .map(|entry| {
            let (line, counted) = memory_change_line(
                entry,
                !demoted_now.contains(&entry.name),
                mark_global && entry.project.is_empty(),
            );
            (entry.name.clone(), line, counted)
        })
        .collect();
    let removed_memories: Vec<String> = previous
        .memories
        .iter()
        .filter(|_| same_project)
        .filter(|entry| {
            !current
                .memories
                .iter()
                .any(|candidate| candidate.name == entry.name)
        })
        .map(tiered_name)
        .collect();

    let expired_memories: Vec<String> = current
        .memories
        .iter()
        .filter(|_| same_project)
        .filter(|entry| {
            entry.expired
                && previous_memories
                    .get(entry.name.as_str())
                    .is_some_and(|before| !before.expired)
        })
        .map(tiered_name)
        .collect();

    if !changed_memories.is_empty() {
        lines.push(format!(
            "- Memories saved or updated: {}",
            budgeted_memory_entries(&changed_memories)
        ));
    }
    // The band's layout moves without any memory changing: an operator door can push a rule out
    // of the budget, and a cleanup can bring one back. Announced in both directions, because the
    // snapshot advances either way.
    if same_project && current.inline_overflow != previous.inline_overflow {
        lines.push(match &current.inline_overflow {
            Some(overflow) => render_inline_overflow_line(overflow, current.memory_tools),
            None => "- Every inline memory fits the budget again.".to_string(),
        });
    }
    // A body coming back into the band is restated whole: the model has only ever seen its
    // description, and the next full render is a compaction away. One the line above restated
    // already is not repeated.
    let recovered: Vec<(String, String, usize)> = current
        .memories
        .iter()
        .filter(|_| same_project)
        .filter(|entry| {
            entry.inline_bytes() > 0
                && demoted_before.contains(&entry.name)
                && !demoted_now.contains(&entry.name)
                && !changed_memories
                    .iter()
                    .any(|(name, ..)| *name == entry.name)
        })
        .map(|entry| {
            let (line, counted) =
                memory_change_line(entry, true, mark_global && entry.project.is_empty());
            (entry.name.clone(), line, counted)
        })
        .collect();
    if !recovered.is_empty() {
        lines.push(format!(
            "- Inline memories shown in full again: {}",
            budgeted_memory_entries(&recovered)
        ));
    }
    if !expired_memories.is_empty() {
        lines.push(format!(
            "- Memories expired, no longer in force: {}",
            name_some_rendered(&expired_memories)
        ));
    }
    if !removed_memories.is_empty() {
        // Bounded like the line above it: without a ceiling, 501 deletions between two turns
        // render 501 names with nothing elided.
        lines.push(format!(
            "- Memories deleted: {}",
            name_some_rendered(&removed_memories)
        ));
    }

    // Jobs the model did not create itself still have to be announced: `meka schedule cancel` and a
    // second attached client both change this behind its back, and a job it believes still exists
    // is one it will not recreate. By id, not by whole entry. A job is immutable once created
    // except for whether its gate can currently fire, so comparing the whole struct would report
    // a job that had merely gone held as newly scheduled, announcing an appearance that never
    // happened, and burying the thing that did change.
    let added_jobs: Vec<String> = current
        .scheduled
        .iter()
        .filter(|entry| {
            !previous
                .scheduled
                .iter()
                .any(|candidate| candidate.short_id == entry.short_id)
        })
        .map(|entry| format!("{} ({}): {}", entry.short_id, entry.schedule, entry.summary))
        .collect();
    let removed_jobs: Vec<&String> = previous
        .scheduled
        .iter()
        .filter(|entry| {
            !current
                .scheduled
                .iter()
                .any(|candidate| candidate.short_id == entry.short_id)
        })
        .map(|entry| &entry.short_id)
        .collect();
    // A job that is still there but has changed whether it can fire. Neither an appearance nor a
    // disappearance, and reporting it as either would be a lie; it gets its own line because it is
    // the moment the model can act on, and the only one it would otherwise have to infer.
    let gate_changes: Vec<String> = current
        .scheduled
        .iter()
        .filter_map(|entry| {
            let before = previous
                .scheduled
                .iter()
                .find(|candidate| candidate.short_id == entry.short_id)?;
            if before.withheld == entry.withheld {
                return None;
            }
            // Three transitions, not two. A job whose reason merely changed (a session
            // dropping from `read` to `none` under a shell gate swaps one refusal for another)
            // would read as "can no longer fire", asserting a transition from firing that never
            // happened and inviting the model to act on a change of state rather than a change of
            // explanation.
            Some(match (&before.withheld, &entry.withheld) {
                (None, Some(reason)) => {
                    format!("{} can no longer fire: {}", entry.short_id, reason)
                }
                (Some(_), Some(reason)) => {
                    format!(
                        "{} still cannot fire, now because {}",
                        entry.short_id, reason
                    )
                }
                (_, None) => format!("{} can fire again", entry.short_id),
            })
        })
        .collect();
    if !added_jobs.is_empty() {
        lines.push(format!("- Jobs scheduled: {}", added_jobs.join("; ")));
    }
    if !removed_jobs.is_empty() {
        lines.push(format!(
            "- Jobs no longer scheduled: {}",
            join_names(removed_jobs.into_iter())
        ));
    }
    if !gate_changes.is_empty() {
        // Budgeted like every other line in this function. Lowering a session to `none` flips every
        // job at once, and the snapshot holds all of them (the 20-entry cap applies only to the
        // rendered section), so at the default `max_jobs = 50` this would be one ~7 KB line of the
        // same sentence fifty times. Past the cap the count carries the fact, which is the part the
        // model acts on.
        let shown = gate_changes.len().min(SCHEDULE_STATUS_MAX_ENTRIES);
        let hidden = gate_changes.len() - shown;
        let mut line = format!(
            "- Scheduled job status: {}",
            gate_changes.get(..shown).unwrap_or_default().join("; ")
        );
        if hidden > 0 {
            line.push_str(&format!("; and {hidden} more"));
        }
        lines.push(line);
    }

    let mut server_blocks: Vec<String> = Vec::new();
    for (server, body) in &current.mcp_instructions {
        if previous.mcp_instructions.get(server) != Some(body) {
            server_blocks.push(format!(
                "Instructions from MCP server {server} (these replace any instructions it provided \
                 earlier):\n{body}",
            ));
        }
    }
    let dropped_servers: Vec<&String> = previous
        .mcp_instructions
        .keys()
        .filter(|server| !current.mcp_instructions.contains_key(*server))
        .collect();
    if !dropped_servers.is_empty() {
        lines.push(format!(
            "- Instructions from these MCP servers no longer apply: {}",
            join_names(dropped_servers.into_iter()),
        ));
    }

    if lines.is_empty() && server_blocks.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "[Context changes]\nThese updates replace the corresponding tool, skill, memory, job, \
         or server facts stated earlier.\n",
    );
    if !lines.is_empty() {
        out.push('\n');
        out.push_str(&lines.join("\n"));
        out.push('\n');
    }
    if let Some(section) = restated_memories {
        out.push('\n');
        out.push_str(&section);
    }
    for block in server_blocks {
        out.push('\n');
        out.push_str(&block);
        out.push('\n');
    }
    out
}

/// One deferred-tool line, matching the `[Tool discovery]` shape of the full render so the model
/// reads the same format whether it arrived as an initial listing or as a later change.
pub(super) fn describe_tool(
    name: &str,
    (required, _, summary): &(Permission, bool, String),
) -> String {
    if summary.is_empty() {
        format!("`{name}` (level `{required}`)")
    } else {
        format!("`{name}` (level `{required}`): {summary}")
    }
}

/// The same line by name alone, for the tier past the byte budget.
pub(super) fn name_tool(name: &str, (required, ..): &(Permission, bool, String)) -> String {
    format!("`{name}` (level `{required}`)")
}

/// One `[Context changes]` line under the `[Tool discovery]` ceilings, or `None` for an
/// empty list. `full` and `named` are the same entries, once with whatever the list shows in full
/// and once by name alone; the label, the separators and the largest count line this list could
/// end in are reserved ahead of the entries, so the whole line fits the budget, not only its
/// entries. The count's remedy names `tool_search` only when the model has it.
pub(super) fn tool_change_line(
    label: &str,
    full: &[String],
    named: &[String],
    searchable: bool,
) -> Option<String> {
    if named.is_empty() {
        return None;
    }
    let trailer = |hidden: usize| {
        format!(
            "; and {hidden} more{}",
            if searchable {
                format!(" (use `{TOOL_SEARCH_TOOL}` to find them)")
            } else {
                String::new()
            }
        )
    };
    let reserved = label.len() + trailer(named.len()).len() + 2 * named.len();
    let (shown, hidden) = lines_within_budget(
        full,
        named,
        TOOL_INDEX_MAX_BYTES.saturating_sub(reserved),
        TOOL_INDEX_MAX_ENTRIES,
    );
    let mut line = format!("{label}{}", shown.join("; "));
    if hidden > 0 {
        line.push_str(&trailer(hidden));
    }
    Some(line)
}

pub(super) fn join_names<'a>(names: impl Iterator<Item = &'a String>) -> String {
    names
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One memory's entry in a diff line: name, priority and description, then its tags and standing,
/// and its whole body indented beneath when `with_body` and the band renders it. Also the bytes
/// the diff's budget charges the entry for, which is the line without the body.
///
/// The line carries what *changed*, not just that something did. A full `[Memory]` render only
/// happens when there is no previous snapshot, so for the rest of a session this delta is the only
/// channel: naming a rewritten inline memory without restating its body leaves the superseded
/// directive as the only rule text in the window, and the model goes on following it. Tags are
/// stated for the weaker version of the same reason: otherwise a tags-only edit emits a line
/// byte-identical to the index entry already in context, which is a change announcement carrying
/// no change.
fn memory_change_line(
    entry: &MemoryIndexEntry,
    with_body: bool,
    mark_global: bool,
) -> (String, usize) {
    // Elided like every other rendered description: descriptions are deliberately unbounded at
    // the write door, and the diff is the only channel after the first turn, so one long
    // description here would outweigh the budget the rest of the section is engineered around.
    let mut line = format!(
        "{} (p{}: {})",
        entry.name,
        entry.priority,
        crate::entry::elide_description_for_index(&entry.description)
    );
    if !entry.tags.is_empty() {
        line.push_str(&format!(" [{}]", entry.tags.join(", ")));
    }
    if mark_global {
        line.push_str(" (global)");
    }
    // Where the entry now stands: out of the index, past its date, or dated. A memory written as
    // expired is announced as such on the one line it gets.
    if entry.load == crate::memory::Load::Search {
        line.push_str(" (search-only)");
    }
    if let Some(expires) = entry.expires {
        line.push_str(&format!(
            " ({})",
            crate::memory::render_expiry(expires, std::time::SystemTime::now())
        ));
    }
    let counted = line.len();
    if with_body && let Some(body) = &entry.inline_body {
        line.push_str(&format!("\n  {}", body.replace('\n', "\n  ")));
    }
    (line, counted)
}

/// `(name, line, counted)` entries joined for one diff line: as many as the budget takes, then the
/// names of the rest.
///
/// Budgeted on the entry lines alone, against the index's own ceiling. The entry count is
/// unbounded (a restore loop through `PUT /v1/memory` moves thousands of rows between two turns),
/// so without a ceiling one line can outweigh the index it updates. The bodies ride on top
/// uncounted: the store admits no more of them than the inline budget, the band shows the rest by
/// description, and a body left out here is a rule the model goes on reading in its superseded
/// form.
///
/// The cut ones are named, not counted and waved at the index: that is the last full render and
/// therefore predates the very writes being announced, so a hundred memories written between two
/// turns would render some and tell the model the others were somewhere they were not. Names are
/// what a `memory_read` needs, and they are short.
fn budgeted_memory_entries(entries: &[(String, String, usize)]) -> String {
    let mut shown = 0;
    let mut bytes = 0;
    for (.., counted) in entries {
        if shown > 0 && bytes + counted > MEMORY_INDEX_MAX_BYTES {
            break;
        }
        bytes += counted;
        shown += 1;
    }
    let rendered = entries
        .iter()
        .take(shown)
        .map(|(_, line, _)| line.as_str())
        .collect::<Vec<_>>()
        .join("; ");
    if shown < entries.len() {
        let cut: Vec<&String> = entries.iter().skip(shown).map(|(name, ..)| name).collect();
        format!(
            "{rendered}; and {}, not restated here; `memory_read` them",
            name_some_of(&cut)
        )
    } else {
        rendered
    }
}

/// How many memory names the world-state diff spells out before it starts counting instead.
///
/// Names are short, which is why the diff names cut entries at all rather than pointing at an index
/// that predates the very writes being announced. Short is not the same as free. A restore loop
/// through `PUT /v1/memory`, a `meka memory` sweep from a second terminal, or a `shell_execute`
/// shelling out to one (all of which move rows behind a host's back) can put thousands of names on
/// one line: 5,000 memories appearing between two turns is roughly 85k tokens of names, in the
/// section whose own index render is held to 8 KB.
///
/// Forty is enough to act on and enough to recognize a bulk change for what it is. Past that the
/// count is the information.
pub(super) const MEMORY_NAMES_MAX: usize = 40;

/// Name up to [`MEMORY_NAMES_MAX`] memories, then say how many are not named.
///
/// The same shape the `[Skills]` branch above uses for unloadable skills, and for the same reason:
/// a list that silently stops reads as the whole list.
pub(super) fn name_some_of(names: &[&String]) -> String {
    let rendered: Vec<String> = names.iter().map(|name| format!("`{name}`")).collect();
    name_some_rendered(&rendered)
}

/// [`name_some_of`] over entries the caller has already rendered, quoting included.
fn name_some_rendered(rendered: &[String]) -> String {
    let shown = rendered.len().min(MEMORY_NAMES_MAX);
    let named = rendered
        .iter()
        .take(shown)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    match rendered.len() - shown {
        0 => named,
        hidden => format!("{named}, and {hidden} more"),
    }
}
