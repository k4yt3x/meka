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
    // (`[Permission context]`, `[Todo list]`, `[Scratchpad entries]`).
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

    let discovery = render_tool_discovery(&catalog);
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
            current.memory_tools,
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
/// Rendered fresh every turn from live state, like `[Todo list]`, rather than living in
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

/// Byte ceiling on the priority-0 bodies rendered in full, separate from
/// [`MEMORY_INDEX_MAX_BYTES`].
///
/// Separate so a long standing directive cannot eat the index, and the index cannot eat the
/// directives. They answer different questions ("what do I always have to do" against "what else
/// do I know"), and one budget would let whichever renders first starve the other.
pub(super) const MEMORY_INLINE_MAX_BYTES: usize = 4_096;

/// Per-entry ceiling on an inlined body, in *characters*, so one runaway memory cannot consume the
/// whole allowance and leave the rest of the band as bare descriptions.
///
/// Characters rather than bytes because it bounds what the model reads, and because the total
/// above is already a byte bound: whatever this lets through, [`MEMORY_INLINE_MAX_BYTES`] still
/// stops the block growing without limit.
pub(super) const MEMORY_INLINE_ENTRY_MAX_CHARS: usize = 1_024;

/// How many distinct tags the histogram names before it stops. Enough to steer a search; this is a
/// signpost, not a census.
pub(super) const MEMORY_TAG_HISTOGRAM_MAX: usize = 6;

/// Render the `[Memory]` index: the entries that fit, then a count of those that did not.
///
/// `memories` arrives pre-sorted by [`crate::store::memory::MemoryStore::index`] (priority
/// ascending, newest first within a band), so the budget simply takes a prefix and everything
/// dropped is genuinely the least important.
///
/// The trailing "N more" line is not optional. Silently truncating an index reads to the model as
/// "this is everything I know", which turns a full store into a confidently incomplete answer;
/// stating the remainder is what makes `memory_search` the obvious next move.
pub(super) fn render_memory_section(memories: &[MemoryIndexEntry], tools: MemoryTools) -> String {
    let now = std::time::SystemTime::now();
    // `memory_read` is unconditional because the section itself is gated on it. The rest is not:
    // naming a disabled tool is an instruction the model cannot follow, which is exactly what the
    // gate one level up exists to prevent.
    let mut out = String::from(
        "[Memory]\nDurable notes available to this session, most important first; an instruction \
         given now is newer than any of them. Call `memory_read` with a name to read one.",
    );
    if tools.write {
        out.push_str(
            " Use `memory_write` for what later sessions need: preferences, constraints, \
             decisions and facts. The scratchpad holds a session's temporary task state and large \
             outputs, and later sessions do not read it.",
        );
    }
    out.push_str("\n\n");
    let (standing, inlined) = render_standing_memories(memories, now);

    // Whatever the standing band rendered in full is not repeated as a description line below.
    // Listing it twice wastes the budget and reads as a duplicate, which a model treats as
    // evidence the entry was planted.
    let listable: Vec<&MemoryIndexEntry> = memories
        .iter()
        .filter(|entry| !inlined.contains(entry.name.as_str()))
        .collect();

    // Built into its own buffer so the standing band's overflow notice can be written *between* the
    // band and the index, after both budgets have been spent. The notice cannot be computed any
    // earlier: it is a claim about what the index below contains, and the index does not know until
    // it has laid itself out.
    //
    // Measured against the index's own bytes, not `out.len()`. Charging the standing band to this
    // budget is what [`MEMORY_INLINE_MAX_BYTES`] says it does not do (four ordinary directives
    // would cost the index 40% of its entries), and the separation is only real if the two are
    // counted separately.
    let mut index = String::new();
    let mut index_bytes = 0;
    let mut shown = 0;
    let mut standing_listed = 0;
    for entry in listable.iter().take(MEMORY_INDEX_MAX_ENTRIES) {
        let line = format!(
            "- **{}** (p{}, {}): {}\n",
            entry.name,
            entry.priority,
            crate::memory::render_age(entry.created, now),
            crate::entry::elide_description_for_index(&entry.description)
        );
        // Always emit at least one entry: a single pathological description longer than the whole
        // budget should still be visible rather than collapsing the section to a bare count.
        if shown > 0 && index_bytes + line.len() > MEMORY_INDEX_MAX_BYTES {
            break;
        }
        index_bytes += line.len();
        index.push_str(&line);
        shown += 1;
        if entry.inline_body.is_some() {
            standing_listed += 1;
        }
    }

    if !standing.is_empty() {
        let overflow = listable
            .iter()
            .filter(|entry| entry.inline_body.is_some())
            .count();
        out.push_str(&standing);
        out.push_str(&render_standing_overflow(overflow, standing_listed, tools));
        out.push('\n');
    }
    out.push_str(&index);

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
            render_tag_histogram(&listable[shown..]),
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
    out
}

/// The priority-0 band, rendered with its bodies in full, or an empty string when there is none.
///
/// For a standing directive the body *is* the directive, and a one-line description with the text
/// behind a `memory_read` is a rule the model has to choose to look up before it can follow it.
/// This is the always-in-context tier the priority band was already trying to be.
pub(super) fn render_standing_memories(
    memories: &[MemoryIndexEntry],
    now: std::time::SystemTime,
) -> (String, std::collections::HashSet<&str>) {
    let mut inlined: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let standing: Vec<&MemoryIndexEntry> = memories
        .iter()
        .filter(|entry| entry.inline_body.is_some())
        .collect();
    if standing.is_empty() {
        return (String::new(), inlined);
    }

    // The header states the contract, because without it the band is ambiguous: a model shown a
    // body cannot tell "this is the whole note" from "this is a preview", and hedges that the full
    // stored body "may contain more", which is the kind of provisionality a standing rule must not
    // acquire. `clip_chars` already marks a real truncation with an ellipsis; saying so is what
    // turns that mark into a signal the reader can act on.
    let mut out = String::from(
        "Standing preferences and constraints, subject to current instructions. Bodies are complete unless marked \u{2026}; read a truncated entry with `memory_read` before relying on it.\n\n",
    );
    for entry in &standing {
        let Some(body) = &entry.inline_body else {
            continue;
        };
        // Elided like every other index line. Descriptions are deliberately not bounded at parse
        // time (see `crate::entry::elide_description_for_index`), so an unbounded one here (and
        // the first block is emitted whatever its size) lets a single memory blow the band's
        // whole allowance on its description alone.
        let mut block = format!(
            "- **{}** ({}): {}\n",
            entry.name,
            crate::memory::render_age(entry.created, now),
            crate::entry::elide_description_for_index(&entry.description)
        );
        // Deliberately *not* `elide`, which collapses whitespace: it exists for one-line index
        // entries, and a standing directive is very often a short list of rules. Flattening
        // "Answer in kind.\nNever apologize." into one run-on line is a legibility loss in exactly
        // the case this band was built for, and it happens even when the body is well inside the
        // budget.
        for line in clip_chars(body, MEMORY_INLINE_ENTRY_MAX_CHARS).lines() {
            block.push_str(&format!("  {line}\n"));
        }
        if !inlined.is_empty() && out.len() + block.len() > MEMORY_INLINE_MAX_BYTES {
            break;
        }
        out.push_str(&block);
        inlined.insert(entry.name.as_str());
    }

    // What became of the overflow is stated by [`render_standing_overflow`], which the caller
    // appends once the index below has laid itself out. This block deliberately does not say,
    // because from here the answer is a guess.
    (out, inlined)
}

/// What became of the priority-0 memories the inline band could not fit.
///
/// Separate from [`render_standing_memories`] because only the caller knows the answer. The band
/// does not state its own overflow as "N further priority-0 memories are listed by description
/// below", tempting as that is on the reasoning that a standing memory the inline budget dropped
/// still falls through to the index like everything else. That holds for a small store and fails
/// for a large one: the index rations [`MEMORY_INDEX_MAX_BYTES`] across the whole store, so the
/// overflow competes with it, and past a few dozen standing memories some of them lose and reach
/// the model nowhere at all.
///
/// The count being wrong is the smaller half. Priority 0 is the tier whose contract is "these
/// always apply", so one that appears in no part of the context is a rule the model is being held
/// to and cannot read, and a confident sentence saying otherwise removes the one clue that it
/// should go looking.
pub(super) fn render_standing_overflow(
    overflow: usize,
    listed: usize,
    tools: MemoryTools,
) -> String {
    if overflow == 0 {
        return String::new();
    }
    if listed >= overflow {
        return format!(
            "\n{overflow} further priority-0 {} listed by description below rather than in full; \
             read {} with `memory_read`.\n",
            if overflow == 1 {
                "memory is"
            } else {
                "memories are"
            },
            if overflow == 1 { "it" } else { "them" }
        );
    }
    format!(
        "\n{overflow} further priority-0 memories are not shown in full: {listed} listed by \
         description below, {} left out entirely because this index is full. Their standing guidance still matters{}\n",
        overflow - listed,
        if tools.search {
            "; reach the ones left out with `memory_search`."
        } else {
            ", and nothing in this context names the ones left out."
        }
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
    if let Some(line) = tool_change_line(
        "- New deferred tools (use `tool_load` for schemas): ",
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

    // Memories move whenever the agent writes one, which is often, so the diff carries the delta
    // rather than re-listing the index. Priority is included because it decides where the entry
    // will sit when the index is next stated in full.
    let previous_memories: std::collections::HashMap<&str, &MemoryIndexEntry> = previous
        .memories
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect();
    // Name alongside line, so an entry the budget cuts can still be named rather than counted.
    let changed_memories: Vec<(String, String)> = current
        .memories
        .iter()
        .filter(|entry| {
            // Compare only what the model was told, not `created`. Rewriting a memory with
            // identical content is noise to re-announce; the timestamp still rides in the
            // snapshot, because it decides ordering the next time the index renders in full.
            //
            // `inline_body` and `tags` are in the comparison because both are things the model was
            // told: a priority-0 body is rendered in full, so editing one changes what is in
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
                })
        })
        // The line carries what *changed*, not just that something did. A full `[Memory]` render
        // only happens when there is no previous snapshot, so for the rest of a session this delta
        // is the only channel: naming a rewritten priority-0 memory without restating its body
        // leaves the superseded directive as the only rule text in the window, and the model goes
        // on following it. Tags are stated for the weaker version of the same reason: otherwise
        // a tags-only edit emits a line byte-identical to the index entry already in context,
        // which is a change announcement carrying no change.
        .map(|entry| {
            // Elided like every other rendered description: descriptions are deliberately
            // unbounded at the write door, and the diff is the only channel after the first turn,
            // so one long description here would outweigh the 8 KB budget the rest of the section
            // is engineered around.
            let mut line = format!(
                "{} (p{}: {})",
                entry.name,
                entry.priority,
                crate::entry::elide_description_for_index(&entry.description)
            );
            if !entry.tags.is_empty() {
                line.push_str(&format!(" [{}]", entry.tags.join(", ")));
            }
            if let Some(body) = &entry.inline_body {
                line.push_str(&format!(
                    "\n  {}",
                    clip_chars(body, MEMORY_INLINE_ENTRY_MAX_CHARS).replace('\n', "\n  ")
                ));
            }
            (entry.name.clone(), line)
        })
        .collect();
    let removed_memories: Vec<&String> = previous
        .memories
        .iter()
        .filter(|entry| {
            !current
                .memories
                .iter()
                .any(|candidate| candidate.name == entry.name)
        })
        .map(|entry| &entry.name)
        .collect();
    if !changed_memories.is_empty() {
        // Budgeted like every other memory render. The entry count is unbounded (a compaction
        // checkpoint writes several standing memories at once) and each may carry a whole
        // priority-0 body, so without a ceiling this one line can outweigh the 8 KB the index
        // itself is held to.
        let mut shown = 0;
        let mut bytes = 0;
        for (_, entry) in &changed_memories {
            if shown > 0 && bytes + entry.len() > MEMORY_INLINE_MAX_BYTES {
                break;
            }
            bytes += entry.len();
            shown += 1;
        }
        let rendered = changed_memories[..shown]
            .iter()
            .map(|(_, line)| line.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        // The cut ones are named, not counted and waved at the index: that is the last full render
        // and therefore predates the very writes being announced, so eight priority-0 directives
        // written in one turn would render three and tell the model the other five were somewhere
        // they were not. Names are what a `memory_read` needs, and they are short; it is the
        // bodies that spent the budget.
        lines.push(format!(
            "- Memories saved or updated: {}{}",
            rendered,
            if shown < changed_memories.len() {
                let cut: Vec<&String> = changed_memories[shown..]
                    .iter()
                    .map(|(name, _)| name)
                    .collect();
                format!(
                    "; and {}, not restated here; `memory_read` them",
                    name_some_of(&cut)
                )
            } else {
                String::new()
            }
        ));
    }
    if !removed_memories.is_empty() {
        // Bounded like the line above it: without a ceiling, 501 deletions between two turns
        // render 501 names with nothing elided.
        lines.push(format!(
            "- Memories deleted: {}",
            name_some_of(&removed_memories)
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
            gate_changes[..shown].join("; ")
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
    let shown = names.len().min(MEMORY_NAMES_MAX);
    let named = join_names(names[..shown].iter().copied());
    match names.len() - shown {
        0 => named,
        hidden => format!("{named}, and {hidden} more"),
    }
}
