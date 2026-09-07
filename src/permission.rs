//! Permission state machine governing what tools the agent may invoke. Levels: `none` (read-only,
//! no env info), `read` (filesystem reads), `workspace` (writes confined to the workspace roots),
//! `unrestricted` (no boundary at all). The level is held in an [`AtomicU8`] so the REPL can mutate
//! it concurrently with the agent loop.
//!
//! Beside the level sits one switch, **approvals**: whether a call needing more than the level is
//! submitted to the user for approval rather than refused. The level bounds *reach* and the switch
//! decides what happens at the edge of it; neither changes the other. An approved call still runs
//! at the session's level, so an approved write at `read` lands inside the workspace roots and an
//! approved command at `read` runs in the read-only sandbox.

use std::{
    fmt,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};

use crossterm::style::Color;

use crate::error::MekaError;

/// Permission levels ordered by *reach*: `None < Read < Workspace < Unrestricted`. Each level
/// contains the ones below it, so the derived `Ord` is the containment order as well as the
/// display and cycle order.
///
/// Two operations are defined over these:
///
/// - [`Permission::allows`] is the **capability predicate**: may a tool requiring `required` be
///   dispatched at this level without asking. `Workspace` and `Unrestricted` are equal here,
///   because scope is enforced at the write door rather than by hiding tools. Keeping the tool set
///   independent of the level is what holds the API tools array byte-identical across mid-session
///   toggles, which the Claude prompt-cache prefix depends on.
/// - [`Permission::clamp_to`] is the **authority bound** for a sub-agent.
///
/// One spelling per level, [`Self::name`], is what `Display`, `FromStr` and serde all go through,
/// so a persisted level reads the way `config.toml` writes it and nothing accepts a second form.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(try_from = "String", into = "String")]
#[repr(u8)]
pub(crate) enum Permission {
    None = 0,
    Read = 1,
    Workspace = 2,
    Unrestricted = 3,
}

impl Permission {
    /// Every level, in reach order, which is also the display and cycle order.
    pub(crate) const ALL: [Permission; 4] =
        [Self::None, Self::Read, Self::Workspace, Self::Unrestricted];

    /// The one spelling of this level: what the flag, the variable, the file and a session row
    /// all take.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Read => "read",
            Self::Workspace => "workspace",
            Self::Unrestricted => "unrestricted",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    pub(crate) fn supported() -> String {
        Self::ALL
            .iter()
            .map(|level| level.name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The level after this one in cycle order, wrapping from `unrestricted` to `none`.
    pub(crate) fn cycle_next(self) -> Permission {
        match self {
            Permission::None => Permission::Read,
            Permission::Read => Permission::Workspace,
            Permission::Workspace => Permission::Unrestricted,
            Permission::Unrestricted => Permission::None,
        }
    }

    /// Single-character prompt indicator: the level name's first letter. Display only; the flag
    /// and the config file take the full name and nothing else.
    pub(crate) fn indicator(self) -> &'static str {
        match self {
            Permission::None => "n",
            Permission::Read => "r",
            Permission::Workspace => "w",
            Permission::Unrestricted => "u",
        }
    }

    /// Prompt-indicator color. Green, yellow, orange and red form one temperature ramp over the
    /// four levels.
    ///
    /// Orange is spelled as truecolor rather than `DarkYellow`, which renders as olive or brown on
    /// most themes and would be confusable with the `Yellow` immediately below it in the cycle.
    /// `Color::Rgb` is already what the prompt line uses (`crate::config::default_input_style`), so
    /// this adds no assumption that was not being made one column to the left.
    pub(crate) fn indicator_color(self) -> Color {
        match self {
            Permission::None => Color::Green,
            Permission::Read => Color::Yellow,
            Permission::Workspace => Color::Rgb {
                r: 215,
                g: 135,
                b: 0,
            },
            Permission::Unrestricted => Color::Red,
        }
    }

    /// Returns true if this permission level allows using a tool that requires `required` without
    /// asking anyone.
    pub(crate) fn allows(self, required: Permission) -> bool {
        match self {
            Permission::None => required == Permission::None,
            Permission::Read => matches!(required, Permission::None | Permission::Read),
            Permission::Workspace | Permission::Unrestricted => true,
        }
    }

    /// Whether this level's authority is contained by `parent`'s: how far writes reach.
    ///
    /// The ladder is a total order, so this is `self <= parent`; it is kept as a named predicate
    /// because every door that bounds a sub-agent asks the question in these words, and a
    /// comparison operator at each of them is what would let the answer drift if the ladder were
    /// ever not total.
    pub(crate) fn is_within(self, parent: Permission) -> bool {
        self <= parent
    }

    /// Whether this level may author a shell command that runs **unattended and unconfined**: a
    /// scheduled job's gate, which outlives the turn that created it and fires on a timer with
    /// nobody watching.
    ///
    /// `Unrestricted` alone. A scheduled gate is spawned by `run_shell_probe` as a bare `sh -c`
    /// with no `Confinement`, no sandbox and meka's full environment, so the level that authorizes
    /// it must be the one that promises no boundary.
    ///
    /// `Workspace` does not pass, tempting as it is on the reasoning that it is *safer* than the
    /// top rung. That is true of `execute_command`, which `workspace` confines, and false of a
    /// gate, which bypasses every backend. Passing it is a one-call escape: at `workspace`, a
    /// single `schedule_create` with a `gate` runs arbitrary commands outside the boundary within
    /// one poll interval, no race and no user interaction, while `execute_command` at the same
    /// level is confined and is refused outright when it cannot be. The interactive shell must not
    /// have a higher bar than the unattended one.
    ///
    /// A named predicate rather than a `matches!` repeated at each door, so the four sites cannot
    /// drift into phrasing the same rule different ways.
    pub(crate) fn allows_unattended_shell(self) -> bool {
        matches!(self, Permission::Unrestricted)
    }

    /// Whether waking the agent unattended at this level could accomplish anything.
    ///
    /// Only `none` fails, and it fails completely: nothing is executable there, so a scheduled turn
    /// reads nothing, acts on nothing, and cannot even cancel the job that woke it. Registration is
    /// permission-independent, so the model does see the job in `[Scheduled]` and is offered
    /// `schedule_cancel`; the refusal happens at dispatch, which leaves it able to describe its
    /// predicament and unable to do anything about it. An `every = "5s"` job on such a session is
    /// a turn's worth of tokens every five seconds, forever, stoppable only by an operator.
    ///
    /// Distinct from [`Self::allows_unattended_shell`], which asks what a *gate* may run. This asks
    /// whether the job is worth running at all, so it applies to ungated jobs too.
    pub(crate) fn allows_unattended_work(self) -> bool {
        !matches!(self, Permission::None)
    }

    /// The highest level contained by **both** `self` and `other`.
    ///
    /// Replaying a **recorded** grant asks this: `agent_followup` re-clamps a stored
    /// `spec.permission` against the parent's *current* level, and a later parent change must never
    /// widen what the spawn call asked for. On a total order this is the minimum; it keeps its
    /// name so the replay door reads as the question it asks.
    pub(crate) fn greatest_within_both(self, other: Permission) -> Permission {
        self.min(other)
    }

    /// The authority a sub-agent actually gets when it asks for `self` under `parent`: the request
    /// when the parent holds it, else the parent's own level.
    pub(crate) fn clamp_to(self, parent: Permission) -> Permission {
        if self.is_within(parent) { self } else { parent }
    }
}

impl fmt::Display for Permission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl FromStr for Permission {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|level| level.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a permission level. Supported: {}",
                    Self::supported()
                )
            })
    }
}

impl TryFrom<String> for Permission {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<Permission> for String {
    fn from(level: Permission) -> Self {
        level.name().to_string()
    }
}

/// Read a permission level back off a database row, given the subject to name if it cannot be read.
///
/// The store's own reader: every session row is parsed once, where it is read, into
/// `Option<Permission>`, so nothing above the store handles the column as text. Returns `None` for
/// both an absent column and an unreadable one, because every caller has the same fallback for the
/// two. What differs is that an unreadable value is *noticed*: spelling this
/// `.and_then(|value| value.parse().ok())` collapses "this session never recorded a level" into
/// "this session recorded a level meka cannot read" and resumes at the process default either way,
/// silently.
///
/// Repeats. The scheduler reads a job's session level every time that job comes due, so a row it
/// cannot read behind an `every = "1m"` watcher warns once a minute until the row is fixed. That is
/// deliberate rather than overlooked: the condition silently changes what a session runs at, and
/// the message names the session so it can be corrected.
pub(crate) fn parse_recorded_permission(
    recorded: Option<&str>,
    subject: &dyn fmt::Display,
) -> Option<Permission> {
    let raw = recorded?;
    match raw.parse() {
        Ok(permission) => Some(permission),
        Err(error) => {
            tracing::warn!(
                "{subject} records permission level '{raw}', which meka does not recognize ({error}); \
                 the row is read as recording no level"
            );
            None
        }
    }
}

/// Set of permission levels the user is allowed to switch into at runtime. Backed by a `u8`
/// bitmask indexed by [`Permission`]'s `repr(u8)` discriminant. Constructed via
/// [`EnabledPermissions::from_levels`] (or the constants); the constructor guarantees the set is
/// non-empty so [`Self::lowest`] is always well-defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EnabledPermissions {
    bits: u8,
}

impl EnabledPermissions {
    /// Every level enabled. Used by test fixtures that don't care about the runtime gate;
    /// production code constructs the set from config.
    #[cfg(test)]
    pub(crate) const ALL: Self = Self { bits: 0b1111 };
    /// `none / read / workspace / unrestricted`: every level.
    ///
    /// `workspace` sits before `unrestricted` because it is the rung a user reaching for "let the
    /// agent change things" should land on: Shift+Tab passes through it before `unrestricted`, so
    /// the confined level is the one that costs fewer keystrokes.
    pub(crate) const DEFAULT: Self = Self {
        bits: (1 << Permission::None as u8)
            | (1 << Permission::Read as u8)
            | (1 << Permission::Workspace as u8)
            | (1 << Permission::Unrestricted as u8),
    };

    /// Build an `EnabledPermissions` from any iterable of [`Permission`]s. Returns `None` if the
    /// iterator yields no items. An empty enabled set is meaningless (meka would have no level to
    /// start in), so the caller has to handle that case explicitly (typically by falling back to
    /// [`Self::DEFAULT`]).
    pub(crate) fn from_levels<I: IntoIterator<Item = Permission>>(iter: I) -> Option<Self> {
        let mut bits: u8 = 0;
        for permission in iter {
            bits |= 1 << (permission as u8);
        }
        if bits == 0 { None } else { Some(Self { bits }) }
    }

    /// Whether `permission` is in the set.
    pub(crate) fn is_enabled(self, permission: Permission) -> bool {
        self.bits & (1 << (permission as u8)) != 0
    }

    /// Iterate enabled levels in `none → read → workspace → unrestricted` order.
    pub(crate) fn iter(self) -> impl Iterator<Item = Permission> {
        Permission::ALL
            .into_iter()
            .filter(move |&level| self.is_enabled(level))
    }

    /// Lowest-discriminant enabled level. Every constructor ([`Self::from_levels`] returns `None`
    /// on empty input, [`Self::DEFAULT`] is non-empty by definition) refuses an empty set, so
    /// there is always one.
    #[allow(
        clippy::expect_used,
        reason = "the constructor refuses an empty set, so `iter().next()` is always `Some`"
    )]
    pub(crate) fn lowest(self) -> Permission {
        self.iter()
            .next()
            .expect("EnabledPermissions invariant: set is non-empty")
    }

    /// The enabled levels by name, in cycle order, for a message that lists them.
    fn names(self) -> Vec<String> {
        self.iter().map(|level| level.to_string()).collect()
    }

    /// The refusal for a level this set does not admit, rendered once for every door that sets a
    /// session's level: `/permission`, `POST` and `PATCH /v1/sessions`, ACP's `session/set_mode`
    /// and its permission config option, and [`SharedPermission::try_set`] behind them all.
    pub(crate) fn disabled_level(self, level: Permission) -> MekaError {
        MekaError::DisabledLevel {
            level: level.to_string(),
            enabled: self.names(),
        }
    }

    /// The level a session's row records, if this set still admits it.
    ///
    /// A row records what a session was *set* to, not what this installation still permits, and
    /// the two diverge the moment an operator narrows `[permissions].enabled`. Filtering here,
    /// once, is what keeps every reader of the column from granting authority the configuration
    /// withdrew: a resume on any host, the scheduler's fire door, `meka schedule show`.
    ///
    /// `None` both for a row that records nothing and for one whose level this set excludes. The
    /// caller supplies what stands in, because it differs by reader (the configured default for a
    /// host, `none` for the scheduler, which runs nothing on a level nobody set). The excluded case
    /// is warned about, naming `subject`, because it silently changes what a session runs at.
    pub(crate) fn admit_recorded(
        self,
        recorded: Option<Permission>,
        subject: &str,
    ) -> Option<Permission> {
        let level = recorded?;
        if self.is_enabled(level) {
            return Some(level);
        }
        tracing::warn!(
            "{subject} records permission level '{level}', which is not in [permissions].enabled \
             (enabled: {enabled}); the row is read as recording no level",
            enabled = self.names().join(", ")
        );
        None
    }
}

/// Lock-free shared handle to the current [`Permission`] level and the approvals switch. Cloned
/// freely across agent, REPL, and tool-dispatch tasks. The REPL mutates it when the user cycles
/// permission via `Shift+Tab` or `/permission`, or toggles `/approvals`; the dispatch loop reads it
/// once at the enforcement site so mid-turn changes can't leave a tool acting on a stale snapshot.
#[derive(Clone)]
pub(crate) struct SharedPermission {
    inner: Arc<AtomicU8>,
    enabled: EnabledPermissions,
    /// A parent's live level, for a sub-agent's handle. [`Self::get`] returns the greatest level
    /// within *both* this and `inner`, so a parent downgrade takes effect on the sub-agent's very
    /// next tool call.
    ///
    /// Without it the clamp happens only at spawn, from a snapshot of the parent's level, and
    /// nothing propagates afterwards: a user who presses Shift+Tab to `none` to stop a runaway
    /// sub-agent sees the prompt indicator change and the parent's next call denied, while the
    /// sub-agent keeps writing files and running unsandboxed commands to completion, and
    /// `permissions.md` presents cycling the parent as the way to restrict sub-agents.
    ceiling: Option<Arc<AtomicU8>>,
    /// Whether a call needing more than the level is submitted for approval rather than refused.
    ///
    /// One cell shared down the tree: a sub-agent built by [`Self::with_ceiling`] holds its
    /// parent's, because the parent's frontend is where a sub-agent's requests are answered
    /// and a sub-agent whose switch disagreed with its parent's would prompt a user who had
    /// turned prompts off.
    approvals: Arc<AtomicBool>,
}

impl SharedPermission {
    /// A root handle at `initial`, with approvals off.
    pub(crate) fn new(initial: Permission, enabled: EnabledPermissions) -> Self {
        Self {
            inner: Arc::new(AtomicU8::new(initial as u8)),
            enabled,
            ceiling: None,
            approvals: Arc::new(AtomicBool::new(false)),
        }
    }

    /// [`Self::new`] with the approvals switch already set, for the hosts that seed it from config
    /// or from a session row.
    pub(crate) fn with_approvals(self, approvals: bool) -> Self {
        self.set_approvals(approvals);
        self
    }

    /// A handle bounded from above by `parent`'s live level, for a sub-agent.
    ///
    /// The child keeps its own level (it may sit below the parent, and `agent_spawn` clamps it at
    /// creation), but can never reach further than the parent does right now. A raise is inherited
    /// too, which is the same rule read in the other direction: the child's authority is always
    /// [`Permission::greatest_within_both`] of its own grant and what the human currently permits.
    /// The approvals switch is the parent's own cell, not a copy.
    pub(crate) fn with_ceiling(
        initial: Permission,
        enabled: EnabledPermissions,
        parent: &SharedPermission,
    ) -> Self {
        Self {
            inner: Arc::new(AtomicU8::new(initial as u8)),
            enabled,
            // Share the parent's *own* cell rather than its effective value, and flatten a chain:
            // a grandchild whose parent is itself bounded takes the root's cell, and its own
            // spawn-time clamp already folded the intermediate level in. That keeps `get` a fixed
            // two loads however deep the tree goes.
            //
            // That precondition is taken once, at spawn, and does not survive the root moving
            // afterwards. `a_grandchild_cannot_escape_an_intermediate_parent` in
            // `crate::tools::subagent` states the residual window precisely under "What this does
            // not cover"; nothing exceeds the root, which is the human's own level, but the
            // direct-parent bound does not hold across a root that cycles between two spawns.
            ceiling: Some(
                parent
                    .ceiling
                    .clone()
                    .unwrap_or_else(|| parent.inner.clone()),
            ),
            approvals: Arc::clone(&parent.approvals),
        }
    }

    /// The levels this handle may be set to.
    pub(crate) fn enabled(&self) -> EnabledPermissions {
        self.enabled
    }

    /// The level in force right now, bounded by the parent's for a sub-agent.
    pub(crate) fn get(&self) -> Permission {
        let own = Self::decode(self.inner.load(Ordering::Relaxed));
        match &self.ceiling {
            Some(parent) => own.greatest_within_both(Self::decode(parent.load(Ordering::Relaxed))),
            None => own,
        }
    }

    /// Whether a call needing more than [`Self::get`] is submitted for approval rather than
    /// refused.
    pub(crate) fn approvals(&self) -> bool {
        self.approvals.load(Ordering::Relaxed)
    }

    /// Turn the approvals switch on or off.
    pub(crate) fn set_approvals(&self, approvals: bool) {
        self.approvals.store(approvals, Ordering::Relaxed);
    }

    /// Decode a stored discriminant. An unrecognized byte falls to `None`, which is the safe
    /// direction: a corrupt or future value denies rather than grants.
    fn decode(raw: u8) -> Permission {
        match raw {
            0 => Permission::None,
            1 => Permission::Read,
            2 => Permission::Workspace,
            3 => Permission::Unrestricted,
            _ => Permission::None,
        }
    }

    /// Switch to `permission`. Refuses with [`MekaError::DisabledLevel`] when the level is not in
    /// [`Self::enabled`], leaving the current level unchanged.
    pub(crate) fn try_set(&self, permission: Permission) -> Result<(), MekaError> {
        if !self.enabled.is_enabled(permission) {
            return Err(self.enabled.disabled_level(permission));
        }
        self.set_unchecked(permission);
        Ok(())
    }

    /// Low-level setter that bypasses the enabled-set check. Used by `try_set` / `cycle` and by
    /// tests that need to construct edge cases.
    pub(crate) fn set_unchecked(&self, permission: Permission) {
        self.inner.store(permission as u8, Ordering::Relaxed);
    }

    /// Advance to the next enabled level in `none → read → workspace → unrestricted → ...` order,
    /// skipping any disabled levels. If only one level is enabled the cycle is a visual no-op
    /// (returns the current level without changing it). Bounded to 4 iterations so it can never
    /// spin forever.
    pub(crate) fn cycle(&self) -> Permission {
        let mut next = self.get();
        for _ in 0..4 {
            next = next.cycle_next();
            if self.enabled.is_enabled(next) {
                self.set_unchecked(next);
                return next;
            }
        }
        // Unreachable when the constructor invariant holds (set non-empty), because the loop walks
        // through all four variants. Return current for safety instead of panicking.
        self.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every level in the canonical order, for the exhaustive matrices below.
    const EVERY: [Permission; 4] = [
        Permission::None,
        Permission::Read,
        Permission::Workspace,
        Permission::Unrestricted,
    ];

    #[test]
    fn a_level_allows_itself_and_every_level_below() {
        assert!(Permission::Unrestricted.allows(Permission::None));
        assert!(Permission::Unrestricted.allows(Permission::Read));
        assert!(Permission::Unrestricted.allows(Permission::Workspace));
        assert!(Permission::Unrestricted.allows(Permission::Unrestricted));

        assert!(Permission::Read.allows(Permission::None));
        assert!(Permission::Read.allows(Permission::Read));
        assert!(!Permission::Read.allows(Permission::Workspace));
        assert!(!Permission::Read.allows(Permission::Unrestricted));

        assert!(Permission::None.allows(Permission::None));
        assert!(!Permission::None.allows(Permission::Read));
        assert!(!Permission::None.allows(Permission::Workspace));
        assert!(!Permission::None.allows(Permission::Unrestricted));
    }

    /// `workspace` must dispatch every tool, including the ones that declare the rung above it.
    ///
    /// Scope is enforced at the write door, not by withholding the tool, and this is also what
    /// keeps the API tools array byte-identical across mid-session toggles. A `workspace` that
    /// filtered the array would silently break the Claude prompt-cache prefix on every switch.
    #[test]
    fn workspace_dispatches_every_tool_and_read_still_does_not() {
        for required in EVERY {
            assert!(
                Permission::Workspace.allows(required),
                "workspace must dispatch a tool requiring {required}"
            );
        }
        assert!(Permission::Read.allows(Permission::Read));
        assert!(!Permission::Read.allows(Permission::Workspace));
    }

    #[test]
    fn levels_order_from_none_up_to_unrestricted() {
        assert!(Permission::None < Permission::Read);
        assert!(Permission::Read < Permission::Workspace);
        assert!(Permission::Workspace < Permission::Unrestricted);
    }

    /// The sub-agent clamp never returns authority the parent does not hold, for any of the 16
    /// pairs, and containment is the ladder's own order.
    #[test]
    fn clamp_to_never_exceeds_the_parent_for_any_pair() {
        for child in EVERY {
            for parent in EVERY {
                assert_eq!(
                    child.is_within(parent),
                    child <= parent,
                    "{child}.is_within({parent}) disagrees with the ladder"
                );
                let granted = child.clamp_to(parent);
                assert!(
                    granted.is_within(parent),
                    "{child} under {parent} yielded {granted}, which is not within {parent}"
                );
                assert!(
                    granted.is_within(child),
                    "{child} under {parent} yielded {granted}, wider than the request"
                );
            }
        }
        assert_eq!(
            Permission::Unrestricted.clamp_to(Permission::Workspace),
            Permission::Workspace,
            "a child asking for the whole filesystem under a `workspace` parent gets `workspace`"
        );
        assert_eq!(
            Permission::Read.clamp_to(Permission::Unrestricted),
            Permission::Read,
            "an ordinary narrower request is granted as asked"
        );
    }

    /// A replayed grant is bounded by the spawn call as well as by the parent, and it is the
    /// greatest level under both: a sub-agent must not be crippled beyond what either bound
    /// requires.
    #[test]
    fn a_replayed_grant_is_never_wider_than_the_spawn_call_asked_for() {
        for recorded in EVERY {
            for parent in EVERY {
                let replayed = recorded.greatest_within_both(parent);
                assert!(
                    replayed.is_within(recorded),
                    "{recorded} replayed under {parent} gave {replayed}, wider than what was \
                     recorded"
                );
                assert!(
                    replayed.is_within(parent),
                    "{recorded} replayed under {parent} gave {replayed}, outside the parent"
                );
                for candidate in EVERY {
                    if candidate.is_within(recorded) && candidate.is_within(parent) {
                        assert!(
                            candidate.is_within(replayed),
                            "{candidate} is within both {recorded} and {parent}, so {replayed} is \
                             not the greatest"
                        );
                    }
                }
            }
        }
        assert_eq!(
            Permission::Workspace.greatest_within_both(Permission::Unrestricted),
            Permission::Workspace
        );
        assert_eq!(
            Permission::Read.greatest_within_both(Permission::Read),
            Permission::Read
        );
    }

    /// Absent and unreadable are different answers, and the second one says so.
    ///
    /// The warn arm is reached by any row holding a value this build does not resolve: without the
    /// warning the level silently becomes the configured default and the only later clue is a
    /// message naming a level the row was never created at. Collapsing the two into a bare `None`
    /// is a one-character edit that nothing caught.
    /// Everything `tracing` emits while the returned guard lives, for a test about a warning.
    struct CapturedLogs {
        bytes: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
        _guard: tracing::subscriber::DefaultGuard,
    }
    impl CapturedLogs {
        fn logged(&self) -> String {
            String::from_utf8_lossy(&crate::sync::lock(&self.bytes)).to_string()
        }
    }
    fn capture_logs() -> CapturedLogs {
        #[derive(Clone)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
                crate::sync::lock(&self.0).extend_from_slice(buffer);
                Ok(buffer.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
            type Writer = Self;

            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let bytes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(Capture(std::sync::Arc::clone(&bytes)))
            .with_ansi(false)
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        CapturedLogs {
            bytes,
            _guard: guard,
        }
    }

    #[test]
    fn an_unreadable_recorded_permission_warns_where_an_absent_one_is_silent() {
        let captured = capture_logs();

        assert_eq!(
            parse_recorded_permission(Some("unrestricted"), &"session x"),
            Some(Permission::Unrestricted)
        );
        assert_eq!(parse_recorded_permission(None, &"session x"), None);
        assert!(
            captured.logged().is_empty(),
            "neither a good value nor an absent one is worth a warning"
        );

        assert_eq!(parse_recorded_permission(Some("ask"), &"session x"), None);
        let logged = captured.logged();
        assert!(
            logged.contains("session x") && logged.contains("'ask'"),
            "an unreadable value is named, with its subject: {logged}"
        );
    }

    /// The live half of the rule `SubagentSpec::effective_permission` enforces at replay: a
    /// sub-agent spawned at `workspace` drops with its parent and never exceeds it.
    #[test]
    fn tightening_a_parent_bounds_a_running_worker() {
        let parent = SharedPermission::new(Permission::Unrestricted, EnabledPermissions::ALL);
        let sub_agent =
            SharedPermission::with_ceiling(Permission::Workspace, EnabledPermissions::ALL, &parent);
        assert_eq!(sub_agent.get(), Permission::Workspace, "the control");

        parent.set_unchecked(Permission::Read);
        assert_eq!(
            sub_agent.get(),
            Permission::Read,
            "a `workspace` sub-agent under a `read` parent falls to `read`"
        );
        parent.set_unchecked(Permission::Unrestricted);
        assert_eq!(
            sub_agent.get(),
            Permission::Workspace,
            "and never rises above what it was spawned with"
        );
    }

    /// The approvals switch is one cell down the tree: a sub-agent asks exactly when its parent
    /// would, because the parent's frontend is where the answer comes from.
    #[test]
    fn a_worker_shares_its_parents_approvals_switch() {
        let parent = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        let sub_agent =
            SharedPermission::with_ceiling(Permission::Read, EnabledPermissions::ALL, &parent);
        assert!(!sub_agent.approvals(), "off by default");
        parent.set_approvals(true);
        assert!(
            sub_agent.approvals(),
            "the parent's switch reaches a running sub-agent"
        );
        sub_agent.set_approvals(false);
        assert!(
            !parent.approvals(),
            "and it is the same cell in both directions"
        );
        let seeded =
            SharedPermission::new(Permission::Read, EnabledPermissions::ALL).with_approvals(true);
        assert!(seeded.approvals());
    }

    /// Only `unrestricted` may authorize a *shell* gate.
    ///
    /// A gate whose probe is a tool call is authorized by
    /// [`crate::schedule::gate_probe_is_authorized`] instead, at the tool's own resolved level;
    /// this predicate answers only for the bare `sh -c` case, which is why it is the strict one.
    /// `workspace` is out because a shell gate is spawned with no sandbox at all, so a level whose
    /// whole meaning is a write boundary cannot honestly authorize one: `schedule_create` with a
    /// `gate` would run arbitrary unconfined commands from inside the confined level.
    #[test]
    fn only_unrestricted_may_authorize_a_gate() {
        assert!(Permission::Unrestricted.allows_unattended_shell());
        assert!(
            !Permission::Workspace.allows_unattended_shell(),
            "a gate runs unsandboxed, so the confined level must not authorize one"
        );
        assert!(!Permission::Read.allows_unattended_shell());
        assert!(!Permission::None.allows_unattended_shell());
    }

    #[test]
    fn the_cycle_walks_the_ladder_and_wraps() {
        assert_eq!(Permission::None.cycle_next(), Permission::Read);
        assert_eq!(Permission::Read.cycle_next(), Permission::Workspace);
        assert_eq!(Permission::Workspace.cycle_next(), Permission::Unrestricted);
        assert_eq!(Permission::Unrestricted.cycle_next(), Permission::None);
    }

    #[test]
    fn every_level_parses_from_its_name_and_nothing_else_does() {
        assert_eq!(Permission::from_str("none"), Ok(Permission::None));
        assert_eq!(Permission::from_str("read"), Ok(Permission::Read));
        assert_eq!(Permission::from_str("workspace"), Ok(Permission::Workspace));
        assert_eq!(
            Permission::from_str("unrestricted"),
            Ok(Permission::Unrestricted)
        );
        assert!(Permission::from_str("invalid").is_err());
        assert!(
            Permission::from_str("ask").is_err(),
            "the retired level resolves to nothing; approvals are a switch, not a rung"
        );
    }

    /// One spelling per level on every surface: `name()` round-trips through `FromStr`, `Display`
    /// and serde, and a case variant is refused rather than folded, so a file, a flag and a row
    /// cannot disagree about what they wrote.
    #[test]
    fn every_level_round_trips_through_its_one_spelling_and_no_other() {
        for level in Permission::ALL {
            assert_eq!(level.name().parse::<Permission>(), Ok(level));
            assert_eq!(level.to_string(), level.name());
            let json = serde_json::to_string(&level).expect("serialize");
            assert_eq!(json, format!("\"{}\"", level.name()));
            assert_eq!(
                serde_json::from_str::<Permission>(&json).expect("deserialize"),
                level
            );
            assert!(
                level.name().to_uppercase().parse::<Permission>().is_err(),
                "{} must be the only spelling of {level}",
                level.name()
            );
            let capitalized = {
                let mut chars = level.name().chars();
                let first = chars.next().expect("non-empty").to_uppercase();
                format!("{first}{}", chars.as_str())
            };
            assert!(
                serde_json::from_str::<Permission>(&format!("\"{capitalized}\"")).is_err(),
                "serde must refuse '{capitalized}'"
            );
        }
    }

    /// The prompt's indicator is display, not a second name: one spelling per level, so the flag,
    /// the environment variable and the config file all take the full word and nothing shorter.
    #[test]
    fn an_indicator_character_is_not_a_level_name() {
        for level in EVERY {
            assert_eq!(level.indicator().chars().count(), 1);
            assert!(
                Permission::from_str(level.indicator()).is_err(),
                "indicator {} must not parse as {level}",
                level.indicator()
            );
        }
    }

    /// The refusal names every level, so a user typing a level meka does not have learns what it
    /// does have without a second lookup.
    #[test]
    fn an_unknown_level_is_refused_and_the_refusal_names_the_four() {
        let error = Permission::from_str("elevated").expect_err("not a level");
        for level in EVERY {
            assert!(
                error.contains(&level.to_string()),
                "the refusal must list {level}: {error}"
            );
        }
        assert!(
            !error.contains("ask"),
            "the retired level is not offered: {error}"
        );
    }

    #[test]
    fn every_level_displays_as_its_name() {
        assert_eq!(Permission::None.to_string(), "none");
        assert_eq!(Permission::Read.to_string(), "read");
        assert_eq!(Permission::Workspace.to_string(), "workspace");
        assert_eq!(Permission::Unrestricted.to_string(), "unrestricted");
    }

    #[test]
    fn the_default_set_enables_every_level() {
        let default = EnabledPermissions::DEFAULT;
        for level in EVERY {
            assert!(default.is_enabled(level), "{level} is enabled by default");
        }
    }

    #[test]
    fn the_all_set_enables_every_level() {
        let all = EnabledPermissions::ALL;
        for level in EVERY {
            assert!(all.is_enabled(level));
        }
    }

    #[test]
    fn enabled_permissions_from_levels() {
        let set = EnabledPermissions::from_levels([Permission::Read, Permission::Unrestricted])
            .expect("non-empty");
        assert!(set.is_enabled(Permission::Read));
        assert!(set.is_enabled(Permission::Unrestricted));
        assert!(!set.is_enabled(Permission::None));
        assert!(!set.is_enabled(Permission::Workspace));
        assert!(
            EnabledPermissions::from_levels(std::iter::empty()).is_none(),
            "an empty set is refused"
        );
    }

    #[test]
    fn enabled_permissions_iter_order() {
        let levels: Vec<Permission> = EnabledPermissions::ALL.iter().collect();
        assert_eq!(levels, EVERY.to_vec());
    }

    #[test]
    fn the_lowest_enabled_level_is_the_first_in_the_set() {
        assert_eq!(EnabledPermissions::DEFAULT.lowest(), Permission::None);
        assert_eq!(
            EnabledPermissions::from_levels([Permission::Workspace, Permission::Unrestricted])
                .unwrap()
                .lowest(),
            Permission::Workspace
        );
        assert_eq!(
            EnabledPermissions::from_levels([Permission::Unrestricted])
                .unwrap()
                .lowest(),
            Permission::Unrestricted
        );
    }

    #[test]
    fn a_shared_permission_reads_back_what_was_set() {
        let shared = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        assert_eq!(shared.get(), Permission::Read);

        shared.try_set(Permission::Unrestricted).unwrap();
        assert_eq!(shared.get(), Permission::Unrestricted);
    }

    #[test]
    fn a_clone_shares_the_level_and_the_approvals_switch() {
        let shared = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        let cloned = shared.clone();

        shared.try_set(Permission::Unrestricted).unwrap();
        assert_eq!(cloned.get(), Permission::Unrestricted);
        cloned.set_approvals(true);
        assert!(shared.approvals(), "the switch is shared with the level");
    }

    #[test]
    fn shared_permission_try_set_disabled() {
        let only_read = EnabledPermissions::from_levels([Permission::Read]).unwrap();
        let shared = SharedPermission::new(Permission::Read, only_read);
        let error = shared.try_set(Permission::Unrestricted).unwrap_err();
        assert!(
            matches!(
                &error,
                MekaError::DisabledLevel { level, enabled }
                    if level == "unrestricted" && enabled == &["read".to_string()]
            ),
            "the refusal names the level and the set that excludes it: {error}"
        );
        // Current level unchanged.
        assert_eq!(shared.get(), Permission::Read);
    }

    /// A recorded level the enabled set excludes is read as no level at all, and said so once.
    ///
    /// Every host reads a session's row through this. Before there was one definition the REPL
    /// warned, ACP logged at `debug!` and the HTTP re-attach said nothing, so the same row was
    /// loud on one surface and silent on the other two.
    #[test]
    fn a_recorded_level_the_enabled_set_excludes_is_admitted_as_none_with_one_warning() {
        let captured = capture_logs();
        let only_read = EnabledPermissions::from_levels([Permission::Read]).unwrap();

        assert_eq!(
            only_read.admit_recorded(Some(Permission::Read), "session x"),
            Some(Permission::Read)
        );
        assert_eq!(only_read.admit_recorded(None, "session x"), None);
        assert!(
            captured.logged().is_empty(),
            "neither an admitted level nor an absent one is worth a warning"
        );

        assert_eq!(
            only_read.admit_recorded(Some(Permission::Unrestricted), "session x"),
            None
        );
        let logged = captured.logged();
        assert!(
            logged.contains("WARN")
                && logged.contains("session x")
                && logged.contains("'unrestricted'")
                && logged.contains("enabled: read"),
            "an excluded level is warned about, naming the subject, the level and the set: {logged}"
        );
    }

    #[test]
    fn shared_permission_cycle_skips_disabled() {
        let without_workspace = EnabledPermissions::from_levels([
            Permission::None,
            Permission::Read,
            Permission::Unrestricted,
        ])
        .unwrap();
        let shared = SharedPermission::new(Permission::Read, without_workspace);
        assert_eq!(shared.cycle(), Permission::Unrestricted);
        assert_eq!(shared.get(), Permission::Unrestricted);
        assert_eq!(shared.cycle(), Permission::None);
        assert_eq!(shared.cycle(), Permission::Read);
    }

    /// Shift+Tab reaches the confined rung before the unbounded one.
    ///
    /// Ordering, not just membership: a user cycling toward "let the agent write" stops at
    /// `workspace` first, and has to press again to give up the boundary.
    #[test]
    fn shared_permission_cycle_all_enabled() {
        let shared = SharedPermission::new(Permission::None, EnabledPermissions::ALL);
        assert_eq!(shared.cycle(), Permission::Read);
        assert_eq!(shared.cycle(), Permission::Workspace);
        assert_eq!(shared.cycle(), Permission::Unrestricted);
        assert_eq!(shared.cycle(), Permission::None);
    }

    #[test]
    fn a_cycle_over_a_single_enabled_level_stays_put() {
        let only_read = EnabledPermissions::from_levels([Permission::Read]).unwrap();
        let shared = SharedPermission::new(Permission::Read, only_read);
        // Cycle returns the same level and doesn't loop forever.
        assert_eq!(shared.cycle(), Permission::Read);
        assert_eq!(shared.get(), Permission::Read);
    }

    #[test]
    fn shared_permission_set_unchecked_bypasses_enabled() {
        // Used by tests that need to construct edge cases regardless of the configured enabled set
        // (e.g. prompt-cache invariance tests).
        let only_read = EnabledPermissions::from_levels([Permission::Read]).unwrap();
        let shared = SharedPermission::new(Permission::Read, only_read);
        shared.set_unchecked(Permission::Unrestricted);
        assert_eq!(shared.get(), Permission::Unrestricted);
    }
}
