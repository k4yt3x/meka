//! The installation as `config.toml` and the store hold it, edited: the accounts and their
//! credentials, the profiles, and the MCP servers. One set of doors for every surface that writes
//! them, `meka account`, `meka profile` and `meka mcp` in a shell and `/v1/accounts`,
//! `/v1/profiles` and `/v1/mcp` on a running `meka serve`, so the file keeps its comments and
//! its order whoever edits it, the same refusals apply in the same words, and a secret goes to
//! the database and never to the file. What a surface adds is its own: a prompt, a printed
//! table, a request body.

pub(crate) mod account;
pub(crate) mod mcp;
pub(crate) mod profile;

/// A setting on an edit: left as it is, removed so the documented default applies, or set. On the
/// wire a key left out keeps, `null` removes and a value sets; a string key reads a blank value as
/// a removal too, since a form that clears a field and one that sends `null` ask for the same
/// thing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum Patch<T> {
    #[default]
    Keep,
    Remove,
    Set(T),
}

/// Which of the three a [`Patch`] is, for a caller that names keys without reading values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PatchState {
    Keep,
    Remove,
    Set,
}

impl<T> Patch<T> {
    /// Which of the three this is.
    pub(crate) fn state(&self) -> PatchState {
        match self {
            Self::Keep => PatchState::Keep,
            Self::Remove => PatchState::Remove,
            Self::Set(_) => PatchState::Set,
        }
    }
}

impl Patch<String> {
    /// The patch with a blank value read as a removal, the rule every string key follows.
    pub(crate) fn blank_removes(self) -> Self {
        match self {
            Self::Set(value) if value.trim().is_empty() => Self::Remove,
            other => other,
        }
    }
}

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Patch<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match Option::<T>::deserialize(deserializer)? {
            Some(value) => Self::Set(value),
            None => Self::Remove,
        })
    }
}
