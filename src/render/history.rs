//! What a stored nudge stands for on a replay.

/// The stage direction a replayed nudge stands for, or none for a message that is not one. The
/// live run announced a checklist nudge with its counts, which the message does not carry, so the
/// replay says what it can; the nudge for a visible reply was announced by nothing, and stands for
/// nothing here.
pub(crate) fn nudge_direction(message: &crate::conversation::Message) -> Option<&'static str> {
    match message.nudge() {
        Some((crate::conversation::NudgeKind::Checklist, _)) => {
            Some("checklist: items still open, continuing")
        }
        Some((crate::conversation::NudgeKind::VisibleReply, _)) | None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A replay does not put the nudge's words in the user's mouth: a checklist nudge stands for
    /// the stage direction the live run showed, and the thinking-only one for nothing, as live.
    #[test]
    fn a_replayed_nudge_is_a_stage_direction_or_nothing() {
        let checklist = crate::conversation::Message::nudge_message(
            crate::conversation::NudgeKind::Checklist,
            "[Checklist: items are still open, so this turn is not over.]",
        );
        assert_eq!(
            nudge_direction(&checklist),
            Some("checklist: items still open, continuing")
        );
        let thinking = crate::conversation::Message::nudge_message(
            crate::conversation::NudgeKind::VisibleReply,
            crate::conversation::THINKING_ONLY_NUDGE,
        );
        assert_eq!(nudge_direction(&thinking), None);
    }
}
