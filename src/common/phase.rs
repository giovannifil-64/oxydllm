//! The phase of the forward in flight, for the kernels whose choice must not
//! depend on how many tokens share the forward.
//!
//! A linear layer sees only the rows it is given, and the rows a forward
//! carries depend on scheduling: a prompt prefilled beside other prompts, a
//! prompt whose prefix the cache already holds, a token decoded beside other
//! sequences. The kernel families differ in the order they reduce, so a
//! choice made on the row count alone gave the same token different logits
//! in each of those cases. The engine states the phase instead, once per
//! forward; the forward runs on the engine's thread, and no other forward
//! shares it, which is what lets the state live in the thread rather than in
//! every call down the model.

use std::cell::Cell;

/// What the forward in flight computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Prompt tokens, from one to a chunk's worth, each a row of a GEMM.
    Prefill,
    /// One token per sequence, up to a batch of them, each a gemv row.
    Decode,
}

thread_local! {
    static PHASE: Cell<Option<Phase>> = const { Cell::new(None) };
}

/// Runs `f` with `phase` stated for the kernels it reaches, restoring what
/// was stated before.
pub fn with_phase<T>(phase: Phase, f: impl FnOnce() -> T) -> T {
    let previous = PHASE.with(|p| p.replace(Some(phase)));
    let out = f();
    PHASE.with(|p| p.set(previous));
    out
}

/// The phase of the forward in flight; `None` outside the engine, where the
/// row count alone decides.
pub fn current() -> Option<Phase> {
    PHASE.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contract: the phase is stated for the duration of the forward and no
    /// longer, and nests.
    #[test]
    fn the_phase_holds_for_the_forward_and_restores_after() {
        assert_eq!(current(), None);
        let inner = with_phase(Phase::Prefill, || {
            assert_eq!(current(), Some(Phase::Prefill));
            with_phase(Phase::Decode, current)
        });
        assert_eq!(inner, Some(Phase::Decode));
        assert_eq!(current(), None);
    }
}
