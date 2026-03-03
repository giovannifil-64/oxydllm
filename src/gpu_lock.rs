// ─────────────────────────────────────────────────────────────────────────────
// gpu_lock.rs — Global GPU lock for cross-model serialisation
// ─────────────────────────────────────────────────────────────────────────────
//
// When multiple models are loaded simultaneously, their independent 
// engine loops would otherwise submit Metal/CUDA work concurrently, 
// causing contention and degraded throughput for both.
//
// This module provides a process-wide Mutex that every engine acquires
// around its GPU forward pass.  On a single-GPU system this is optimal:
// only one model uses the GPU at any time, and switching between them
// has near-zero overhead (an uncontended std::sync::Mutex is ~25ns).
//
// Scheduling and sampling happen **outside** the lock, so the only
// serialised portion is the actual GPU compute.
// ─────────────────────────────────────────────────────────────────────────────

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Opaque GPU lock handle.  Clone-cheap (Arc inside).
#[derive(Clone)]
pub struct GpuLock {
    inner: Arc<Mutex<()>>,
}

impl GpuLock {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(())),
        }
    }

    /// Acquire exclusive GPU access.  Returns a guard that releases it on drop.
    ///
    /// If a previous holder panicked the lock is recovered automatically.
    pub fn acquire(&self) -> MutexGuard<'_, ()> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Process-wide singleton — every engine loop shares the same lock.
static INSTANCE: OnceLock<GpuLock> = OnceLock::new();

/// Return the global GPU lock (created on first call).
pub fn gpu_lock() -> GpuLock {
    INSTANCE.get_or_init(GpuLock::new).clone()
}
