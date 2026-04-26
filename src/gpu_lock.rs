// ─────────────────────────────────────────────────────────────────────────────
// gpu_lock.rs — Per-device GPU lock for cross-model serialisation
// ─────────────────────────────────────────────────────────────────────────────
//
// When multiple models are loaded simultaneously, their independent engine
// loops would otherwise submit Metal/CUDA work concurrently, causing
// contention and degraded throughput.
//
// This module provides a per-device Mutex: models on different physical
// devices (e.g. CUDA:0 and CUDA:1) do NOT block each other, while models
// on the same device are still serialised.
//
// On single-GPU systems the behaviour is identical to the former global lock.
// On multi-GPU systems, throughput scales with the number of distinct devices.
//
// Scheduling and sampling happen **outside** the lock, so the only serialised
// portion is the actual GPU compute.
// ─────────────────────────────────────────────────────────────────────────────

use candle_core::Device;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};

/// Opaque GPU lock handle.  Clone-cheap (Arc inside).
#[derive(Clone)]
pub struct GpuLock {
    inner: Arc<Mutex<()>>,
}

impl GpuLock {
    /// Acquire exclusive access to this device.  Returns a guard that releases
    /// it on drop.  Recovers automatically if a previous holder panicked.
    pub fn acquire(&self) -> MutexGuard<'_, ()> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Canonical integer key for a candle `Device`.
///
/// - CPU                  → 0
/// - CUDA ordinal N       → (1 << 32) | N
/// - Metal (single GPU)   → 2 << 32
fn device_key(device: &Device) -> u64 {
    match device {
        Device::Cpu => 0,
        // ordinal() is only available when the cuda feature is enabled.
        #[cfg(feature = "cuda")]
        Device::Cuda(d) => (1u64 << 32) | d.ordinal() as u64,
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => 1u64 << 32,
        Device::Metal(_) => 2u64 << 32,
    }
}

/// Process-wide table: one `Mutex` per distinct device key.
static LOCKS: OnceLock<RwLock<HashMap<u64, Arc<Mutex<()>>>>> = OnceLock::new();

fn lock_table() -> &'static RwLock<HashMap<u64, Arc<Mutex<()>>>> {
    LOCKS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Return the GPU lock for the given device (created lazily on first call).
pub fn gpu_lock_for(device: &Device) -> GpuLock {
    let key = device_key(device);

    // Fast path: lock already exists.
    {
        let map = lock_table().read().unwrap_or_else(|e| e.into_inner());
        if let Some(inner) = map.get(&key) {
            return GpuLock {
                inner: Arc::clone(inner),
            };
        }
    }

    // Slow path: insert a new entry.
    let mut map = lock_table().write().unwrap_or_else(|e| e.into_inner());
    let inner = map
        .entry(key)
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone();
    GpuLock { inner }
}
