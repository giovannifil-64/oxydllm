// ─────────────────────────────────────────────────────────────────────────────
// gpu_lock.rs: Per-device GPU lock for cross-model serialisation
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

use candle_core::{Device, DeviceLocation};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};

#[derive(Clone)]
pub struct GpuLock {
    inner: Arc<Mutex<()>>,
}

impl GpuLock {
    pub fn acquire(&self) -> MutexGuard<'_, ()> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn device_key(device: &Device) -> u64 {
    match device.location() {
        DeviceLocation::Cpu => 0,
        DeviceLocation::Cuda { gpu_id } => (1u64 << 32) | gpu_id as u64,
        DeviceLocation::Metal { gpu_id } => (2u64 << 32) | gpu_id as u64,
    }
}

static LOCKS: OnceLock<RwLock<HashMap<u64, Arc<Mutex<()>>>>> = OnceLock::new();

fn lock_table() -> &'static RwLock<HashMap<u64, Arc<Mutex<()>>>> {
    LOCKS.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn gpu_lock_for(device: &Device) -> GpuLock {
    let key = device_key(device);

    {
        let map = lock_table().read().unwrap_or_else(|e| e.into_inner());
        if let Some(inner) = map.get(&key) {
            return GpuLock {
                inner: Arc::clone(inner),
            };
        }
    }

    let mut map = lock_table().write().unwrap_or_else(|e| e.into_inner());
    let inner = map
        .entry(key)
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone();
    GpuLock { inner }
}
