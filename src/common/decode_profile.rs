//! Env-gated per-phase decode profiler (`OXYDLLM_PROFILE_DECODE=1`).
//!
//! Times the major decode phases with a Metal sync after each. Decode is
//! serialized through the residual stream (op N+1 needs op N's output), so the
//! sync reflects real per-token cost rather than pipelined peak throughput.
//! Accumulates over decode forwards only (M=1) and auto-reports every 64 so a
//! single generation request yields a steady-state breakdown. Zero overhead
//! when the env var is unset.

use candle_core::{Device, Result};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("OXYDLLM_PROFILE_DECODE").as_deref() == Ok("1"))
}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

static ACC: Mutex<BTreeMap<&'static str, (u64, u128)>> = Mutex::new(BTreeMap::new());
static FORWARDS: AtomicU64 = AtomicU64::new(0);

/// Mark whether the current forward should be profiled (decode-only, M=1).
pub fn set_active(is_decode: bool) {
    if enabled() {
        ACTIVE.with(|c| c.set(is_decode));
    }
}

fn active() -> bool {
    enabled() && ACTIVE.with(|c| c.get())
}

/// Sync at a phase boundary so the next phase's timing starts from an idle GPU
/// (otherwise the first phase absorbs the previous forward's unsynced tail).
pub fn barrier(device: &Device) {
    if active() {
        sync(device);
    }
}

fn sync(device: &Device) {
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = device {
        let _ = dev.wait_until_completed();
    }
    #[cfg(not(feature = "metal"))]
    let _ = device;
}

/// Time `f` (a decode phase) including its GPU execution and attribute the
/// elapsed wall-clock to `name`. No-op unless profiling an active decode forward.
pub fn phase<T>(device: &Device, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !active() {
        return f();
    }
    let t0 = Instant::now();
    let r = f()?;
    sync(device);
    let ns = t0.elapsed().as_nanos();
    let mut acc = ACC.lock().unwrap();
    let e = acc.entry(name).or_insert((0, 0));
    e.0 += 1;
    e.1 += ns;
    Ok(r)
}

/// Call once at the end of each forward. Reports the cumulative breakdown every
/// 64 decode forwards.
pub fn mark_forward_end() {
    if !active() {
        return;
    }
    let n = FORWARDS.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(64) {
        report();
    }
}

fn report() {
    let acc = ACC.lock().unwrap();
    let total_ns: u128 = acc.values().map(|(_, ns)| *ns).sum();
    let fwds = FORWARDS.load(Ordering::Relaxed).max(1) as f64;
    eprintln!(
        "=== decode profile: {} forwards, {:.3} ms/token timed (sync-serialized) ===",
        FORWARDS.load(Ordering::Relaxed),
        total_ns as f64 / 1e6 / fwds
    );
    let mut items: Vec<_> = acc.iter().collect();
    items.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (name, (cnt, ns)) in items {
        let pct = if total_ns > 0 {
            *ns as f64 / total_ns as f64 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {:14} {:6.2}%  {:7.3} ms/token  ({} calls total)",
            name,
            pct,
            *ns as f64 / 1e6 / fwds,
            cnt
        );
    }
}
