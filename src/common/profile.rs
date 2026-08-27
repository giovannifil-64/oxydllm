//! Env-gated per-phase forward profiler (`OXYDLLM_PROFILE=1`).
//!
//! Times the major phases of a forward with a device sync after each, so a
//! phase is charged for the GPU work it queued rather than for queueing it.
//! Prefill and decode are accumulated apart: they run the same code over
//! wildly different shapes, and averaging them together hides which one is
//! slow. Reports every 64 forwards of a regime, so one request is enough to
//! see a breakdown. Zero overhead when the variable is unset.

use candle_core::{Device, Result};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Forwards of a regime between reports. The totals are cumulative, so this
/// only decides how often the running breakdown is printed.
const REPORT_EVERY: u64 = 16;

/// How many forwards of one regime a report waits for, overridable while
/// hunting: a prefill that now runs in two chunks never reaches sixteen.
fn report_every() -> u64 {
    std::env::var("OXYD_PROFILE_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &u64| n > 0)
        .unwrap_or(REPORT_EVERY)
}

fn enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("OXYDLLM_PROFILE").as_deref() == Ok("1"))
}

/// Which regime a forward belongs to; they are reported separately.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Regime {
    Prefill,
    Decode,
}

impl Regime {
    fn label(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

thread_local! {
    static CURRENT: Cell<Option<Regime>> = const { Cell::new(None) };
}

/// Per regime: forwards seen, tokens seen, and per-phase (calls, nanoseconds).
#[derive(Default)]
struct Totals {
    forwards: u64,
    tokens: u64,
    phases: BTreeMap<&'static str, (u64, u128)>,
}

static ACC: Mutex<BTreeMap<Regime, Totals>> = Mutex::new(BTreeMap::new());

/// Opens a forward of `tokens` tokens, decode when every sequence contributes
/// exactly one.
pub fn begin_forward(is_decode: bool, tokens: usize) {
    if !enabled() {
        return;
    }
    let regime = if is_decode {
        Regime::Decode
    } else {
        Regime::Prefill
    };
    CURRENT.with(|c| c.set(Some(regime)));
    let mut acc = ACC.lock().unwrap();
    let t = acc.entry(regime).or_default();
    t.forwards += 1;
    t.tokens += tokens as u64;
}

fn current() -> Option<Regime> {
    if !enabled() {
        return None;
    }
    CURRENT.with(|c| c.get())
}

/// Sync at a phase boundary so the next phase's timing starts from an idle
/// device instead of absorbing the previous one's queued tail.
pub fn barrier(device: &Device) {
    if current().is_some() {
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

/// Times `f` including the GPU work it queued and charges it to `name`.
pub fn phase<T>(device: &Device, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let Some(regime) = current() else {
        return f();
    };
    let t0 = Instant::now();
    let r = f()?;
    sync(device);
    let ns = t0.elapsed().as_nanos();
    let mut acc = ACC.lock().unwrap();
    let e = acc.entry(regime).or_default();
    let p = e.phases.entry(name).or_insert((0, 0));
    p.0 += 1;
    p.1 += ns;
    Ok(r)
}

/// Closes the forward, reporting every [] of the regime.
pub fn mark_forward_end() {
    let Some(regime) = current() else {
        return;
    };
    CURRENT.with(|c| c.set(None));
    let due = {
        let acc = ACC.lock().unwrap();
        acc.get(&regime)
            .is_some_and(|t| t.forwards.is_multiple_of(report_every()))
    };
    if due {
        report(regime);
    }
}

fn report(regime: Regime) {
    let acc = ACC.lock().unwrap();
    let Some(t) = acc.get(&regime) else {
        return;
    };
    let total_ns: u128 = t.phases.values().map(|(_, ns)| *ns).sum();
    let tokens = t.tokens.max(1) as f64;
    eprintln!(
        "=== {} profile: {} forwards, {} tokens, {:.0} tok/s timed (sync-serialized) ===",
        regime.label(),
        t.forwards,
        t.tokens,
        tokens / (total_ns as f64 / 1e9).max(f64::MIN_POSITIVE),
    );
    let mut items: Vec<_> = t.phases.iter().collect();
    items.sort_by_key(|item| std::cmp::Reverse(item.1.1));
    for (name, (calls, ns)) in items {
        let pct = if total_ns > 0 {
            *ns as f64 / total_ns as f64 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {:14} {:6.2}%  {:9.4} ms/token  ({calls} calls)",
            name,
            pct,
            *ns as f64 / 1e6 / tokens,
        );
    }
}
