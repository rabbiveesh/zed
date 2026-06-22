//! Ad-hoc, env-gated CPU / redraw rate probe for performance investigations.
//!
//! Enable by setting `ZED_CPU_PROBE=1`. While enabled, call [`record`] (or the
//! [`time`] scoped helper) from hot paths; roughly once per second the
//! accumulated per-category counts and timings are emitted via `log::info!`
//! under the `cpu_probe` target. When disabled, every entry point is a single
//! cached atomic-ish load plus an early return, so probes left in place are
//! cheap enough to keep across an investigation.
//!
//! This exists to answer a specific question for issue #57349 (agent panel CPU
//! storm): how often the window redraws, how many of those redraws are driven
//! by a continuously-animating element, and how much markdown re-shaping each
//! frame triggers. It is a throwaway investigation tool, not production
//! telemetry.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ZED_CPU_PROBE")
            .map(|value| value != "0" && !value.is_empty())
            .unwrap_or(false)
    })
}

struct CategoryStats {
    count: u64,
    total: Duration,
    max: Duration,
}

struct Probe {
    last_flush: Instant,
    stats: BTreeMap<&'static str, CategoryStats>,
}

thread_local! {
    static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) };
}

/// Record one occurrence of `category`, optionally with the time it took.
/// No-op unless `ZED_CPU_PROBE` is set.
pub fn record(category: &'static str, elapsed: Option<Duration>) {
    if !enabled() {
        return;
    }
    PROBE.with(|cell| {
        let mut probe = cell.borrow_mut();
        let probe = probe.get_or_insert_with(|| Probe {
            last_flush: Instant::now(),
            stats: BTreeMap::new(),
        });
        let entry = probe.stats.entry(category).or_insert(CategoryStats {
            count: 0,
            total: Duration::ZERO,
            max: Duration::ZERO,
        });
        entry.count += 1;
        if let Some(elapsed) = elapsed {
            entry.total += elapsed;
            entry.max = entry.max.max(elapsed);
        }

        let now = Instant::now();
        let interval = now.duration_since(probe.last_flush);
        if interval >= Duration::from_secs(1) {
            flush(probe, interval);
            probe.last_flush = now;
        }
    });
}

/// Count one occurrence of `category` without timing it.
pub fn count(category: &'static str) {
    record(category, None);
}

/// Returns a guard that records the elapsed time into `category` when dropped.
/// Returns `None` (and does nothing) unless `ZED_CPU_PROBE` is set.
pub fn time(category: &'static str) -> Option<Timing> {
    enabled().then(|| Timing {
        category,
        start: Instant::now(),
    })
}

/// Guard returned by [`time`] that records its lifetime into a probe category on drop.
pub struct Timing {
    category: &'static str,
    start: Instant,
}

impl Drop for Timing {
    fn drop(&mut self) {
        record(self.category, Some(self.start.elapsed()));
    }
}

fn flush(probe: &mut Probe, interval: Duration) {
    let secs = interval.as_secs_f64().max(f64::EPSILON);
    for (category, stats) in &probe.stats {
        let rate = stats.count as f64 / secs;
        if stats.total.is_zero() {
            log::info!(target: "cpu_probe", "{category}: {} ({rate:.0}/s)", stats.count);
        } else {
            let avg_ms = stats.total.as_secs_f64() * 1000.0 / stats.count.max(1) as f64;
            let max_ms = stats.max.as_secs_f64() * 1000.0;
            let busy = stats.total.as_secs_f64() / secs * 100.0;
            log::info!(
                target: "cpu_probe",
                "{category}: {} ({rate:.0}/s) avg {avg_ms:.2}ms max {max_ms:.2}ms busy {busy:.1}%",
                stats.count,
            );
        }
    }
    probe.stats.clear();
}
