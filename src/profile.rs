//! Opt-in synchronized wall-clock profiling, portable across Burn devices.
//! Waiting at boundaries perturbs fusion/overlap; these are not GPU timestamps.
use burn::tensor::Device;
use serde::Serialize;
use std::{cell::RefCell, collections::BTreeMap, path::Path, sync::{Mutex, OnceLock, atomic::{AtomicBool, Ordering}}, time::{Duration, Instant}};

static ENABLED: AtomicBool = AtomicBool::new(false);
#[derive(Default)]
struct State { rows: BTreeMap<String, Row>, events: BTreeMap<String, u64>, errors: Vec<String> }
#[derive(Clone, Serialize)]
struct Row {
    label: String,
    calls: u64,
    total_ms: f64,
    self_ms: f64,
    first_ms: f64,
    min_ms: f64,
    max_ms: f64,
    mean_ms: f64,
}
fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}
thread_local! { static CHILDREN: RefCell<Vec<Duration>> = const { RefCell::new(Vec::new()) }; }

pub fn enabled() -> bool { ENABLED.load(Ordering::Relaxed) }
/// May be called inside a queued custom operation: no GPU synchronization here.
pub fn event(label: impl FnOnce() -> String) {
    if enabled() { *state().lock().unwrap().events.entry(label()).or_default() += 1; }
}
pub fn configure(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
    if enabled {
        eprintln!("[profile] Synchronized wall times enabled. GPU waits perturb fusion/overlap; compilation and autotuning are included on first use. Compare normal-run throughput without --verbose.");
    }
}
fn synchronize(device: &Device, label: &str) {
    if let Err(error) = device.sync() {
        let message = format!("{label}: {error:?}");
        eprintln!("[profile] Synchronization failed: {message}");
        state().lock().unwrap().errors.push(message);
    }
}

pub struct Span { device: Device, label: String, start: Instant, announce: bool }
pub fn span(device: &Device, label: impl FnOnce() -> String) -> Option<Span> {
    begin(device, label, false)
}
pub fn phase(device: &Device, label: impl FnOnce() -> String) -> Option<Span> {
    begin(device, label, true)
}
fn begin(device: &Device, label: impl FnOnce() -> String, announce: bool) -> Option<Span> {
    if !enabled() { return None; }
    let label = label();
    synchronize(device, &label);
    CHILDREN.with(|stack| stack.borrow_mut().push(Duration::ZERO));
    Some(Span { device: device.clone(), label, start: Instant::now(), announce })
}
impl Drop for Span {
    fn drop(&mut self) {
        if std::thread::panicking() { return; }
        synchronize(&self.device, &self.label);
        let elapsed = self.start.elapsed();
        let children = CHILDREN.with(|stack| {
            let mut stack = stack.borrow_mut();
            let children = stack.pop().unwrap_or_default();
            if let Some(parent) = stack.last_mut() { *parent += elapsed; }
            children
        });
        let ms = elapsed.as_secs_f64() * 1000.0;
        let self_ms = elapsed.saturating_sub(children).as_secs_f64() * 1000.0;
        let mut state = state().lock().unwrap();
        let row = state.rows.entry(self.label.clone()).or_insert_with(|| Row {
            label: self.label.clone(), calls: 0, total_ms: 0.0, self_ms: 0.0,
            first_ms: ms, min_ms: ms, max_ms: ms, mean_ms: 0.0,
        });
        row.calls += 1;
        row.total_ms += ms;
        row.self_ms += self_ms;
        row.min_ms = row.min_ms.min(ms);
        row.max_ms = row.max_ms.max(ms);
        row.mean_ms = row.total_ms / row.calls as f64;
        if self.announce { eprintln!("[profile] {}: {:.3} ms (self {:.3} ms)", self.label, ms, self_ms); }
    }
}

/// Call after all spans for the object have closed. Retains all rows in JSON,
/// prints the 40 largest exclusive costs, then starts a fresh object report.
pub fn write_report(path: &Path, context: serde_json::Value) -> anyhow::Result<()> {
    if !enabled() { return Ok(()); }
    let mut state = state().lock().unwrap();
    let mut rows: Vec<_> = state.rows.values().cloned().collect();
    rows.sort_by(|a, b| b.self_ms.total_cmp(&a.self_ms));
    eprintln!("[profile] Top exclusive costs (ms); inclusive totals overlap and must not be summed:");
    eprintln!("    calls       self      total       mean      first        max  operation / shape");
    for row in rows.iter().take(40) {
        eprintln!("{:9} {:10.3} {:10.3} {:10.3} {:10.3} {:10.3}  {}", row.calls, row.self_ms, row.total_ms, row.mean_ms, row.first_ms, row.max_ms, row.label);
    }
    for (event, count) in &state.events { eprintln!("[profile] route {count} calls: {event}"); }
    let report = serde_json::json!({
        "timing": "synchronized wall-clock milliseconds, including CPU dispatch and first-use compilation/autotuning",
        "profiling_perturbs_execution": true, "valid": state.errors.is_empty(),
        "context": context, "synchronization_errors": state.errors, "operations": rows,
        "route_counts": state.events,
    });
    std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    eprintln!("[profile] Full report: {}", path.display());
    *state = State::default();
    Ok(())
}
