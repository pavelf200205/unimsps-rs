//! Optional wall-clock tracing, plus explicitly requested device timing windows.
use burn::tensor::Device;
use burn_backend::{ExecutionError, ProfileDuration, ProfileOptions, TimingMethod};
use serde_json::{Value,json};
use std::{path::Path,sync::{Mutex,OnceLock,atomic::{AtomicBool,Ordering}},time::{Instant,SystemTime,UNIX_EPOCH}};
static ENABLED: AtomicBool=AtomicBool::new(false);
static GPU_ENABLED: AtomicBool=AtomicBool::new(false);
static PATCH_OPEN: AtomicBool=AtomicBool::new(false);
struct State { start:Instant, rows:Vec<Value>, pending:Vec<(usize,ProfileDuration)> }
fn state()->&'static Mutex<State> { static STATE:OnceLock<Mutex<State>>=OnceLock::new(); STATE.get_or_init(||Mutex::new(State {start:Instant::now(),rows:Vec::new(),pending:Vec::new()})) }
pub fn configure(enabled:bool) { ENABLED.store(enabled,Ordering::Relaxed); }
pub fn configure_gpu(enabled:bool) {
    GPU_ENABLED.store(enabled,Ordering::Relaxed);
    if enabled {
        configure(true);
        eprintln!("[trace] Device timing enabled: timestamp windows change command boundaries and add query readbacks. Results resolve after each view; system-clock fallback is labeled explicitly.");
    }
}
pub fn gpu_enabled()->bool { GPU_ENABLED.load(Ordering::Relaxed) }
pub fn enabled()->bool { ENABLED.load(Ordering::Relaxed) }
pub fn begin_view() { if enabled() { if let Ok(mut s)=state().lock() { s.start=Instant::now();s.rows.clear();s.pending.clear(); } } }
fn append(mut row:Value) {
    if let Ok(mut s)=state().lock() {
        row["view_offset_ms"]=json!(s.start.elapsed().as_secs_f64()*1000.0);
        row["unix_ms"]=json!(SystemTime::now().duration_since(UNIX_EPOCH).map_or(0,|d|d.as_millis() as u64));
        s.rows.push(row);
    }
}
pub struct Span {start:Instant,label:&'static str}
pub fn span(label:&'static str)->Option<Span> { enabled().then(||Span {start:Instant::now(),label}) }
impl Drop for Span {fn drop(&mut self) { append(json!({"kind":self.label,"ms":self.start.elapsed().as_secs_f64()*1000.0})); }}

/// CPU wall time spent issuing a stage, including any runtime backpressure.
/// No flush or synchronization is inserted at these boundaries.
pub struct Detail {start:Instant,label:String}
pub fn detail(label:impl FnOnce()->String)->Option<Detail> {
    (gpu_enabled() && PATCH_OPEN.load(Ordering::Relaxed))
        .then(||Detail {start:Instant::now(),label:label()})
}
impl Drop for Detail {
    fn drop(&mut self) {
        append(json!({"kind":"network_submit_stage","label":self.label,"ms":self.start.elapsed().as_secs_f64()*1000.0}));
    }
}

/// Bracket submitted patch work on the backend's execution stream. The caller
/// flushes the patch's final operations inside `work`, exactly as in normal
/// execution. Do not resolve here: that would add a wait before the usual patch
/// completion boundary and could hide the problem we are measuring.
pub fn gpu_window<O:Send+'static>(device:&Device,size:usize,y:usize,x:usize,
    work:impl FnOnce()->O+Send)->Result<O,ExecutionError> {
    if !gpu_enabled() { return Ok(work()); }
    let start=Instant::now();
    let mut work_ms=0.0;
    let (output,duration)=device.profile_with(ProfileOptions::default(),|| {
        let begin=Instant::now();
        let output=work();
        work_ms=begin.elapsed().as_secs_f64()*1000.0;
        output
    })?;
    let bracket_ms=start.elapsed().as_secs_f64()*1000.0;
    let mut s=state().lock().expect("patch trace lock");
    let index=s.rows.len();
    let row=json!({"kind":"patch_device_window","scale":size,"y":y,"x":x,
        "profile_bracket_wall_ms":bracket_ms,"work_submission_wall_ms":work_ms,
        "profile_setup_and_close_wall_ms":(bracket_ms-work_ms).max(0.0),
        "view_offset_ms":s.start.elapsed().as_secs_f64()*1000.0,
        "unix_ms":SystemTime::now().duration_since(UNIX_EPOCH).map_or(0,|d|d.as_millis() as u64)});
    s.rows.push(row);
    s.pending.push((index,duration));
    Ok(output)
}

pub struct Patch {start:Instant,last:Instant,size:usize,y:usize,x:usize,foreground:usize,phases:serde_json::Map<String,Value>}
impl Patch {
    pub fn start(size:usize,y:usize,x:usize,foreground:usize)->Option<Self> {
        enabled().then(||{PATCH_OPEN.store(true,Ordering::Relaxed);let now=Instant::now(); Self {start:now,last:now,size,y,x,foreground,phases:Default::default()} })
    }
    pub fn mark(&mut self,name:&str) {let now=Instant::now();self.phases.insert(name.into(),json!(now.duration_since(self.last).as_secs_f64()*1000.0));self.last=now;}
    pub fn finish(self,device:&Device,completed:usize) {
        PATCH_OPEN.store(false,Ordering::Relaxed);
        let milliseconds=self.start.elapsed().as_secs_f64()*1000.0;
        // Pool telemetry only after a wait the existing execution policy already
        // requested. At queue depth 0 it is deliberately absent.
        let depth=crate::execution::depth();
        let completed_on_device=crate::profile::enabled() || (depth!=0 && completed%depth==0);
        let memory=if completed_on_device {device.memory_pool_usage().map(|u|json!({"reserved_bytes":u.bytes_reserved,"in_use_bytes":u.bytes_in_use,"padding_bytes":u.bytes_padding,"allocations":u.number_allocs}))} else {None};
        if milliseconds>2000.0 {eprintln!("[trace] slow patch size={} y={} x={} total={:.0}ms phases={:?} pool={:?}",self.size,self.y,self.x,milliseconds,self.phases,memory);}
        append(json!({"kind":"patch","scale":self.size,"y":self.y,"x":self.x,"foreground":self.foreground,"ms":milliseconds,"phases_ms":self.phases,"completion_waited":completed_on_device,"pool":memory}));
    }
}
pub fn write_report(path:&Path,context:Value)->std::io::Result<()> {
    if !enabled() {return Ok(());}
    let (mut rows,pending)={
        let mut s=state().lock().expect("patch trace lock");
        (std::mem::take(&mut s.rows),std::mem::take(&mut s.pending))
    };
    // Normal-map readback has already completed. Keep timing-query resolution
    // out of the patch measurements, and never hold the trace lock while waiting.
    for (index,duration) in pending {
        let method=duration.timing_method();
        let start=Instant::now();
        let ticks=cubecl_environment::future::block_on(duration.resolve());
        let elapsed=ticks.map(|t|t.duration().as_secs_f64()*1000.0);
        rows[index]["timing_method"]=json!(method.to_string());
        rows[index]["device_elapsed_ms"]=json!(if method==TimingMethod::Device {elapsed} else {None});
        rows[index]["system_elapsed_ms"]=json!(if method==TimingMethod::System {elapsed} else {None});
        rows[index]["resolve_wall_ms"]=json!(start.elapsed().as_secs_f64()*1000.0);
    }
    let document=json!({"format_version":2,"context":context,
        "gpu_timing_enabled":gpu_enabled(),"extra_gpu_waits":gpu_enabled(),
        "timing":"CPU wall time; network submission can block on runtime backpressure. Device windows measure elapsed stream time, including gaps/preemption, not summed kernel busy time. Device profiling may add waits on fallback/Metal backends and changes command boundaries. Stage rows are CPU submission times and may overlap parent patch times.",
        "pool_scope":"this process only; not adapter budget or other applications","rows":rows});
    std::fs::write(path,serde_json::to_vec_pretty(&document)?)
}
