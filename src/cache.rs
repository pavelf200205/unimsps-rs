//! Persistent CubeCL artifacts and process-local compile-only replay metadata.
use serde::{Deserialize,Serialize};
use std::{collections::BTreeSet,path::{Path,PathBuf},sync::OnceLock};
use cubecl::config::{CubeClRuntimeConfig,RuntimeConfig,cache::CacheConfig};

#[derive(Default,Serialize,Deserialize)]
pub struct KernelCache {
    pub version: u32,
    pub warmed_signatures: BTreeSet<String>,
    pub unavailable_flash_shapes: BTreeSet<String>,
}
const CACHE_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum CacheMode { Shaders, Full, Memory }
impl CacheMode {
    pub fn persistent(self) -> bool { self != Self::Memory }
    pub fn name(self) -> &'static str {
        match self { Self::Shaders => "shaders", Self::Full => "full", Self::Memory => "memory" }
    }
}

pub fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("UNIMSPS_CACHE_DIR") {
        if !dir.trim().is_empty() { return PathBuf::from(dir); }
    }
    std::env::current_exe().ok().and_then(|p|p.parent().map(Path::to_path_buf))
        .unwrap_or_else(||PathBuf::from(".")).join(".unimsps-kernels")
}

/// Source/dependency/toolchain identity emitted by build.rs. Renaming or copying
/// an executable does not invalidate its cache.
pub fn build_id() -> &'static str {
    env!("UNIMSPS_CACHE_BUILD_ID")
}

/// Must run before any device/configuration is initialized. The runtime store
/// persists SPIR-V and autotune choices; a manifest alone never skips replay.
pub fn configure_runtime(cache_dir: &Path, mode: CacheMode) {
    install_warning_logger();
    use cubecl_environment::persistence::{set_persistence_policy, PersistencePolicy};
    set_persistence_policy(match mode {
        CacheMode::Shaders => PersistencePolicy::Shaders,
        CacheMode::Full => PersistencePolicy::All,
        CacheMode::Memory => PersistencePolicy::Memory,
    });
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    // Keep process-local reuse in every mode; route persistence at Storage.
    // Disabling throughput caching also disables RAM reuse and repeats probes.
    config.compilation.cache = true;
    config.autotune.disable_cache = false;
    config.throughput.disable_cache = false;
    config.environment.path = CacheConfig::Directory(cache_dir.join("runtime"));
    config.environment.name = format!("unimsps-{}",build_id());
    if !CubeClRuntimeConfig::try_set(config) {
        eprintln!("Runtime configuration was already initialized; cache settings could not be applied.");
    }
    eprintln!("Cache mode: {}; build: {}",mode.name(),build_id());
    if mode.persistent() {
        eprintln!("Persistent CubeCL store: {} (environment unimsps-{})",cache_dir.join("runtime").display(),build_id());
        probe_database();
    }
}

pub fn init_environment(cache_dir: &Path, _explicit_custom_dir: bool) -> KernelCache {
    if let Err(error) = std::fs::create_dir_all(cache_dir) {
        eprintln!("Cannot create cache directory {}: {error}",cache_dir.display());
    }
    let cache = std::fs::read(cache_dir.join(format!("kernel-state-{}.json",build_id()))).ok()
        .and_then(|b|serde_json::from_slice::<KernelCache>(&b).ok())
        .filter(|c|c.version==CACHE_VERSION)
        .unwrap_or_else(||KernelCache { version:CACHE_VERSION,..Default::default() });
    // Flash availability depends on the device/runtime. Rediscover it in each
    // process instead of loading another adapter's shape-only negative cache.
    cache
}
impl KernelCache {
    /// Historical hint only; not evidence of live pipelines in this process.
    pub fn is_warmed(&self,signature:&str)->bool { self.warmed_signatures.contains(signature) }
    pub fn record_and_save(&mut self,cache_dir:&Path,signature:Option<String>) {
        if let Some(signature)=signature { self.warmed_signatures.insert(signature); }
        self.unavailable_flash_shapes.extend(crate::attention::snapshot_unavailable());
        self.version=CACHE_VERSION;
        let saved = (|| -> Result<(),Box<dyn std::error::Error>> {
            std::fs::create_dir_all(cache_dir)?;
            // Runtime artifacts live in the CubeCL database; this file only records workloads.
            std::fs::write(cache_dir.join(format!("kernel-state-{}.json",build_id())),serde_json::to_vec_pretty(self)?)?;
            Ok(())
        })();
        if let Err(error)=saved { eprintln!("Could not save warmup metadata: {error}"); }
    }
}

// The runtime otherwise reports database failures through an uninstalled log
// facade and silently uses process memory. Always expose cache-related warnings.
struct CacheWarningLogger;
impl log::Log for CacheWarningLogger {
    fn enabled(&self, metadata:&log::Metadata<'_>)->bool {
        metadata.level()<=log::Level::Warn &&
            (metadata.target().starts_with("cubecl") || metadata.target().starts_with("turso"))
    }
    fn log(&self, record:&log::Record<'_>) {
        if !self.enabled(record.metadata()) { return; }
        static SEEN: OnceLock<std::sync::Mutex<BTreeSet<String>>> = OnceLock::new();
        let message: String = format!("{}: {}",record.target(),record.args()).chars().take(2048).collect();
        if let Ok(mut seen)=SEEN.get_or_init(||std::sync::Mutex::new(BTreeSet::new())).lock() {
            if seen.len()<32 && seen.insert(message.clone()) { eprintln!("[cache warning] {message}"); }
        }
    }
    fn flush(&self) {}
}
fn install_warning_logger() {
    static LOGGER: CacheWarningLogger = CacheWarningLogger;
    if log::set_logger(&LOGGER).is_ok() { log::set_max_level(log::LevelFilter::Warn); }
}

fn database() -> Result<cubecl_environment::persistence::Database,String> {
    use cubecl_environment::persistence::{Access,Database};
    Database::open_at(&cubecl::environment::path(),Access::ReadWrite)
}

fn probe_database() {
    use cubecl_environment::persistence::{Insertion,Origin};
    let checked=(|| -> Result<(),String> {
        let db=database()?;
        let namespace="unimsps/persistence-probe/v1";
        let existed=db.get(namespace,b"probe").as_ref().is_some_and(|v|&v[..]==b"ok");
        if !existed {
            match db.insert(namespace,b"probe",b"ok",Origin::Local) {
                Insertion::Stored=>{},
                Insertion::Conflict(value) if &value[..]==b"ok"=>{},
                other=>return Err(format!("database write failed: {other:?}")),
            }
        }
        if !db.get(namespace,b"probe").as_ref().is_some_and(|v|&v[..]==b"ok") {
            return Err("cache probe could not be read back".into());
        }
        eprintln!("Cache database read/write verified; previous-process probe: {}",if existed {"found"} else {"not found (new cache)"});
        print_summary(&db,"startup");
        Ok(())
    })();
    if let Err(error)=checked {
        eprintln!("CACHE PERSISTENCE FAILED: {error}. Inference can continue, but cache reuse across launches is unavailable.");
    }
}

fn print_summary(db:&cubecl_environment::persistence::Database,phase:&str) {
    let rows=db.summary();
    let total: u64=rows.iter().map(|r|r.entries).sum();
    let shader: u64=rows.iter().filter(|r|r.namespace.starts_with("vulkan/") || r.namespace.starts_with("cuda/")).map(|r|r.entries).sum();
    let tune: u64=rows.iter().filter(|r|r.namespace.starts_with("autotune/")).map(|r|r.entries).sum();
    let bytes: u64=rows.iter().map(|r|r.bytes).sum();
    eprintln!("Cache {phase}: {total} entries, {shader} compiled artifacts, {tune} tuning decisions, {:.1} MiB of payload",bytes as f64/(1024.0*1024.0));
}

/// Also makes the main .db size meaningful: Turso does not checkpoint on close.
pub fn checkpoint_and_report(phase:&str) {
    match database() {
        Ok(db)=>{
            print_summary(&db,phase);
            match db.checkpoint() {
                Ok(true)=>{},
                Ok(false)=>eprintln!("Cache checkpoint is busy; committed entries remain in the -wal sidecar."),
                Err(error)=>eprintln!("Cache checkpoint failed: {error}; keep the -wal sidecar."),
            }
            let path=cubecl::environment::path();
            if let Ok(metadata)=std::fs::metadata(&path) { eprintln!("Cache database: {} ({} bytes)",path.display(),metadata.len()); }
        },
        Err(error)=>eprintln!("Cannot inspect/checkpoint the cache database: {error}"),
    }
}
