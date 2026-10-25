//! UniMSPS startup-only persistence routing. Store semantics remain upstream.
use core::sync::atomic::{AtomicU8, Ordering};

/// Configure before opening any stores or initializing a runtime. Existing
/// stores are not migrated; never change this during a running workload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PersistencePolicy {
    /// Unmodified upstream routing.
    All = 0,
    /// Persist compiler artifacts only; all other stores use process memory.
    Shaders = 1,
    /// Keep every store in process memory.
    Memory = 2,
}
static POLICY: AtomicU8 = AtomicU8::new(0);
pub fn set_persistence_policy(policy: PersistencePolicy) {
    POLICY.store(policy as u8, Ordering::Relaxed);
}
pub fn persistence_policy() -> PersistencePolicy {
    match POLICY.load(Ordering::Relaxed) {
        1 => PersistencePolicy::Shaders,
        2 => PersistencePolicy::Memory,
        _ => PersistencePolicy::All,
    }
}
pub(crate) fn is_shader(namespace: &str) -> bool {
    // CubeCL 0.11.0 compiler stores use these backend namespace prefixes.
    namespace.starts_with("vulkan/") || namespace.starts_with("cuda/")
}

#[cfg(std_io)]
mod telemetry {
    use core::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;
    static HITS: AtomicU64 = AtomicU64::new(0);
    static MISSES: AtomicU64 = AtomicU64::new(0);
    static WRITES: AtomicU64 = AtomicU64::new(0);
    static FAILURES: AtomicU64 = AtomicU64::new(0);
    static READ_NS: AtomicU64 = AtomicU64::new(0);
    static WRITE_NS: AtomicU64 = AtomicU64::new(0);

    /// Process totals for shader database operations, not GPU execution times.
    /// A hit means bytes were fetched; successful decoding/loading is a later step.
    #[derive(Clone, Copy, Debug, Default, serde::Serialize)]
    pub struct ShaderCacheStats {
        pub read_hits: u64,
        pub read_misses: u64,
        pub accepted_writes: u64,
        pub write_failures: u64,
        pub read_ns: u64,
        pub write_ns: u64,
    }
    pub fn shader_cache_stats() -> ShaderCacheStats {
        ShaderCacheStats {
            read_hits: HITS.load(Ordering::Relaxed),
            read_misses: MISSES.load(Ordering::Relaxed),
            accepted_writes: WRITES.load(Ordering::Relaxed),
            write_failures: FAILURES.load(Ordering::Relaxed),
            read_ns: READ_NS.load(Ordering::Relaxed),
            write_ns: WRITE_NS.load(Ordering::Relaxed),
        }
    }
    pub(crate) fn record_read(namespace: &str, hit: bool, duration: Duration) {
        if !super::is_shader(namespace) { return; }
        (if hit { &HITS } else { &MISSES }).fetch_add(1, Ordering::Relaxed);
        READ_NS.fetch_add(duration.as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
    }
    pub(crate) fn record_write(namespace: &str, stored: bool, failed: bool, duration: Duration) {
        if !super::is_shader(namespace) { return; }
        if stored { WRITES.fetch_add(1, Ordering::Relaxed); }
        if failed { FAILURES.fetch_add(1, Ordering::Relaxed); }
        WRITE_NS.fetch_add(duration.as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
    }
}
#[cfg(std_io)]
pub use telemetry::{ShaderCacheStats, shader_cache_stats};
#[cfg(std_io)]
pub(crate) use telemetry::{record_read, record_write};
