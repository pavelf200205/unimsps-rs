//! Bound asynchronous work at chunk/stage boundaries without forcing host-blocking GPU stalls by default.
//! Borrows `supernormal-rs`'s `trim_memory` pool threshold to bound reserved WGPU memory without per-chunk syncs.
use burn::tensor::Device;
use std::sync::atomic::{AtomicUsize, Ordering};

static QUEUE_DEPTH: AtomicUsize = AtomicUsize::new(0);
static POOL_CACHE_BYTES: AtomicUsize = AtomicUsize::new(1536 * 1024 * 1024);

pub fn configure(depth: usize) { QUEUE_DEPTH.store(depth, Ordering::Relaxed); }
pub fn depth() -> usize { QUEUE_DEPTH.load(Ordering::Relaxed) }
pub fn configure_pool_cache(mib: usize) {
    POOL_CACHE_BYTES.store(mib.saturating_mul(1024 * 1024), Ordering::Relaxed);
}

pub(crate) fn checkpoint(device: &Device, completed: usize) {
    let depth = depth();
    if depth != 0 && completed % depth == 0 {
        finish(device);
    }
}

pub(crate) fn flush(device: &Device) {
    // Submit recorded fusion streams asynchronously when queue_depth == 0,
    // or wait synchronously only when an explicit positive queue_depth is configured.
    if !crate::profile::enabled() {
        if depth() != 0 {
            device.sync().expect("GPU completion failed at inference queue boundary");
        } else {
            let _ = device.flush();
        }
    }
}

pub(crate) fn trim_memory(device: &Device) {
    let threshold = POOL_CACHE_BYTES.load(Ordering::Relaxed);
    if threshold == 0 { return; }
    let _profile = crate::profile::span(device, || "allocator/check-and-trim".into());
    let _ = device.flush();
    if let Some(usage) = device.memory_pool_usage() {
        if usage.bytes_reserved > threshold as _ {
            crate::profile::event(|| format!("allocator/cleanup reserved_mib={}", usage.bytes_reserved / (1024 * 1024)));
            device.sync().expect("GPU completion failed before allocator cleanup");
            device.memory_cleanup();
        }
    }
}

pub(crate) fn finish(device: &Device) {
    // Verbose profiling already waits at finer boundaries. Normal execution
    // waits only when an explicit non-zero queue depth is requested.
    if depth() != 0 && !crate::profile::enabled() {
        device.sync().expect("GPU completion failed at inference queue boundary");
    }
}
