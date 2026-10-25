//! Submit inner chunks and patches with bounded in-flight queue depth and pool memory.
use burn::tensor::Device;
use std::sync::atomic::{AtomicUsize, Ordering};

static QUEUE_DEPTH: AtomicUsize = AtomicUsize::new(1);
static POOL_CACHE_BYTES: AtomicUsize = AtomicUsize::new(1024 * 1024 * 1024);

pub fn configure(depth: usize) {
    QUEUE_DEPTH.store(depth, Ordering::Relaxed);
}

pub fn depth() -> usize {
    QUEUE_DEPTH.load(Ordering::Relaxed)
}

pub fn configure_pool_cache(mib: usize) {
    POOL_CACHE_BYTES.store(mib.saturating_mul(1024 * 1024), Ordering::Relaxed);
}

/// Enforces `queue_depth` backpressure between attention / set-transformer chunks.
///
/// With `--queue-depth 1` (the default), synchronizing at chunk boundaries guarantees
/// that at most one `--score-budget-mib` (192-256 MiB) attention score matrix is in flight
/// on the GPU at a time, preventing both `cubecl-wgpu`'s 512-task `SubmissionLoad::regulate`
/// mid-patch stalls and Windows WDDM VidMm queue throttling.
pub(crate) fn checkpoint(device: &Device, completed: usize) {
    let depth = depth();
    if depth != 0 && completed % depth == 0 && !crate::profile::enabled() {
        let _profile = crate::profile::span(device, || "queue/chunk-completion".into());
        device.sync().expect("GPU completion failed at chunk boundary");
    } else if depth == 0 || completed % depth == 0 {
        flush(device);
    }
}

pub(crate) fn flush(device: &Device) {
    if !crate::profile::enabled() {
        device.flush().expect("GPU submission failed");
    }
}

pub(crate) fn finish(device: &Device) {
    let depth = depth();
    if depth != 0 && !crate::profile::enabled() {
        let _profile = crate::profile::span(device, || "queue/chunk-finish".into());
        device.sync().expect("GPU completion failed at chunk finish");
    } else {
        flush(device);
    }
}

pub(crate) fn checkpoint_patch(device: &Device, completed: usize) {
    let depth = depth();
    if depth != 0 && completed % depth == 0 && !crate::profile::enabled() {
        let _profile = crate::profile::span(device, || "queue/patch-completion".into());
        device.sync().expect("GPU completion failed at patch boundary");
    } else {
        flush(device);
    }
}

/// Checks pool reservation against `--pool-cache-mib` and trims unused pool slabs
/// whenever total `bytes_reserved` exceeds `threshold` or `unused` exceeds `threshold / 2`.
pub(crate) fn trim_memory(device: &Device) {
    let _trace = crate::trace::span("allocator/check-and-trim");
    let threshold = POOL_CACHE_BYTES.load(Ordering::Relaxed) as u64;
    if threshold == 0 {
        return;
    }
    let _profile = crate::profile::span(device, || "allocator/check-and-trim".into());
    device.flush().expect("GPU submission failed before allocator check");
    if let Some(usage) = device.memory_pool_usage() {
        let unused = usage
            .bytes_reserved
            .saturating_sub(usage.bytes_in_use)
            .saturating_sub(usage.bytes_padding);
        crate::profile::event(|| {
            format!(
                "allocator/reserved_mib={} unused_mib={}",
                usage.bytes_reserved / (1024 * 1024),
                unused / (1024 * 1024)
            )
        });
        if usage.bytes_reserved > threshold || unused > (threshold / 2) {
            device
                .sync()
                .expect("GPU completion failed before allocator cleanup");
            if let Some(now) = device.memory_pool_usage() {
                let free = now
                    .bytes_reserved
                    .saturating_sub(now.bytes_in_use)
                    .saturating_sub(now.bytes_padding);
                if now.bytes_reserved > threshold || free > (threshold / 2) {
                    crate::profile::event(|| "allocator/cleanup".into());
                    let _cleanup_trace = crate::trace::span("allocator/cleanup");
                    device.memory_cleanup();
                }
            }
        }
    }
}

/// Synchronizes the GPU and purges idle pool slabs at view / warmup boundaries so that
/// temporary allocations from warmup or earlier views do not fragment subsequent views.
pub(crate) fn cleanup_view(device: &Device) {
    let _trace = crate::trace::span("allocator/view-cleanup");
    device.sync().expect("GPU completion failed at view boundary");
    device.memory_cleanup();
}
