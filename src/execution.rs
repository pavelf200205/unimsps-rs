//! Submit inner chunks asynchronously; pace completed patches and bound total pool reservation.
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

pub(crate) fn checkpoint(_device: &Device, _completed: usize) {
    // Keep inner attention/set-transformer chunks inside the current stage's command
    // encoder rather than splitting a single patch into 25-55 micro-submissions;
    // encoder stages call `flush` and `checkpoint_patch` paces completed patches.
}

pub(crate) fn flush(device: &Device) {
    if !crate::profile::enabled() {
        device.flush().expect("GPU submission failed");
    }
}

pub(crate) fn finish(device: &Device) {
    flush(device);
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

/// Called only at scale boundaries. Reclaims unused pool slabs only when total
/// `bytes_reserved` exceeds `--pool-cache-mib` (`1024 MiB`), keeping the working
/// pool (~750-850 MiB) resident across all patches and views without WDDM churn.
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
        if usage.bytes_reserved > threshold {
            device
                .sync()
                .expect("GPU completion failed before allocator cleanup");
            if let Some(now) = device.memory_pool_usage() {
                if now.bytes_reserved > threshold {
                    crate::profile::event(|| "allocator/cleanup".into());
                    let _cleanup_trace = crate::trace::span("allocator/cleanup");
                    device.memory_cleanup();
                }
            }
        }
    }
}

pub(crate) fn cleanup_view(device: &Device) {
    // Readback has already synced the device. Keep reusable pool slabs resident
    // for the next view unless total reservation exceeded `--pool-cache-mib`.
    trim_memory(device);
}

/// Unconditionally synchronizes and purges temporary warmup allocations before timed views begin.
pub(crate) fn purge_pool(device: &Device) {
    let _trace = crate::trace::span("allocator/warmup-purge");
    device.sync().expect("GPU completion failed before warmup purge");
    device.memory_cleanup();
}
