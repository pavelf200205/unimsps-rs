//! Submit inner chunks asynchronously; pace only complete patches.
use burn::tensor::Device;
use std::sync::atomic::{AtomicUsize,Ordering};
static QUEUE_DEPTH: AtomicUsize = AtomicUsize::new(1);
static POOL_CACHE_BYTES: AtomicUsize = AtomicUsize::new(1024 * 1024 * 1024);
pub fn configure(depth:usize) { QUEUE_DEPTH.store(depth,Ordering::Relaxed); }
pub fn depth()->usize { QUEUE_DEPTH.load(Ordering::Relaxed) }
pub fn configure_pool_cache(mib:usize) { POOL_CACHE_BYTES.store(mib.saturating_mul(1024*1024),Ordering::Relaxed); }
pub(crate) fn checkpoint(device:&Device,completed:usize) {
    // Never drain the GPU from inside attention or set-transformer chunks.
    let depth=depth();
    if depth==0 || completed % depth==0 { flush(device); }
}
pub(crate) fn flush(device:&Device) {
    if !crate::profile::enabled() { device.flush().expect("GPU submission failed"); }
}
pub(crate) fn finish(device:&Device) { flush(device); }
pub(crate) fn checkpoint_patch(device:&Device,completed:usize) {
    let depth=depth();
    if depth!=0 && completed % depth==0 && !crate::profile::enabled() {
        let _profile=crate::profile::span(device,||"queue/patch-completion".into());
        device.sync().expect("GPU completion failed at patch boundary");
    } else { flush(device); }
}
/// Called only at scale boundaries. Treat the budget as reusable *unused*
/// reservation, not a reason to purge buffers currently needed by the network.
pub(crate) fn trim_memory(device:&Device) {
    let threshold=POOL_CACHE_BYTES.load(Ordering::Relaxed) as u64;
    if threshold==0 { return; }
    let _profile=crate::profile::span(device,||"allocator/check-and-trim".into());
    device.flush().expect("GPU submission failed before allocator check");
    if let Some(usage)=device.memory_pool_usage() {
        let unused=usage.bytes_reserved.saturating_sub(usage.bytes_in_use).saturating_sub(usage.bytes_padding);
        crate::profile::event(||format!("allocator/reserved_mib={} unused_mib={}",usage.bytes_reserved/(1024*1024),unused/(1024*1024)));
        if unused>threshold {
            device.sync().expect("GPU completion failed before allocator cleanup");
            if let Some(now)=device.memory_pool_usage() {
                let free=now.bytes_reserved.saturating_sub(now.bytes_in_use).saturating_sub(now.bytes_padding);
                if free>threshold {
                    crate::profile::event(||"allocator/cleanup".into());
                    device.memory_cleanup();
                }
            }
        }
    }
}
pub(crate) fn cleanup_view(device:&Device) {
    // Readback has already completed. Keep reusable allocations and uniform
    // metadata resident for the next view; the next scale checks excess cache.
    flush(device);
}
