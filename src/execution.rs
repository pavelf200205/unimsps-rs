//! Bound asynchronous work at chunk boundaries so temporary buffers can be reused.
use burn::tensor::Device;
use std::sync::atomic::{AtomicUsize, Ordering};

static QUEUE_DEPTH: AtomicUsize = AtomicUsize::new(2);

pub fn configure(depth: usize) { QUEUE_DEPTH.store(depth, Ordering::Relaxed); }
pub fn depth() -> usize { QUEUE_DEPTH.load(Ordering::Relaxed) }

pub(crate) fn checkpoint(device: &Device, completed: usize) {
    let depth = depth();
    if depth != 0 && completed % depth == 0 { finish(device); }
}

pub(crate) fn finish(device: &Device) {
    // Verbose profiling already waits at finer boundaries. Normal execution
    // waits only between bounded groups, never after every individual operator.
    if depth() != 0 && !crate::profile::enabled() {
        device.sync().expect("GPU completion failed at inference queue boundary");
    }
}
