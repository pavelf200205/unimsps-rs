# Persistent preparation and bounded execution

CubeCL 0.11 supports persistent compiled SPIR-V artifacts and autotune decisions.
The project now enables cubecl/vulkan and cubecl/persistence and configures the
runtime store before initializing the device. No CUDA dependency is required.
The runtime database is under <kernel-cache-dir>/runtime, with a separate named
environment for each executable build. Moving the working directory does not
change this root. --kernel-cache-dir and UNIMSPS_CACHE_DIR select the root.
--no-kernel-cache disables compiled-artifact and tuning persistence.

The first launch may still compile and tune many shapes. Subsequent launches
of the same executable reuse those artifacts. Rebuilding the executable selects
a new environment to avoid stale artifacts across code changes. Copying the
executable without changing its size/timestamp preserves its environment name.
Autotune keys and compilation stores provide additional runtime/device namespacing.

The workload JSON is historical metadata only. It never bypasses process-local
setup. Startup uses CubeCL DryRun for shape-driven model passes: ordinary kernels
are loaded/compiled but not dispatched. Autotuning measurements still execute if
no tuning decision exists. A final GPU synchronization drains all deferred fusion
work before the guard is dropped, so actual inference runs normally afterwards.
Warmup does not read generated outputs. Sparse pool buckets 256..32768 are
prepared explicitly in addition to dense 32/64/128/256-scale networks.
--no-warmup skips this replay, at the cost of first-use setup during inference.

SPIR-V artifact persistence applies to the Vulkan compiler path. DX12/Metal via
WGSL still need driver compilation; autotune persistence remains useful there.
A disk artifact is not a serialized live device or driver pipeline. Some pipeline
creation, fusion setup, buffer allocation and shapes outside preparation remain
process-local, so zero startup cost or identical first/later views is not promised.

# Memory access and quality

Only pool transformers use foreground gathering. Light transformers remain dense
because later encoder stages use global attention. Decoder dependency-cone masks,
FP32 normal blending and prior-halo handling are unchanged.

Gather and inverse scatter use one output element per lane and contiguous output
channels. Foreground indices remain in raster order. The inverse map has one entry
per output pixel: a foreground row or -1. It directly writes zero for background,
with no serial scatter loop, atomics, trash slot or repeated padding accumulation.
The fixed bucket/output shapes avoid compiling for every foreground count. Small
power-of-two buckets restore the previous low padding cost; 8192 is supported.
Non-Cube backends keep the generic valid-row fallback.

--queue-depth now counts completed patches, not inner attention/set-transformer
chunks. Default 1 bounds CPU lead at one complete patch; inner work is submitted
asynchronously. 0 removes patch waits. Skipped patches do not consume queue slots.
--pool-cache-mib is an unused reservation threshold checked at scale boundaries,
not a hard VRAM cap. It no longer purges based on allocations still in use, and
view readback no longer unconditionally discards reusable buffers/uniform metadata.
0 disables explicit trimming. Memory use can exceed this threshold due to live
activations, queued work, weights and cached input images.

# Manual validation

Rebuild with cargo build --release. The Vulkan compiler/persistence features may
require Cargo.lock updates and dependency downloads on the first build.
Launch the same workload twice with the same binary and cache directory. Compare
startup preparation time, first-view time, later-view time and maximum patch delay.
Use normal execution for throughput; --verbose adds GPU synchronization.
Do not reuse a cache marker as evidence of successful artifact retrieval.

This change was inspected in source only. No compilation, inference, benchmarks
or numerical comparisons were executed by the coding agent. Startup savings,
steady-state speed, driver behavior and VRAM consumption remain to be measured.
