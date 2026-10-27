# Persistent preparation and bounded execution

## Current driver-cache policy (2026-10-27)

Windows Vulkan inference is now supervised by `src/driver_cache.rs`: each worker
gets a disposable NVIDIA GLCache copied from an immutable first-successful-run
seed. The supervisor waits for worker exit before publishing a new seed. Later
sessions never update that seed. See the README's **NVIDIA driver-cache snapshots**
section for paths, refresh behavior, concurrency and limitations.

The sections below retain historical implementation notes. In current code the
CubeCL database environment is schema-named `unimsps-v4`, not revision-named;
only the new driver snapshot profile includes the executable content hash.
Current warmup executes model passes with pool cleanup; it is not a DryRun-only
replay. Database write freezing does not prohibit tuning on cache misses.
These statements supersede older descriptions below.

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

# Windows persistence repair

A zero-byte main .db with no WAL sidecar was traced to the installed
cubecl-environment 0.11.0 opening Turso 0.8.3 with experimental multiprocess WAL.
Turso's default WindowsIO lacks shared-WAL byte-locking/mapping primitives, and
database initialization is rejected. CubeCL silently falls back to MemoryStorage
when its log warnings have no installed sink.

The repository-local Cargo patch in vendor/cubecl-environment disables that
experimental mode on Windows only. Normal Windows database locking remains in
use; Unix multiprocess WAL is unchanged. Concurrent Windows instances should
use separate cache directories, since the default Windows backend locks the
cache file rather than supporting concurrent multiprocess WAL writers.
The patch does not modify the global Cargo registry or neural inference code.

A cache-only warning logger now exposes database failures. Startup explicitly
opens the active database, writes and reads a small probe if absent, and reports
whether it found the probe from an earlier process. This checks real durable
storage rather than a JSON warmup marker. Namespace statistics report compiled
artifacts, autotune decisions and payload bytes at startup, after preparation
and at normal exit.

Turso does not automatically checkpoint on close. An explicit checkpoint after
preparation and at exit folds committed WAL frames into the main .db. While a
program is running, an unchanged/small .db alone does not imply failure: entries
may still be in the -wal sidecar. Never remove a live -wal file. A failed/busy
checkpoint is reported; the committed WAL remains part of the database.

After rebuilding, the first run creates a new environment (build ID changes).
On a second run of that exact binary with identical options, look for
previous-process probe: found and nonzero compiled-artifact/tuning counts at
startup. Preparation still replays shapes to hydrate process-local pipelines.
The agent did not compile, open the database through Turso, or run inference;
the repair is based on source inspection and must be verified by the user's run.

# Diagnosing intermittent execution stalls

--trace-performance writes timing_trace.json per view without adding device
synchronizations. It records patch preparation, network submission, blending,
flush/completion wait, and pool usage after waits already requested by the
queue policy. Network submission is not pure CPU time: it can include blocked
driver calls, tuning, compilation or GPU work. At queue depth 0, times can
represent submission rather than completion and pool samples are omitted.
Slow patches over two seconds print a concise console record. Allocator checks
and cleanup durations are recorded separately. Disabled by default.

Run this without --verbose when diagnosing real throughput. Pool data only
describes this process's allocations; it does not report the Windows adapter
budget, other processes, GPU clock, power or temperature. The trace is not a
GPU kernel profiler. No scheduling or inference behavior was changed for this
diagnostic feature.

For external contention, tools/monitor_gpu.ps1 can be launched manually in
a second PowerShell window. It records Windows GPU engine utilization and
dedicated/shared memory counters, including process IDs and names, plus available
system RAM. Example from the unimsps-rs folder:

powershell -File tools/monitor_gpu.ps1 -Out experiments/gpu_resources.csv -Seconds 900

Counter names may need localization on non-English Windows installations.
The script does not change power, clocks or priority and has not been executed
by the agent. Per-engine utilization should not be blindly summed across engines
or adapters. Clocks/temperature need a separate hardware monitor.

# Device timestamps for intermittent stalls

`--trace-gpu` implies `--trace-performance` and adds one device timing window
per executed patch, plus CPU submission spans for each encoder batch, pooling,
light transformer and decoder. It cannot be combined with `--verbose`. This is
diagnostic instrumentation, not a change to the model, queue depth or allocator
policy. It uses Burn's portable profiling API, without a CUDA/NVIDIA dependency.

The patch still flushes its final operations and uses the existing patch wait.
Timestamp results are retained and resolved after the normal map's readback,
when writing `timing_trace.json`. Profiling itself changes command boundaries,
adds timestamp readbacks, and can add backend waits (notably Metal and the
system-clock fallback). Do not treat profiled throughput as an unaffected
baseline. A disappearance of the stalls only with this flag is useful evidence
about command boundaries/polling, not proof that normal execution is fixed.

Trace format 2 splits `flush_and_completion_wait` into `final_flush` and
`completion_wait`. With device profiling enabled, `gpu_window_close` also
records the host time needed to close the window. `patch_device_window` rows
carry the same scale/x/y as their patch and report `timing_method` plus
`device_elapsed_ms`, or `system_elapsed_ms` for a fallback. Missing measurements
are null, never zero. Device elapsed time spans the command stream's first and
last sampled work; it includes intervening submission gaps/preemption and is
not the sum of individual kernels' busy times. CPU stage rows may reflect
runtime backpressure and lazy work from previous stages.

Use a new output folder and keep the other inference settings unchanged. One
nine-image run that includes both fast and stalled patches is useful; there is
no need to repeat the cache-isolation sequence or run monitor_gpu.ps1. Rebuild
manually with `cargo build --release`, then add `--trace-gpu` to the usual
command. The new executable gets a new cache environment, so first preparation
can take longer. Neither compilation nor inference was run by the coding agent.
