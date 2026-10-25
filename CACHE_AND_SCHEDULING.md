# Persistence and scheduling

Dependencies: Burn 0.22.1, CubeCL 0.11.3, CubeK 0.3.1. CubeCL's patch releases
include server wakeup/scheduling and SPIR-V fixes. WGPU remains the default;
no CUDA dependency is required for these changes.

## Cache modes

| Option | Disk | Process-local reuse |
| --- | --- | --- |
| `--cache-mode shaders` (default) | Compiled Vulkan/CUDA artifacts | Shaders, tuning, throughput |
| `--cache-mode full` | Compiled artifacts, tuning, throughput, runtime records | All |
| `--cache-mode memory` | None | All |

`--no-kernel-cache` is an alias for memory mode. It does not disable in-process
throughput caching: doing that would cause repeated calibration. Shader mode
isolates reusable compiled code from timing-dependent tuning choices; it still
needs fresh tuning each process. Full mode is available to retain those choices.

Storage routing is configured once, before runtime/device initialization. Shader
mode also prevents runtime records from opening the database directly. The local
patch is described in vendor/cubecl-environment/UNIMSPS_PATCH.md.

The runtime store lives under `<kernel-cache-dir>/runtime`. Select the root with
`--kernel-cache-dir` or `UNIMSPS_CACHE_DIR`; otherwise it is beside the executable
in `.unimsps-kernels`. A build-time content hash covers source, manifests, lockfile,
vendored code, toolchain, features and relevant build settings. Moving/copying the
binary or rebuilding unchanged inputs preserves its namespace. Changed inputs
select a new namespace. Old caches are left intact; this revision starts a new one.

Changing six lights to nine does not erase six-light kernels. The same build's
store retains both, with CubeCL's kernel keys and device namespaces distinguishing
artifacts. Full mode additionally retains the runtime's shape-specific tuning.
Flash-attention failure decisions are rediscovered each process instead of loading
shape-only negative decisions from a different adapter.

The database probe reports durable storage availability, while namespace counts
report stored artifacts. Neither alone proves every kernel was a cache hit. The
DB is checkpointed after preparation and at normal exit. During execution
committed data can remain in its `-wal` sidecar; main-file size alone is insufficient.
Concurrent Windows instances should use separate writable cache directories.

## Startup and execution

The workload JSON is historical metadata, never permission to skip process-local
setup. Startup uses CubeCL DryRun to replay dense and sparse shapes without normal
kernel dispatch; missing autotune measurements still execute. Warmup synchronizes
before leaving DryRun. `--no-warmup` defers this setup to inference.

Persisted SPIR-V avoids frontend shader compilation, not all driver pipeline
creation, fusion setup or buffer allocation. DX12/Metal use the WGSL path and do
not receive Vulkan's SPIR-V persistence. Zero startup cost is not expected.

Inner attention chunks submit asynchronously. `--queue-depth 1` waits at complete
patch boundaries; zero removes those waits. There are no additional per-chunk
waits introduced by this revision.

`--pool-cache-mib` (default 1024) is an unused-reservation cleanup trigger at scale
and view boundaries, not a hard VRAM cap. View cleanup now runs after all view-local
GPU tensors have dropped, including error returns. Default `--view-cleanup threshold`
retains bounded reusable storage. `--view-cleanup always` synchronizes and purges
unused memory after each view for diagnosing retention, potentially increasing
allocation overhead. A threshold of zero disables threshold cleanup only.

## Next manual run

Build with `cargo build --release`, then run the usual inference command into a new
output directory. Default shader mode deliberately uses a fresh namespace for this
revision. Repeat with the same binary/root to inspect shader reuse. No need to
delete caches or change encoder/pixel batch sizes.

Metadata includes build identity, cache mode, framework versions, preparation
time, image count, cleanup policy and allocator usage after each view. Allocator
usage is application pool accounting, not total physical VRAM or Windows residency.
The `seconds` field and console time cover inference, cleanup, restoration and
image saving; preparation and input decoding remain outside it. Use normal runs
for throughput because `--verbose` adds synchronization.

No build, inference or performance test was run by the coding agent. These changes
address identified persistence/lifetime issues and incorporate upstream fixes;
whether they eliminate the intermittent stalls requires the user's next run.
