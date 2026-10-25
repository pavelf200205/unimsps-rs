# Shader persistence (v4)

The default is now `--cache-mode shaders`. Compiled Vulkan SPIR-V (or CUDA
artifacts on the optional CUDA backend) persists across processes. Autotune winners
and hardware-throughput measurements are computed anew per process and reused in
RAM. `--cache-mode memory`, also selected by `--no-kernel-cache`, keeps all runtime
caching in RAM. Neither mode repeatedly disables in-process reuse.

## What was corrected

CubeCL already has a process-local compiled-pipeline cache in front of its disk
artifact store. `Store::remove` consumes a cached artifact from RAM after loading;
it does not delete the durable entry. Its lazy loading and consumption semantics
are restored exactly to upstream. Forced eager ingestion, cloning retained shader
entries, the shared-connection rewrite, and writes that falsely returned Stored
have been removed. All actual shader cache misses may be saved, including those
first encountered during a view.

`autotune.disable_cache = true` is the official disk-only switch. In contrast,
`throughput.disable_cache = true` bypasses RAM reuse too. We leave that false and
route its namespace to MemoryStorage. Runtime records are disabled. The app no
longer sets NVIDIA/Mesa driver cache environment variables. Flash-attention
failure decisions are rediscovered in each process.

A shader artifact is not a live driver pipeline. Loading it still creates the
module/pipeline and process-local fusion state. Startup performs the existing real
warmup for every process; it no longer changes behavior based on a JSON marker or
freezes persistence. Workload markers are historical hints only. Shader-only mode
still pays for fresh tuning, so startup cannot be assumed to become instantaneous.

The build hash covers sources, dependency lockfile, manifests, toolchain and build
settings. It now selects `unimsps-v4-shaders-...` in `<cache-dir>/runtime`.
Old caches are left intact and are not loaded. Six- and nine-image shader variants
can coexist under the same build and device keys; image count is not a reason to
switch environments or discard previous variants.

This change intentionally keeps Burn 0.22.0 / CubeCL 0.11.0 / CubeK 0.3.0 and the
current inference kernels and scheduling. It targets cache semantics specifically.

## Why a local environment crate remains

The exact 0.11.0 source was checked against the installed registry package:

- StoreOptions::storage_with permits custom storage for stores an application
  constructs. It does not override the stores built inside CubeCL's runtime.
- CacheConfig chooses locations, with no memory or namespace-routing variant.
- ThroughputCache::new unconditionally selects environment storage when compiled
  with persistence. Disabling throughput caching does not offer memory-only reuse.
- Turso is opened with experimental_multiprocess_wal(true). Turso 0.8.3 WindowsIO
  inherits supports_shared_wal_coordination() == false. The local Windows-only
  compatibility adjustment is still needed for this installed stack.

Thus a local patch remains necessary for this shader-only policy on the pinned
stack. It is limited to startup storage routing, Windows WAL compatibility,
checkpoint exposure and diagnostic counters. Store itself is byte-for-byte
upstream. No global Cargo registry source is modified. See
vendor/cubecl-environment/UNIMSPS_PATCH.md for the exact patch boundary.

## Diagnostics and interpretation

Every view's metadata records cache/build identity, preparation time, shader DB
hits/misses/accepted writes/I/O duration during preparation and that view, live
allocator usage, inference time and output time. Shader DB hits mean bytes were
retrieved: they do not count live pipeline hits, certify successful decoding, or
measure GPU execution. A warm pipeline performs no DB read on every dispatch.
A cache read failure also counts as a miss and is logged separately as a warning.

With `--cache-diagnostics`, OUT/_runtime contains:
- preparation.json: startup duration, per-preparation and process-total counters.
- compilation.log: CubeCL's native basic compilation log on actual compilation.
- autotune.jsonl: CubeCL's native tuning decision records, including candidates.

These log sinks do not enable GPU profiling waits. Fresh tuning is expected on
every process. A repeated run should show shader DB hits; additional misses may
occur when tuning explores a previously unseen candidate. Timing differences
between cold and warm runs include tuning and driver setup, so they are not a
pure shader-compilation benchmark. The shader database should have zero autotune
and throughput entries. It may also contain the small persistence probe.

The Windows main DB may be small while committed data resides in its WAL.
Checkpointing runs after preparation and at exit. Do not delete a live WAL.
Concurrent Windows processes should use separate writable cache directories.

## Manual sequence

The coding agent did not compile, run tests or run inference. From unimsps-rs,
build once and then run four separate processes with the same executable and
cache root. Use 6 -> 6 -> 9 -> 6, so ordinary reuse is checked before changing
image count. Keep other GPU workloads quiet, leave --verbose and --trace-gpu off,
and keep the cache between all four runs. --trace-performance adds wall-clock
patch tracing without extra GPU waits.

```powershell
cargo build --release
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$common = @('--batch', '--model', '..\unimsps-weights\model_uncalibrated.safetensors', '--data', '..\data\own_objects_normals\horse_960\mvps_png_full', '--kernel-cache-dir', ".\experiments\cache_v4_$tag", '--cache-mode', 'shaders', '--cache-diagnostics', '--trace-performance')
.\target\release\unimsps.exe infer @common --nb_img 6 --out ".\experiments\v4_${tag}_6a"
.\target\release\unimsps.exe infer @common --nb_img 6 --out ".\experiments\v4_${tag}_6b"
.\target\release\unimsps.exe infer @common --nb_img 9 --out ".\experiments\v4_${tag}_9"
.\target\release\unimsps.exe infer @common --nb_img 6 --out ".\experiments\v4_${tag}_6c"
```

If a run slows down, preserve these directories and its console output. Do not
purge the cache: the bad state is valuable evidence. A memory-only control can
then be run with --cache-mode memory, using the same binary and new output path.
The correct control now preserves RAM throughput reuse.

## Documentation consulted

- https://tracel.ai/blog/release-0.22.0/ (environments, adaptive tuning, throughput)
- https://docs.rs/cubecl-server/latest/cubecl_server/compiler/index.html (two cache layers)
- https://docs.rs/cubecl-environment/latest/src/cubecl_environment/environment.rs.html (environment lifetime)
- Installed 0.11.0 API source: persistence/store.rs, persistence/storage.rs,
  persistence/root.rs, config/autotune.rs, throughput/cache.rs,
  throughput/benchmarker.rs, and cubecl-wgpu/src/backend/base.rs.

Version-specific installed source was used where online docs described a different
release. Cache isolation is implemented; eliminating the observed slowdown remains
to be verified by the user's runs.
