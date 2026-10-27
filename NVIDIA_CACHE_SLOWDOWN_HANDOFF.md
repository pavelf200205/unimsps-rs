# UniMSPS Rust: repeated-run NVIDIA driver-cache slowdown

Prepared 2026-10-27, Europe/Kiev. This document is a self-contained investigation handoff, not a confirmed root-cause report.

**Subsequent implementation:** the repository now contains the supervisor-based
immutable-seed strategy in `src/driver_cache.rs`, seeded random image sampling
(`--seed`, default 42), and the missing vendored API documentation. See README
for current usage. The remainder of this handoff preserves the investigation
state before those changes; its “not implemented” statements are historical.
No compilation or inference validation of the new implementation was run by
the agent. Later cache restoration experiments are recorded in
`experiments/cache_theory_findings.md` and `cache_theory_binary_analysis.json`.

## 1. Objective and constraints

Resolve intermittent, severe inference slowdowns while retaining fast startup and cross-vendor GPU support. The application rewrites Uni-MS-PS Turbo inference in Rust/Burn. Vulkan support for non-NVIDIA GPUs is a central project requirement; switching the application to CUDA is not an acceptable general solution.

User instructions: **do not initiate compilation, cargo checks/tests, inference runs, or numerical parity tests.** The user builds and benchmarks. Source inspection and analysis are allowed. Preserve existing binaries, caches, and experiment evidence. Do not reset Git history or overwrite a run directory. The user has spent considerable time on unsuccessful scheduling/cache experiments; prefer narrowly discriminating experiments over speculative changes.

The immediate request that produced this document was documentation only. No read-only-cache implementation, strict tuning mode, new GPU instrumentation, permission changes, or driver changes were made during this investigation.

## 2. Environment and relevant locations

- Workspace: `E:\SuperNormal_Rust_Burn_Framework_Rewrite`
- Independent repository: `E:\SuperNormal_Rust_Burn_Framework_Rewrite\unimsps-rs`
- HEAD, verified when writing: `ae771b2ffb336b2be0d475146b188ec52f4a2bed` (`v8`). Working tree was clean before adding this document.
- GPU: NVIDIA GeForce RTX 3060; Windows; PowerShell.
- Frameworks: Burn `0.22.0`, CubeCL `0.11.0`, CubeK `0.3.0`.
- Active GPU path: WGPU/Vulkan, F16, native GEMM attention with custom portable kernels for other operations.
- Experiments: `unimsps-rs\experiments`
- Binaries and test caches: `unimsps-rs\target\release`
- Model: `unimsps-weights\model_uncalibrated.safetensors`
- Dataset: `data\own_objects_normals\horse_960\mvps_png_full`
- Each view is nominally 960x960; processing pads/crops into scales, often 1024 or 512. Compare like resolutions. Foreground masks and patch counts vary by view.
- Common settings: encoder batch automatically equals 6 or 9 here; pixel batch 65536; queue depth 1; pool cache 1024 MiB; image cache 512 MiB; score budget 256 MiB; PNG output.
- Exact currently installed NVIDIA driver version was not verified in this investigation. Do not infer it from linked reports or release notes.

Relevant code:

- `src/cache.rs`: CubeCL persistence configuration, NVIDIA/Mesa environment variables, manifest and database reporting.
- `src/main.rs`: workload signature, warmup selection, write-freeze policy, metadata.
- `src/model.rs`: network and warmup workload.
- `src/pipeline.rs`: multiscale/patch inference.
- `src/execution.rs`: submission, synchronization and pool cleanup.
- `src/trace.rs`: optional wall/GPU tracing.
- `vendor/cubecl-environment/src/persistence/turso.rs`: local database modifications, including write suppression.
- `vendor/cubecl-environment/UNIMSPS_PATCH.md`: patch description; inspect actual code too, because the description does not enumerate every added behavior.

## 3. Executive finding

The strongest recent evidence is a controlled sequence where the **same CubeCL database** gives fast inference with a reset NVIDIA GLCache, then slow inference after repeated reuse of that driver cache, then fast inference again after resetting only that driver cache.

Switching between six and nine images is **not necessary** to reproduce the failure. During a nine-image run using a separate cache root, the six-image driver-cache files remained byte-for-byte unchanged. Direct cross-directory file contamination was therefore not observed in that experiment.

This strongly implicates the NVIDIA driver-cache reuse path in the current configuration. It does **not** establish malformed/corrupt cache contents, a particular NVIDIA bug, or a universal two-run failure threshold. A reset also changes preparation duration and driver compilation behavior; these remain experimental confounders.

## 4. Terminology: distinct kinds of caching

Do not conflate these layers:

1. **CubeCL compiled-artifact cache**: includes SPIR-V artifacts in the runtime database. SPIR-V is not the final GPU machine code.
2. **CubeCL autotuning decisions**: selected operation implementations for tuning keys.
3. **CubeCL throughput measurements**: measurements used by tuning logic.
4. **In-process pipelines and runtime caches**: GPU objects recreated for each process, plus RAM caches.
5. **NVIDIA driver disk cache**: `.bin`/`.toc` files under the redirected `nvidia_gl_cache/GLCache/...` directory.
6. **Manifest `kernel_state.json`**: historical workload signatures and unavailable flash-attention shapes. This is not a complete inventory of compiled pipelines or proof that every future tuning key is covered.

Disabling CubeCL persistence does not necessarily disable the driver's independent persistence. Conversely, freezing the CubeCL database does not freeze driver-cache files.

## 5. Current implementation at ae771b2

Persistence was restored by another model after an earlier removal commit. This restoration also changed warmup and patch preparation, so comparisons across these commits are not pure persistence-only comparisons.

### Runtime configuration

When caching is enabled, `src/cache.rs` sets:

```text
compilation.cache = true
autotune.disable_cache = false
throughput.disable_cache = false
environment.path = <cache_root>/runtime
environment.name = unimsps-v4
```

It attempts to configure these environment variables **only if they are absent**:

```text
__GL_SHADER_DISK_CACHE=1
__GL_SHADER_DISK_CACHE_PATH=<cache_root>/nvidia_gl_cache
__GL_SHADER_DISK_CACHE_SIZE=2147483648
__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1
CUDA_CACHE_DISABLE=0
CUDA_CACHE_PATH=<cache_root>/nvidia_compute_cache
CUDA_CACHE_MAXSIZE=2147483648
MESA_SHADER_CACHE_DIR=<cache_root>/mesa_shader_cache
MESA_SHADER_CACHE_MAX_SIZE=2G
```

Inherited variables can therefore override the intended isolation. Log effective values in any future diagnostic implementation. Nevertheless, actual NVIDIA cache files in the requested roots demonstrate that redirection took effect in the observed tests. Do not dismiss this solely because many public descriptions of these variables concern Linux.

The forced `SKIP_CLEANUP=1` is a nonessential override worth testing independently. There is no evidence yet that it causes this failure. The observed caches are only several MB, far below the configured 2 GiB limit; ordinary capacity eviction is not a convincing explanation on size alone.

### Warmup and write freeze

- A new signature executes warmup with durable writes enabled, then freezes autotune/throughput writes while allowing other artifacts to be saved.
- A previously warmed signature freezes durable database writes before replaying warmup.
- The vendored `Database::insert`, `replace`, and `insert_many` return success-like results when writes are frozen, without performing those writes.
- This suppresses persistence, **not computation**. A tuning miss can still trigger benchmarking and compile candidate kernels; its durable result may then be suppressed.
- Warmup purges temporary pools after stages and sparse buckets.
- Patch preparation now keeps RGB and prior normals separately rather than concatenating repeated priors over the full frame.

The vendored environment crate also works around unsupported experimental multiprocess WAL on Windows and exposes checkpointing. These are separate from the added freeze policy. Concurrent writable instances should use separate cache roots.

### Actual CubeCL tuning semantics, checked in local 0.11.0 source

Registry root:
`C:\Users\Pavel\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f`

Relevant files:

- `cubecl-runtime-0.11.0/src/config/autotune.rs`
- `cubecl-runtime-0.11.0/src/tune/local.rs`
- `cubecl-runtime-0.11.0/src/tune/tuner.rs`
- `cubecl-runtime-0.11.0/src/tune/tune_cache.rs`

`LocalTuner` executes a valid cached selection directly. On a miss, the tuner synchronizes persistent state, validates unchecked entries, and ultimately starts tuning if no valid result is available. Configuration levels are minimal/balanced/extensive/full; there is no strict cache-only level in the inspected configuration.

`autotune.disable_cache` disables persistent decision reuse, **not autotuning**; the RAM cache remains. Setting it to true is not a solution to “never tune during inference.”

A strict inference mode would need to reject/report missing or invalid tuning decisions before benchmarking, or route them through an explicitly chosen non-tuned fallback. It should not silently choose an arbitrary cached kernel for a different shape. Throughput probes and pipeline compilation need separate consideration too.

### Device fingerprint: not a workload key

Both manifests contain `spirv_4318_9476`. This is expected. In `cubecl-wgpu-0.11.0/src/runtime.rs`:

```rust
let fingerprint = format!("spirv_{}_{}", adapter_info.vendor, adapter_info.device);
```

The SPIR-V store uses the same vendor/device pair in `compute/server.rs`. The value identifies hardware/backend compatibility, not the number of lights.

The complete manifest signatures already differ, including `eb=6|lights=6` versus `eb=9|lights=9`. Other fields include device identity, API, dtype, pixel/query batch, score budget, attention/softmax/encoder/convolution options.

Changing this hardware fingerprint or editing it in JSON does not control NVIDIA's internal cache identity. Use a separate workload identifier for directory organization if needed.

## 6. Recent measured sequence: C through H

All folder names below are under `unimsps-rs/experiments/` and begin `inference_horse_960_ae771b2ffb33_`.

The six-image run C folder uses a **Cyrillic capital С**: `6imgs_runС`. This is easy to miss when looking up files.

| Suffix | Cache/use | Completed views | View 00 seconds | Mean 1024-view seconds | Preparation seconds |
|---|---|---:|---:|---:|---:|
| `6imgs_runС` | fresh `cache_6` | 20 | 15.24 | 14.72 | 130.33 |
| `6imgs_runD` | reused `cache_6` | 20 | 15.50 | 15.21 | 1.13 |
| `9imgs_runB` | fresh `cache_9` | 20 | 23.90 | 22.99 | 143.32 |
| `6imgs_runE` | reused `cache_6` after nine-image run | 1 | 39.36 | 39.36 | 1.15 |
| `6imgs_runF` | fresh `cache_6_new` | 20 | 15.38 | 14.83 | 129.54 |
| `6imgs_runG` | copied CubeCL cache; reset NVIDIA cache | 3 | 15.14 | about 14.88 | 29.70 |
| `6imgs_runH` | original preserved `cache_6` | 7 | 15.94 | compare individual views | 1.16 |

G view times: 15.14, 14.73, 14.76 s.

H view times: 15.94, **51.26, 63.17, 45.64**, 18.99, 12.61, 3.73 s. Later small views are different resolutions; do not average them directly against 1024 views.

Reset procedure: copy `cache_6` to `cache_6_driver_reset`; rename only its `nvidia_gl_cache` to `nvidia_gl_cache_saved`. Retain `runtime` and `kernel_state.json`.

Verified after G/H: the two runtime database files were byte-identical:

```text
cache_6/runtime/unimsps-v4.db
cache_6_driver_reset/runtime/unimsps-v4.db
SHA256: 3DE5736D6EDAD102F5ACBA5305EDD3FA35FBC93EF58D4B8F6C8E7D6225DE8295
```

This was a point-in-time comparison, not a claim that files can never change later. The old runtime DB modification time remained at the initial warmup; its observed WAL was only a 32-byte header.

Six-image C/D/E/F view_00 metadata reported exactly the same pool totals:

```text
allocations: 1714
in_use_bytes: 176779104
padding_bytes: 9632
reserved_bytes: 830549952
```

This argues against accumulating CubeCL pool allocations as the explanation for those runs, but does not measure driver-owned memory or transient peaks.

## 7. Strongest experiment: I through L and file hashes

The user backed up the good reset cache as `cache_6_driver_reset_backup`.

Active driver files have the common relative path:

```text
nvidia_gl_cache/GLCache/
  cc7565cb71c89e2a4a7ab8ddbd7aca10/
  c26305090aa51f1c/
  f0aea9799da350c4.bin
  f0aea9799da350c4.toc
```

Matching basenames under separate cache roots are not themselves evidence of a collision.

### User-reported MD5/size sequence

| Checkpoint | BIN bytes | BIN MD5 | TOC bytes | TOC MD5 |
|---|---:|---|---:|---|
| Good reset cache before nine-image probe | 3,529,411 | `839F3AEDD16341BDD60E8C3B7D459358` | 10,040 | `3F7EB9FA30CCD872AC5677896C6C9692` |
| Immediately after nine-image probe, before six-image launch | 3,529,411 | `839F3AEDD16341BDD60E8C3B7D459358` | 10,040 | `3F7EB9FA30CCD872AC5677896C6C9692` |
| After six-image I, fast | 8,690,687 | `41CD2662299CC50DC446CCDF08AC3288` | 36,224 | `9A5ED376DA873615E648F5B7E96278FD` |
| After six-image J, fast | 12,259,457 | `0D9636DF0EDCE9B7920B6770C025B4C2` | 36,224 | `9A5ED376DA873615E648F5B7E96278FD` |
| After six-image K, slow, stopped after view_02 | 12,324,289 | `8F61F29C7D26FF7D6D97A326B2751F11` | 36,224 | `9A5ED376DA873615E648F5B7E96278FD` |
| After NVIDIA-only reset and six-image L, fast | 7,498,343 | `390CEC35DC0C98D0527DFC3F8C01FA5F` | 31,304 | `9B8D90A57653C30F754B458405052846` |

The nine-image probe was `9imgs_runC`, using a new `cache_9_probe`. The six-image runs I/J/K reused `cache_6_driver_reset`. After K, the user renamed its driver directory to `nvidia_gl_cache_after_a_couple_of_runs`; L recreated the active `nvidia_gl_cache`.

### Independently inspected metadata

| Run suffix | Completed views | First three view times (seconds) | Mean 1024-view seconds | Preparation seconds |
|---|---:|---|---:|---:|
| `9imgs_runC` | 20 | 23.1495, 22.8636, 22.8196 | 22.7318 | 135.6100 |
| `6imgs_runI` | 20 | 14.9851, 15.0414, 14.7177 | 14.9167 | 1.1375 |
| `6imgs_runJ` | 20 | 16.7802, 16.6161, 14.8347 | 15.3197 | 1.1428 |
| `6imgs_runK` | 3 | **28.9477, 50.9734, 37.9420** | **39.2877** | 1.1412 |
| `6imgs_runL` | 20 | 14.9994, 14.9907, 14.7250 | 14.9750 | 28.2616 |

These recent run directories contain metadata but do not provide a new per-kernel trace establishing exactly which operation slows down. Do not infer new tuning events just from file growth.

### What this supports

- Nine-image execution did not directly modify the watched six-image files.
- Repeated six-image launches alone can reproduce the problem.
- Retaining CubeCL state while resetting NVIDIA driver-cache state restores fast inference in repeated observations.
- Driver-cache growth is associated with reuse, but neither size nor unchanged TOC proves corruption.
- Fast initial compilation versus fast loading may affect driver behavior differently; read-only reuse must still be tested.

### Why a reset driver cache may initially be smaller

Hypothesis: cold CubeCL autotuning compiles multiple candidate kernels, whereas retaining selected tuning decisions and resetting only GLCache compiles fewer selected kernels. Initial good caches can therefore differ greatly in size without any cross-workload contamination.

Earlier observed BIN sizes included 16,806,973 bytes for slow `cache_6`, 16,571,626 bytes for fast `cache_9`, and 11,792,461 bytes for fast `cache_6_new`. There is no demonstrated universal size threshold.

## 8. Earlier history: important context, weaker attribution

- `0866833bb432b37763b3f23b8b3442ed7d0ae1a2` was an early fast baseline, but user reruns still showed some stutters. An old binary named `unimsps_kernel_reuse_buggy.exe` was often fast; its exact source revision was not initially certain.
- `6219ce46ef0c79d6766501d5c96e112f67ba0d92` was discussed as a scheduling/stuttering-related commit.
- `853268e` pursued shader-only persistence; `91de7d4` added revision-labelled binaries and evidence documentation.
- `2dda784be835` removed application cross-run persistence and vendored environment code. It forced a process-local storage fallback while retaining in-process reuse. This still did not explicitly disable NVIDIA driver caching.
- `ae771b2ffb33` restored persistence and the vendor patch, and modified warmup/patch memory handling.

Good older memory-mode controls existed for both 6 and 9 images. Later memory-only runs also slowed down. Therefore “application SQLite persistence is necessary for failure” is false. Driver persistence remaining enabled is a plausible explanation, but the older runs did not directly prove which driver cache was accessed.

Encoder-batch 9 / pool-cache 2048 were already explored and were not the solution. Avoid recycling these as the principal hypothesis.

## 9. Earlier monitoring and GPUView findings

These measurements concern older `2dda784` runs, not the new I–L sequence. They constrain some theories but must not be presented as direct measurements of the latest failure.

### GPU monitoring

In `inference_horse_960_2dda784be835_9imgs_runE`, slow and fast patch intervals had similar core clocks (~1945–1948 MHz), memory clock 7501 MHz and P0 state. GPU temperatures were low (~57–60 C averages; maximum ~62.5 C). GPU utilization stayed high. Slow intervals drew lower average power (~124.5 W versus ~156.7 W fast).

This does not support simple thermal/clock throttling in that capture. High utilization is not proof of productive occupancy or exclusive ownership.

### GPUView run G

Directory: `inference_horse_960_2dda784be835_9imgs_runG`.
Capture: `Merged.etl`, about 637 MB, ~107.9 seconds, started mid-view_00 and ended after view_01. No lost events/buffers reported.

Saved analysis: `analysis_summary.json`; selected events: `gpu_selected.csv`; streaming extraction helper: `extract_gpu_events.cjs`.

- View_00: 49.4549 s; view_01: 53.9104 s.
- Decoded main-engine context running intervals overwhelmingly belonged to UniMSPS (~99.8% of the decoded running-time sum).
- Important caveat: hardware scheduling buffers can include pre-capture history. Raw GPU timestamps are not UTC. Do not call this 99.8% of the entire capture's wall time.
- CPU was ~69.6% idle in aggregate; this does not exclude a single critical thread's starvation.
- 3,258 matched normal render-queue packets had no reported preemption or timeout flags.
- One slow patch submitted its packet group within ~40 ms but the last completion arrived ~3.34 s later. This argues against CPU preparation alone explaining that patch.
- Queue submission-to-completion latency includes backlog and is not isolated shader execution time.
- Process GPU budget did not drop in the observed events. No claim was established that all paging was absent.
- Most accumulated slowdown appeared in completion waits. Per-shader GPU timings were not yet collected.

The ETL was decoded with CLI tools; GPUView GUI inspection was unsuccessful. Do not claim a visual timeline diagnosis.

### HAGS-off test

Directory: `inference_horse_960_2dda784be835_9imgs_hags_off_A`.
Saved comparison: `hags_comparison.json`.

- 20 views: 896.2544 s inference versus older fast control 378.8626 s (different commit, not a controlled HAGS-only comparison).
- Same-binary first two views: HAGS off 100.7277 s versus prior HAGS-on run G 103.3654 s. No meaningful recovery.
- Median matched patch ratio versus older fast control ~1.012, but 95th percentile ~6.017: severe slow tail rather than uniform slowdown.
- 215/658 patches exceeded one second; 183 exceeded two seconds.
- End-of-view pool totals matched the fast reference for corresponding resolutions; cleanup itself accounted for only ~1.61 s.

HAGS off did not solve the failure. Do not prescribe another HAGS sweep without new evidence.

## 10. Research findings and source limitations

### A. NVIDIA documents read-only shader caches

[NVIDIA Jetson Linux graphics guide](https://docs.nvidia.com/jetson/l4t/Tegra%20Linux%20Driver%20Package%20Development%20Guide/graphics_binary_shaders.html)

The guide explicitly describes generating a cache on a writable system, copying it, and giving the application read-only directory access. Missing shaders may still compile at runtime, but cannot be saved there. It also notes cache search/maintenance overhead and internally generated shaders for some operations.

**Scope:** Jetson/OpenGL ES, not a guarantee of GeForce Windows Vulkan behavior. This finding corrected an earlier overly cautious assessment that read-only caches had no documented workflow. No public Windows-specific “read existing GLCache, never update it” environment switch was established.

[NVIDIA DRIVE automatic shader cache](https://developer.nvidia.com/docs/drive/drive-os/6.0.9.1/public/drive-os-linux-sdk/common/topics/graphics_content/AutomaticShaderCache8.html)

Explains enable/path controls and driver/GPU/application-specific hashed directories. Again, embedded/Linux documentation, not proof of every desktop implementation detail.

### B. Cache growth is not intrinsically corruption

[NVIDIA engineer response about increased cache size](https://forums.developer.nvidia.com/t/455-23-04-driver-produces-very-large-shader-caches-results-in-constant-cache-invalidation/156087/9)

The response explains that additional shader information was intentionally stored to improve runtime performance. This historical Linux report does not explain our unchanged TOC or prove our cache healthy; it simply prevents inferring corruption from growth alone.

### C. Related bugs, not exact matches

[Repeated Vulkan shader-cache cleanup/recompilation](https://forums.developer.nvidia.com/t/pre-compiled-shader-cache-cleaned-by-nvidia-driver-causing-stutter-when-launching-vulkan-apps/336196)

Linux/Steam reports reference NVIDIA tracking IDs 4932793 and 4934720. Large cache capacity, cleanup and environment propagation differ from our small-cache Windows inference case. Do not apply its system-wide environment workaround without justification.

[Windows Vulkan cached-pipeline device-loss reproduction](https://forums.developer.nvidia.com/t/vulkan-windows-617-42-621-01-rtx-5070-ti-device-lost-dma-page-fault-reusing-command-buffers-after-new-ones-are-submitted-pipeline-from-shade/386503)

User report dated October 21, 2026: first launch succeeds, later driver-cache launches trigger device loss with command-buffer reuse. Different GPU, drivers, graphics workload and symptom. No evidence it is the same bug; it demonstrates why cached-versus-fresh behavior deserves investigation. Its reported environment-variable disable attempt did not resolve its issue, so do not assume an environment variable worked solely because it was set.

[NVIDIA Vulkan driver release notes](https://developer.nvidia.com/vulkan-driver)

The page inspected on October 27 listed October 22, 2026 Windows developer beta 597.33 fixes concerning shader-cache key reliability and multithreaded compilation stability. The notes do not tie those fixes to our issue. The installed driver was not checked, and no driver update was recommended as a guaranteed fix.

### D. Portable explicit Vulkan controls

[VK_EXT_pipeline_creation_cache_control](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_pipeline_creation_cache_control.html)

`VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT` prevents pipeline/shader compilation during that pipeline-creation request; the implementation returns `VK_PIPELINE_COMPILE_REQUIRED` when compilation is needed. This is not a blanket prohibition on all driver disk writes or internal maintenance. Capability checking and backend integration are required.

[Khronos pipeline cache guide](https://docs.vulkan.org/guide/latest/pipeline_cache.html)

Application-owned `VkPipelineCache` data can be saved and supplied on future launches. This is distinct from NVIDIA's automatic disk cache. Discarding changes to an application-owned cache does not prove that automatic driver caching has been disabled.

[Khronos explicit pipeline binary sample](https://docs.vulkan.org/samples/latest/samples/extensions/pipeline_binary/README.html)

`VK_KHR_pipeline_binary` offers explicit binary capture/reuse where supported. Binaries remain driver/device-specific; this is a longer-term backend project, not an existing application flag or guaranteed workaround.

[WGPU PipelineCache documentation](https://wgpu.rs/doc/wgpu/struct.PipelineCache.html)

Documents that desktop drivers commonly manage their own caches. In the inspected CubeCL WGPU backend, pipeline creation uses `cache: None`; do not mistake that for disabling NVIDIA's own disk cache.

## 11. Proposed resolution paths, not yet implemented

### First: test a filesystem-protected good driver-cache copy

- Preserve all original evidence.
- Use a dedicated copy of a known-good snapshot, keeping the CubeCL runtime database separate.
- Apply actual filesystem access permissions if testing read-only behavior. The Windows folder “Read-only” checkbox is not a reliable write-access policy.
- Protect against creation/replacement as well as modification; avoid broad ACL changes outside the dedicated test copy.
- Verify both preparation time and inference speed, and compare hashes afterward.
- Unchanged hashes alone are insufficient: the driver might ignore the protected cache and compile from scratch or use another location.
- A read-only cache may still permit runtime compilation into memory. This experiment tests whether preventing persistent evolution helps, not whether every driver compilation is eliminated.

### Second: strict tuning-cache-only mode and diagnostics

- Separate preparation from inference policy.
- During preparation allow required tuning and artifact generation.
- In strict inference, require valid tuning hits and explicitly report missing/invalidated keys instead of silently benchmarking.
- Log tuning misses, selected implementations and actual compilation events with low overhead.
- Do not infer coverage from a nonempty GLCache or the current manifest's warmed flag.
- Sparse mask buckets, changed dimensions/options and other workload variations can introduce unseen keys.
- Verify whether remaining throughput probes or compilation paths need their own strict controls.
- Freezing database writes alone is not this feature.

### Fallback: immutable seed plus disposable writable driver cache

Copy a known-good seed to a fresh per-process driver-cache directory, allow writes there, and never automatically replace the seed with the modified result. This isolates between-run accumulation while avoiding write failures in the driver. It does not prevent an in-process failure developing during a long run and must be tested. No implementation exists yet.

### Simplest reliability baseline

Use a fresh driver-cache directory per launch while retaining CubeCL decisions. Observed recovery cost: ~28–30 seconds preparation rather than ~1 second, but far less than full cold tuning (~130 seconds). This is an observed fallback, not the desired final startup behavior.

### Workload directories

Separate validated snapshots for different image counts/options are sensible for coverage and reproducibility. Include GPU, driver, kernel/application version, precision and relevant workload options. Separation alone is not the fix: six-image-only reuse already fails. Do not change the device fingerprint to manufacture workload identities.

## 12. Questions still unanswered

1. Are there any CubeCL tuning misses during slow inference, despite the warmed manifest and frozen DB?
2. Does Windows Vulkan consume a protected GLCache efficiently, or ignore/fail it?
3. Does the slowdown persist with a byte-identical immutable seed at every launch?
4. Which shader/operation actually slows down, and is it compilation, GPU execution, resource residency, or synchronization?
5. Why does BIN grow while TOC remains byte-identical? This needs format-aware analysis or driver file-access diagnostics, not assumptions.
6. Does removal of the forced `SKIP_CLEANUP` override change the behavior?
7. Is the current driver affected by a known fixed issue? Establish the installed version before considering a controlled driver comparison.
8. Can a small portable compute reproduction demonstrate the issue for an NVIDIA bug report?

## 13. Practical reproduction notes

Typical command, run by the user from `unimsps-rs\target\release` (choose a genuinely new output folder and deliberate cache root):

```powershell
.\unimsps_ae771b2ffb33.exe infer --batch `
  --model ..\..\..\unimsps-weights\model_uncalibrated.safetensors `
  --data ..\..\..\data\own_objects_normals\horse_960\mvps_png_full `
  --out ..\..\experiments\NEW_UNIQUE_RUN_NAME `
  --nb_img 6 --formats png `
  --kernel-cache-dir .\DELIBERATELY_SELECTED_CACHE
```

Do not silently add `--verbose` or `--trace-gpu`: they change synchronization/timing behavior. Wall tracing and GPU tracing are distinct. Record exact flags when adding diagnostics.

For a compact read-only file snapshot:

```powershell
Get-ChildItem .\cache_6_driver_reset\nvidia_gl_cache -Recurse -File |
    ForEach-Object {
        [pscustomobject]@{
            File = $_.FullName
            Bytes = $_.Length
            SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    } | Format-List
```

The strongest evidence to preserve is the good backup, the accumulated bad driver cache, both BIN and TOC together, the corresponding CubeCL DB/manifest, and per-view metadata. Do not overwrite or “repair” these originals during investigation.

## 14. Suggested request to the receiving model

Review this evidence and the current source before proposing changes. Determine how to separate strict cached kernel selection from driver-cache write policy. Prefer a reversible test of a known-good protected driver cache and instrumentation proving whether new tuning/compilation occurs. Do not compile or run inference; the user will do so. Avoid claiming an exact known NVIDIA bug without a matching reproduction. Keep the solution portable, with any NVIDIA workaround isolated from other backends.
