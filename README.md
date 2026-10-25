# Uni-MS-PS in Rust + Burn

Standalone uncalibrated inference port of Uni-MS-PS Turbo, upstream revision
`903abb24a8db3608026d13d204258721c9654ae9`.
Burn is pinned to **0.22.1**, with CubeCL **0.11.3** and CubeK **0.3.1**.
There are no runtime or source dependencies on the separate SuperNormal project.

The primary goal is fast inference across GPU manufacturers through WGPU
(Vulkan, DirectX 12 or Metal, depending on the platform). CUDA is optional;
performance work must benefit the portable path. Cross-vendor performance
and compatibility have not yet been validated for the latest source changes.

## Build manually

Build the default portable WGPU version from this directory:

```powershell
cargo build --release
```

`--api auto` always selects WGPU, even when CUDA support is compiled in.
The optional CUDA backend requires both building with `--features cuda` and
explicitly passing `--api cuda` before `infer`. It is not required for the
default build or for the shared CubeCL attention optimizations.

To reuse the existing shared build cache instead:

```powershell
cargo build --release --target-dir ../supernormal-rs/target
```

The latter puts the executable at `../supernormal-rs/target/release/unimsps.exe`.
A shared target directory is only a Cargo artifact cache; it does not couple the
projects or their Git repositories. The first rebuild after the performance
changes may update Cargo.lock for the flash-kernel dependencies and optional
CUDA backend. Subsequent builds can add `--locked`.
Windows builds retain the static C runtime configuration in `.cargo/config.toml`.

**The latest performance changes have not been built, run, or benchmarked.**
Build and execution are left to the user; no further parity testing is required.

## Inference

Use the executable from whichever target directory you selected:

```powershell
.\target\release\unimsps.exe infer --model ../unimsps-weights/model_uncalibrated.safetensors --data PATH_TO_IMAGES --out runs/object --images 12
```

- `--images N`, `--nb-img N`, and `--nb_img N` select the number of photographs.
  Omit the option or use `-1` for all images. Subsets use lexical filename order;
  the upstream Python tool chooses random subsets.
- Add `--batch` to process each immediate subdirectory as a separate object.
  Weights load once; each object gets its own output subfolder.
- PNG, JPEG, TIFF, grayscale, and 16-bit images are supported. All images must
  have equal dimensions. An optional `mask.png` defines foreground.
- Outputs are `normal.exr` (signed XYZ), `normal.png` (XYZ mapped to RGB),
  `normal.mat` (H x W x 3 array `Normal_est`), `normal.f32` (CHW little-endian
  float32), and `metadata.json`. No normal-axis flip is applied.

## Speed and memory controls

These global options go **before** `infer`:

| Option | Default | Purpose |
|---|---:|---|
| `--precision auto` | auto | FP16 on supporting GPUs; FP32 on CPU/other devices |
| `--precision fp32` | | Explicit full-precision execution |
| `--encoder-batch N` | 8 | Images processed together by the spatial encoder |
| `--pixel-batch N` | 5000 | Pixels processed together by light transformers |
| `--query-batch N` | 0 | Flash query limit; 0 processes all queries together |
| `--image-cache-mib N` | 512 | RGB input/pyramid cache budget |
| `--queue-depth N` | 2 | Chunk groups between GPU completion waits; 0 allows unrestricted submission |
| `--unfused-attention` | off | Explicit matmul/softmax spatial attention fallback |
| `--cpu` | off | Burn Flex CPU backend |
| `--integrated` | off | Select the first integrated GPU |
| `--api vulkan` / `--api dx12` / `--api metal` | auto | WGPU graphics API selection |
| `--api cuda` | off | Explicit optional CUDA backend; requires the cuda build feature |

For example:

```powershell
.\target\release\unimsps.exe --encoder-batch 3 --pixel-batch 5000 --image-cache-mib 256 infer --model ../unimsps-weights/model_uncalibrated.safetensors --data PATH_TO_IMAGES --out runs/object --images 12
```

Normals, interpolation, masking, and patch accumulation stay on the device until
one final normal-map readback. If the RGB pyramid fits its budget, it is built and
kept on the device. Otherwise the host pyramid uses parallel resizing and only
fitting scales are cached; larger scales stream RGB patches. This budget excludes
weights, normal maps, activation buffers, and backend workspace; it is not a cap
on total VRAM usage. Lower encoder/pixel batches to reduce activation memory.

Spatial attention attempts accelerated flash for eligible shapes. Unavailable
configurations are cached per device, dtype and shape, rather than disabling
acceleration globally. Tiled GEMM fallback uses power-of-two query chunks and
a 64 MiB per-score-tensor budget. This is not a bound on total GPU workspace.

Light attention over up to 32 images uses a fused subgroup kernel with FP32
online softmax, without cooperative-matrix requirements or a score buffer.
Subgroup width is read at runtime; the launch accounts for variable AMD/Intel
subgroup widths. CPU, larger light sets and devices without the required
subgroup features retain Burn implementations. K/V projections share one GEMM.
The executable reports when the fused light kernel runs.

Other Burn operators still use autotuning and first-use kernel compilation.
These source changes have not been built or timed.

See [PERFORMANCE.md](PERFORMANCE.md) for the source-level comparison with Turbo.
No speedup factor is claimed until the user measures the rebuilt executable.

## Bounded GPU submission

Normal inference now waits after groups of two attention or light-transformer
chunks and at encoder-batch/patch boundaries. This limits queued intermediate
work and lets the backend reuse temporary buffers without waiting after every
operator. It is a scheduling policy, not a hard VRAM limit. Use `--queue-depth 0`
to disable these waits, or a positive value to change the group size. Verbose
profiling already waits at finer boundaries and skips these redundant waits.

Chunk outputs are written into preallocated buffers instead of retaining all
outputs for a final concatenation. Partial light batches are padded by at most
255 independent pixels, within the requested pixel-batch limit, then trimmed.
This reduces mask-dependent execution shapes without changing cross-pixel math.
Devices reporting no cooperative-matrix configurations select bounded GEMM
attention directly, avoiding failed flash attempts and temporary unit-flash runs.

These changes address the supplied profile; improved throughput is unverified.

## Verbose profiling

Add `--verbose` (or `-v`) to an inference command. It can appear before or after
`infer`. This enables detailed synchronized wall-clock profiling on WGPU as well
as the other backends; no CUDA profiling tools are required.

The console reports model loading, input decoding, pyramid preparation, scales,
processed patches, final readback and output writing. At completion it prints
the 40 largest exclusive costs. The full `profile.json` in the output directory
contains every recorded operation/shape, call counts, inclusive and exclusive
times, first-call time, minimum, maximum and mean, device/settings, synchronization
errors and actual custom-kernel route counts. Timed operations include linear
projections, normalization, convolutions, spatial/light attention, masked set
transformers, decoder stages, uploads, resizing and patch blending.

These are synchronized CPU+GPU wall times, not GPU timestamp measurements.
Profiling forces GPU completion at boundaries and changes fusion and overlap,
so compare overall throughput with verbose mode disabled. First-use compilation
and autotuning are included; the first-call column helps identify startup cost.
Inclusive rows overlap; do not sum them. Exclusive time excludes nested timed
operations but includes dispatch and profiling overhead. A failed synchronization
marks the JSON report invalid. In batch mode each object has its own report;
model loading is included only in the first report because weights load once.

## Weights and behavior

Use the already-converted `model_uncalibrated.safetensors`. Direct `.pth` loading
is deliberately not added: it needs PyTorch pickle/storage deserialization or an
additional import subsystem. Python is needed only for the existing one-time
`tools/export_weights.py` converter, not for inference.

The loader checks all 1,292 tensors (79,151,174 parameters) for names, shapes,
dtypes and finite values. Linear weights are transposed/packed once at load time.
FP16 mode keeps layer-normalization computation and normal-vector reductions in
FP32. The port preserves BGR input order, masked mean normalization, the upstream
exclusive-maximum crop, its patch mask combination, and Gaussian overlap blending.
The supplied uncalibrated architecture is supported; calibrated inference and
training are not implemented.

## Repository and attribution

This directory has its own Git repository and is ignored by the parent repository.
Build outputs, experiment files, results, and weights are excluded by its own
`.gitignore`. The reference Python checkout and supplied weights remain outside
this source repository in the current directory layout.

Original authors: Clément Hardy, Yvain Quéau, David Tschumperlé (GREYC).
The translated implementation retains upstream's CeCILL 2.1 license in `LICENSE`.
No separate weight license was included with the supplied checkpoint/repository.

## Native GEMM follow-up: row softmax and allocator diagnostics

The default GEMM attention path now uses a dedicated row-softmax kernel between
its two native matrix multiplications. Supported WGPU devices use subgroup
max/sum with FP32 intermediates and a single output rounding. The score buffer
is reused only when exclusively owned and contiguous. This retains the existing
GEMM chunk budget, asynchronous submission, packed projections, custom LayerNorm
and light-attention kernels. No CUDA or cooperative matrices are required.

Use `--burn-softmax` to retain the previous general Burn softmax. Devices without
the required subgroup capabilities, wider rows or excessive per-thread register
requirements automatically retain that implementation. Latest source changes
have not been compiled or timed here. FP32 reductions may change FP16 rounding.

`--pool-cache-mib N` controls the allocator reservation threshold that triggers
cleanup. The default is 1024 MiB of unused reservation. Larger values can retain
more reusable buffers at the cost of higher reserved VRAM; 0 disables these
explicit cleanup calls. This is a cleanup trigger, not a hard VRAM limit.
The check also runs after view-local tensors have dropped. `--view-cleanup always`
instead purges unused buffers after every view, at the cost of allocation reuse.

Verbose profiling now separates QK GEMM, softmax and AV GEMM, records softmax
in-place reuse, and measures allocator check/cleanup time. The saved native-GEMM profile in experiments/unimsps_run11 combines both GEMMs
and softmax and includes cold-start overhead. Later runs have metadata only.
These records cannot isolate softmax or allocator costs in the 50–55-second build.

### Encoder performance controls

The default encoder uses native biased linear operations and direct token-layout
depthwise convolution fused with bias and erf GELU. Use --reference-encoder to
select the previous projection and depthwise path, and --burn-softmax to select
the previous softmax independently. LayerNorm also conditionally reuses owned
contiguous buffers. All changes are portable WGPU operations; no CUDA is required.
These source changes have not been compiled or timed here.

For a manually initiated comparison, tools/compare_performance.ps1 accepts
-Exe, -Model, -Data (a batch input directory), -Out (a new results directory),
-Images (default 6), and -Repeats (default 3). It runs four configurations in
alternating order without verbose profiling and writes timings.csv. Each run
includes normal warmup; use per-view medians and inspect first-view timings
separately. The comparison is substantial work and runs only when you launch it.

### Convolution dispatch

Spatial reductions and stride-2 patch embeddings default to bounded patch
packing plus native GEMM, producing token-layout outputs directly. Packed
weights are prepared once during load. --burn-convolutions restores Burn's
convolution dispatch for these layers and skips the extra packed weights.
Use normal runs to compare throughput and --verbose separately to inspect
conv-pack and conv-matmul costs. This source change has not been built or timed.

## Persistent preparation update

See [CACHE_AND_SCHEDULING.md](CACHE_AND_SCHEDULING.md) for the current cache,
compile-only startup replay, sparse remapping and queue/memory policies. This
supersedes earlier claims that a workload marker can skip all warmup. A rebuild
is required to enable the Vulkan compiler and runtime persistence features.
