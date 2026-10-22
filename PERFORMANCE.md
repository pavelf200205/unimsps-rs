# Source comparison with Uni-MS-PS Turbo

This is source inspection, not profiling. No commands, builds, tests or benchmarks
were run for this revision; actual throughput remains unverified.

## Dispatch changes

- Turbo calls SDPA on the whole spatial query sequence. The previous Rust path
  applied a quadratic score-memory limit even to flash attention. At 256 x 256,
  8 images, 1 head and 1024 keys, this split 65536 queries into chunks of 4096:
  16 separate dispatches per encoder block. Flash now defaults to one operation.
- Burn 0.22 default CubeCL attention invokes an autotuner. Its strategy wrapper
  can silently materialize quadratic scores on an invalid flash configuration.
  The new fusion-compatible custom operation calls the flash launcher directly:
  accelerated flash first, then unit flash if unsupported. Neither uses a full
  score matrix. It prints the selected path and does no attention benchmarking.
- Explicit positive query limits still work. CPU and --unfused-attention keep
  score tensors bounded to roughly 128 MiB, including with --query-batch 0.
  Unsupported flash configurations report an error instead of silently falling
  back to quadratic allocation.
- Turbo uses CUDA; the previous Rust binary offered only WGPU and CPU. The cuda
  Cargo feature enables an optional native Burn CUDA backend, selected only
  with --api cuda. WGPU is the default in every build and the primary target
  for performance work across manufacturers. Switching to CUDA does not resolve
  the portable performance goal. The direct flash operation also runs on WGPU.
- Each RGB stack now has one transfer in the chosen precision, replacing an
  FP32 upload, GPU cast and slice write for every photograph. Linear weights
  are packed and converted on the CPU before upload as well.

## Retained behavior

Encoder batches default to 8 images and light-transformer batches to 5000 pixels.
Linear projections use flattened GEMMs. FP16 activations retain FP32 layer
normalization and normal-vector reductions. Prior normals, resizing, masking
and patch accumulation stay on the GPU with one final normal-map readback.
RGB caching remains bounded; larger inputs stream patches.

## Limits

These changes remove source-level overhead, but are not a measured explanation
of the entire regression. Other Burn operations still compile and autotune.
CUDA here uses CubeCL kernels, not PyTorch/cuDNN kernels. Identical hardware
does not imply identical performance. No speedup or VRAM peak is claimed.

## Follow-up using the supplied run3 log

The six-image, 1024px WGPU runs took 111.99s at pixel batch 32768, 177.70s
at 327680, and 152.53s at 5000. These are user measurements, not new benchmarks.
The log reports unavailable accelerated flash and a switch to tiled GEMM.
Larger pixel batches alone therefore do not resolve the gap.

Source changes in this follow-up:

- Removed the process-global accelerated-availability flag. Failed configurations
  are cached by device identity, dtype and Q/K shape. A failure on one shape no
  longer disables every later shape. Removed a dispatch/cache race that could
  leave the GEMM fallback without its transposed keys.
- Added a short-light-attention kernel for up to 32 photographs. It computes
  dot products, online softmax and weighted values in one kernel, accumulates
  in FP32, and writes the layout needed by the following linear projection.
  It needs subgroup arithmetic, not cooperative matrices or CUDA. Variable
  subgroup widths are accounted for without assuming NVIDIA warp dimensions.
- Combined light-transformer K/V weights once during loading, replacing two
  projections with a single GEMM at each MAB. Checkpoint format is unchanged.
- Spatial GEMM fallback uses power-of-two query chunks under a 64 MiB
  per-score-tensor budget instead of irregular chunks under a 32 MiB budget.
  This increases that temporary buffer allowance; it does not raise the RGB
  cache limit or allocate full-image attention matrices.

The run log contains no operation-level timings; these changes target concrete
source inefficiencies but do not establish that the remaining 4x gap is closed.
No commands, builds, tests or inference were run for this follow-up.

## Scheduling follow-up from profile.json

The supplied verbose run completed in 103.60s; the non-verbose log reports
159.43s. The profile is valid and records zero cooperative-matrix configurations.
Exclusive category totals include attention 29.37s, linear 27.08s, convolution
10.92s and normalization 10.73s. These totals include first-use compilation and
tuning and are perturbed by profiling; they are not pure GPU kernel times.
The timing difference is consistent with temporary-buffer lifetime/queueing
pressure, but cold/warm compilation differences prevent treating it as proof.

Changes: default two-chunk completion windows, drains at encoder-batch and patch
boundaries, preallocated attention/light outputs, direct GEMM dispatch on devices
without CMMA configurations, and partial light-batch shape buckets of 256 pixels
(capped at the requested batch size). These are portable WGPU/CubeCL changes.
The --queue-depth option isolates scheduling behavior: 0 disables the added
waits; verbose mode already synchronizes more finely. Normal metadata and profile
context record queue depth. No builds or inference were run for this revision.
