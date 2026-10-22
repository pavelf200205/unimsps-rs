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

## Follow-up after the user-reported 50–55-second improvements

Preserved native GEMM as default, 65536 pixel batches, 512 MiB score budget,
packed weights/KV, precomputed pooling queries, fused LayerNorm, foreground
gather sharing and asynchronous submission. Added a dedicated row-softmax
kernel with FP32 max/sum and conditional in-place score-buffer reuse. The two
GEMMs remain unchanged. Unsupported devices/shapes retain Burn softmax; the
--burn-softmax flag explicitly selects it.

Allocator cleanup was invoked at each patch boundary once reservations exceeded
1536 MiB. This may discard useful reusable allocations; its significance cannot
be established from older profiles. Added a configurable --pool-cache-mib
threshold (same default; 0 disables explicit cleanup) and cleanup timing/events.
Attention profiling now splits QK GEMM, softmax, AV GEMM. No new performance
measurements or compilation were performed, and no speedup is claimed.

Saved evidence: experiments/unimsps_run11/profile.json uses native GEMM and
65536 pixel batches. Its largest repeated attention shape totals about 9 seconds
over 2016 calls, but combines QK, softmax and AV. Cold-start outliers are also
present. Later run14 metadata records 58.2 seconds for view_00, without an
operation profile. These are historical observations, not measurements of this
change or proof that softmax is the remaining dominant cost.

## Encoder and MLP follow-up

Default linear projections now pass bias through Burn's native linear operation
while preserving batch folding. This exposes linear semantics to the backend;
actual GEMM epilogue fusion depends on backend dispatch, and is not claimed here.
The MLP uses a direct token-layout 3x3 depthwise kernel, accumulating in FP32,
adding bias, rounding the convolution result to native precision, and evaluating
erf GELU before storing. It avoids the two NCHW layout conversions and separate
activation pass. FP32 GELU intermediates can change FP16 rounding. Unsupported
backends retain the prior convolution and GELU. This kernel does not use subgroups
or cooperative matrices and is intended for portable WGPU execution.

LayerNorm reuses exclusively owned contiguous inputs after loading each row into
registers. The ownership guard prevents mutation of residual/shared inputs.
Verbose reports now isolate encoder blocks, spatial reduction preparation,
residual additions, patch layout conversion, foreground gathering and mask setup.
These synchronized spans perturb fusion and should only guide diagnosis.

Use --reference-encoder for the prior projection/depthwise implementation; it
retains the current LayerNorm implementation. --burn-softmax independently
selects the prior softmax. tools/compare_performance.ps1 runs all four combinations
in alternating order and exports per-view timings when manually invoked.
It does not compare numerical outputs. No commands, builds or benchmarks were
executed during this source-only implementation. Performance is unverified.

## Convolution regression follow-up

Spatial reduction convolutions (kernel=stride, no padding) and stride-2 3x3 patch
embeddings now use explicit bounded patch packing plus native biased linear GEMM.
Packing reads [B,H*W,C] with actual strides and emits channel-inner patches.
OIHW weights are reordered once on the CPU before upload, not each inference.
The output remains [B,OH*OW,O], directly consumed by LayerNorm. The small first
embedding, depthwise MLP, decoder and output convolutions retain their paths.

Each patch workspace is limited to 16 MiB using power-of-two row chunks. This
is not a total VRAM limit: outputs, weights and queued work also consume memory.
Original convolution weights remain available for fallback, with additional
packed copies for targeted layers when the GEMM option is enabled.

Use --burn-convolutions to select the prior Burn convolution dispatch and skip
packed-weight upload. The switch is independent of --reference-encoder and
--burn-softmax. Unsupported backends retain Burn convolution automatically.
Verbose profiles separate conv-pack, conv-matmul and the encompassing conv-gemm
span, with route counts and gemm_convolutions recorded in metadata.

No compilation, numerical comparisons or timing runs were executed. GEMM and
convolution may use different floating-point accumulation/reduction orders.
Performance remains to be established by non-verbose runs with and without
--burn-convolutions; use verbose separately to identify packing/GEMM costs.
