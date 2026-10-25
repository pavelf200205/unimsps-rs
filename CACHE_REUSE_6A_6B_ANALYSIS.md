# 6a / 6b evidence, 2026-10-25

Inputs: experiments/v4_20261025_192138_6a and ..._6b, build
v4-shaders-ddfcc85927028765 (853268e), six images, shader-only persistence.
No compilation, inference or tests were run during this analysis.

| Measurement | 6a, cold | 6b, reuse |
| --- | ---: | ---: |
| Preparation, seconds | 33.81 | 15.71 |
| Preparation shader DB hits / misses | 1 / 196 | 187 / 0 |
| Preparation shader DB read time, ms | 26.57 | 39.17 |
| View 00, seconds | 42.46 | 15.02 |
| View 01, seconds | 43.45 | 44.31 |
| View 02, seconds | 35.36 | 59.21 |
| View 10, seconds | 14.92 | 37.96 |

Reuse is real and preparation improved by 18.09 seconds (53.5%). This includes
tuning and driver setup, so it is not an isolated compilation benchmark.
The cold run was already slow on views 00-03 and recovered within the same
process by view 10. The reused run started fast before slowing. This does not
support a simple deterministic rule that restored shaders always execute slowly.

6b view 02 spent 49.09 seconds in patch completion waits, 6.27 seconds in network
submission and 1.43 seconds in final flush. Shader database accesses were zero
during that view. Most slow views had zero shader database accesses. Output
saving took only approximately 0.03-0.05 seconds per view.

Consecutive patches alternate between roughly 0.3-0.4 seconds and stretches of
1.4-2.0 seconds. Sampled allocator reservation is similar across both runs,
around 792 MiB after the first view, and no allocator cleanup events occurred.
These samples do not measure transient peak usage, driver allocations, residency
or competing processes. Completion waits are CPU wall times, not device timings:
GPU execution, preemption and delayed CPU completion handling remain confounded.

There are 219 valid unique tuning records per run, with the same keys. 65 winner
indices differ, all matmul. Median recorded fastest-time ratio (6b/6a) is about
0.95. This is not evidence of a broad 3x warmup performance regression, but does
not establish that different choices are harmless during real inference.
Both logs also contain two malformed lines. The current logger opens files with
independent offsets when append=false, without truncating. The application now
clears each log once and enables append for subsequent runtime writers.

## Decision

Keep shader-only reuse available: it demonstrably saves startup time. Do not
claim it is exonerated, or change GPU kernels and cache mechanisms together.
The next control is the SAME 853268e executable, six images, --cache-mode memory,
with identical remaining settings and a new output folder. No 9-image run is
needed yet. Preserve the existing cache. A slowdown with memory mode would show
that restoring shaders is not necessary to trigger the problem; a fast run would
justify a repeated alternating comparison, not prove causality from one sample.
If memory mode also stalls, capture device timestamps/telemetry around a stall
next to distinguish device execution from host/driver waits.

## Build identification

Run tools/build_release.ps1 manually instead of plain cargo build to also create
unimsps_<commit>.exe beside the normal Cargo executable. The normal artifact
remains for Cargo. --version, the startup banner and metadata embed the revision.
Tracked local changes add -dirty; different binaries with the same label receive
a content-hash suffix rather than overwriting an earlier copy. Source archives
without Git use unknown. Untracked files are not included in the dirty marker.
The helper accepts -TargetDir for a custom Cargo output directory.
