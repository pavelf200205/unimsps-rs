# Recent-run comparison (24 October 2026)

Read-only analysis of experiments metadata and the persisted CubeCL database.
No inference, compilation, kernel benchmarks, or hardware monitoring was run.

| Run suffix | Images | Views 00-03 mean | Views 09-19 mean |
|---|---:|---:|---:|
| version4 | 6 | 15.27 s | 15.39 s |
| version4_run2 | 9 | 24.24 s | 26.08 s |
| version4_1 | 6 | 15.03 s | Not available |
| version4_1_kernel_reuse_check | 9 | 24.94 s | 49.26 s |
| version4_1_kernel_reuse_check2 | 6 | 44.69 s | Not available |

The final run currently contains only four metadata files. The latest nine-image
run starts normally; slowdown persists after the short 512-scale views and into
a subsequent six-image process. Same-view comparisons control for mask/workload
variation better than comparing different viewpoints. Recorded queue depth,
score budget, precision and pool threshold are the same. Metadata does not
include patch timing, adapter budget, GPU clocks, temperature, CPU load or
competing processes, so it cannot identify the execution-stall cause.

The cache database contains 218 persisted Vulkan artifacts and 317 tuning
decisions. The nine-image cache-building session began at 16:36:00.777 UTC. Its
last recorded compilation was at +12.783 s and last tuning at +11.098 s, before
the first view began at about 16:36:55.498 UTC. View 09 began slowing around
16:39:07 UTC (19:39:07 Kyiv time). No new compilation/tuning records coincide
with that slowdown. This is evidence against cold compilation being its cause.

Background GPU contention, WDDM VRAM residency/paging, clock or power changes,
and application/driver execution or memory-reuse behavior remain candidates.
A particular background application cannot be blamed without telemetry. A new
scheduling fix cannot be selected responsibly from total view durations alone.

Added optional diagnostics: --trace-performance saves timing_trace.json without
extra GPU waits, and tools/monitor_gpu.ps1 can record vendor-neutral Windows GPU
utilization and process memory counters when the user launches it. Use normal
execution without --verbose, to preserve asynchronous behavior. The trace and
monitor should cover a fast-to-slow transition in the same run. Look for:
- time concentrated in network_submit versus flush_and_completion_wait;
- cleanup duration or pool-reservation growth preceding the transition;
- rising shared GPU memory or other processes' dedicated GPU usage;
- competing GPU-engine utilization; clocks/temperature need a separate monitor.

## Separate-cache experiments (traces 7, 8 and 9)

The user reports separate cache_6/cache_9 directories. These older trace files
do not store the cache path; new format-2 reports include it explicitly.

- Trace 7 (six images, fresh cache): view 09 takes 28.14 s; views 10 and 11
  recover to 15.10/15.11 s without restarting or changing cache. No cleanup.
- Trace 8 (nine images, separate fresh cache): views 00/01 take 23.64/23.72 s,
  then 02/03 take 59.05/48.33 s. Views 04/05 take 10.39/12.76 s. Cleanups occur
  in both the fast and slow views; the first five views each have one cleanup.
- Trace 9 (six images, original cache_6 reused): views 00/01 take 22.95/21.36 s;
  02/03 recover to 15.59/15.20 s and the remaining views finish normally. No cleanup.

These observations argue against cross-image-count database contamination as
a necessary cause. They do not exclude driver state, device clocks/power,
runtime synchronization, or data-dependent kernel behavior. Reported GPU/core
and hotspot temperatures are 65/76 C, but no clock/power trace was recorded.

Specific code paths to distinguish:

1. model.rs forward_masked processes six lights in one batch and nine lights
   in batches of eight and one at every encoder stage. Attention chunking and
   the additional batch increase dispatch/submission work. This is a concrete
   workload difference, not proof of the intermittent defect.
2. CubeCL WGPU stream.rs SubmissionLoad::regulate blocks on an earlier
   submission after 512 accumulated tasks. Thus the application's
   network_submit wall time contains GPU waits. It is not pure CPU overhead.
3. pipeline.rs flushes pending patch work, then execution.rs checkpoint_patch
   calls device.sync. WGPU completion uses device polling and a queue callback.
   The prior trace combined both operations and had no device clock readings.
4. execution.rs trim_memory can purge the reusable pool between nine-image
   views because the unused-byte threshold is exceeded at a small scale. This
   can add allocation churn, but cannot alone explain six-image stalls without
   any cleanup or the nine-image run's alternating fast/slow views.

Added opt-in --trace-gpu to distinguish device elapsed windows, final submission,
completion waits and CPU stage submission. No algorithm, batching or allocator
policy change is claimed as a fix. Source-reviewed only; the user runs builds.

## Device-timestamp results: traces 10, 11 and 12

Trace 10 uses six images and device timing. It has real stalls: view 07 takes 5.82 s, view 08 11.80 s, view 09 30.30 s; view 10 recovers to 15.27 s.
Trace 11 uses nine images and device timing. Views 00/01 take 37.13/66.77 s. Sum of patch device windows is 35.91/62.61 s respectively.
Trace 12 has nine-image metadata for views 00/01 at 39.76/55.38 s, but no timing_trace.json files. It establishes slow unprofiled execution, not a breakdown of its cause.

Decisive examples:
- Trace 10, view 07, scale 512, x=128 y=256: wall 1404.14 ms; device 1400.26 ms; network submission 33.80 ms; final flush 1.19 ms; completion wait 1368.92 ms.
- Trace 11, view 01, scale 512, x=128 y=0: wall 3134.47 ms; device 3128.69 ms. At x=0 y=128 in the same scale/view: wall 598.97 ms; device 596.18 ms.

The slow six-image example has finished submitting almost all patch work in about 35 ms, while the GPU timestamp interval lasts about 1.4 s. A delayed CPU completion callback alone cannot account for that interval. Most slow patches have device elapsed time within roughly 1% of total patch wall time. First patches can include earlier coarse-scale work outside their device window and must not be used to calculate callback delay by subtraction.

CPU stage stalls migrate across encoder stages and light-transformer submission, consistent with runtime backpressure charging earlier GPU work to the currently issuing stage. These labels do not identify a slow GPU kernel. Device windows include gaps/preemption and are not kernel-busy-time sums.

Nine-image 8+1 batching adds submission work, but is neither a necessary cause (six-image stalls exist) nor a sufficient explanation of intermittent 5x changes. Timestamp instrumentation is not required to reproduce the slowdown (trace 12). GPU clocks/power were not recorded; reported normal temperatures do not establish stable clocks. No specific faulty kernel has yet been demonstrated.

No source changes, compilation, or inference were performed for this analysis.
