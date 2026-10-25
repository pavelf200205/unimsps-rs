# Local persistence patch

Base: cubecl-environment 0.11.0, MIT OR Apache-2.0.

Only these source files differ from the registry package:
- persistence/cache_policy.rs (new): startup-only All/Shaders/Memory policy and
  shader DB telemetry counters. Configure before opening stores/devices; the
  policy is never changed during inference.
- persistence/mod.rs: expose that policy and its counters.
- persistence/storage.rs: route noncompiler namespaces to existing MemoryStorage
  in Shaders mode, all namespaces in Memory mode. Avoid durable namespace scans
  from the throughput cleaner in those modes.
- persistence/turso.rs: disable experimental multiprocess WAL on Windows only;
  expose checkpoint; block direct runtime-record opens in restricted modes and
  all explicit DB opens in Memory mode; count shader reads/inserts. Ordinary
  file locking and upstream connection/transaction behavior remain intact.

Shader namespaces are vulkan/ and cuda/, as used by CubeCL 0.11.0. Recheck these
prefixes when upgrading. Database probe/reporting uses explicit opens in Shaders
mode, while runtime records are off. Diagnostic counters cover DB reads/inserts,
not decoded artifacts, live pipeline reuse, compilation duration or GPU time.

persistence/store.rs is exactly upstream: lazy/eager behavior, remove, write-through
conflicts and storage failure semantics have not been changed. There is no write
freeze or shared-connection replacement.

This patch is required because the pinned runtime constructs its own environment
stores and has no public per-namespace routing hook. The autotune disk-only switch
is used directly; the throughput switch would also disable RAM reuse. Upstream's
Turso multiprocess WAL setting also needs the Windows compatibility adjustment.

Concurrent Windows instances should use separate cache directories. No global
Cargo registry source was edited. Full rationale and manual sequence are in
../../CACHE_AND_SCHEDULING.md.
