# Local persistence patch

Upstream: cubecl-environment 0.11.0, MIT OR Apache-2.0.

Changes in src/persistence/turso.rs:
- Disable experimental multiprocess WAL on Windows only. Turso 0.8.3 default
  WindowsIO does not implement shared-WAL coordination; the old open falls back
  to process memory. The normal Windows file lock remains active. Concurrent
  processes cannot share one writable cache; use separate --kernel-cache-dir
  values when running multiple instances. Unix multiprocess WAL is unchanged.
- Expose Database::checkpoint to fold committed WAL data into the main cache
  database without changing journal mode or closing active runtime connections.

This patch is scoped to unimsps-rs by Cargo's local patch section.
No global Cargo registry files were modified.
