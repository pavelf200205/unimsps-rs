# Local persistence patch

Upstream: cubecl-environment 0.11.3, MIT OR Apache-2.0.

Changes:
- Disable experimental multiprocess WAL on Windows. Turso's default WindowsIO
  lacks the required shared-WAL coordination. Normal file locks remain active;
  concurrent Windows processes should use separate cache directories.
- Expose Database::checkpoint for folding the WAL into the main database.
- Add a startup-only PersistencePolicy at the Storage boundary: All preserves
  upstream behavior; Memory routes every namespace to process-local storage;
  Shaders persists only vulkan/ and cuda/ compiler namespaces. Tuning and
  throughput stores still reuse their data in memory.
- Disable direct runtime-record database opens and durable namespace enumeration
  in Shaders/Memory modes. Memory also rejects explicit database opens. Shaders
  permits explicit opens for the application's cache probe and reporting.

Configure the policy before opening any stores or initializing a device. Changing
it later does not migrate existing stores. Compiler prefixes were checked against
CubeCL 0.11.3; review them when updating CubeCL.

Cargo's local patch scopes this to unimsps-rs. No global registry source was edited.
