//! Windows Vulkan fixed-path driver-cache restoration. The supervisor never initializes a GPU.
//! Only a successful worker's first cache is published, after process teardown.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::{self, File, OpenOptions}, io::Read,
    path::{Path, PathBuf}, process::{Command, ExitStatus}, time::{SystemTime, UNIX_EPOCH}};

pub const SESSION_ENV: &str = "UNIMSPS_DRIVER_CACHE_SESSION";

// Rust canonicalizes Windows paths with a verbatim prefix. Third-party driver
// cache code may not accept that syntax; pass the ordinary absolute spelling.
fn ordinary_path(path: &Path) -> Result<PathBuf> {
    let text = path.to_str().context("Driver cache path must be valid Unicode")?;
    if let Some(unc) = text.strip_prefix("\\\\?\\UNC\\") {
        Ok(PathBuf::from(format!("\\\\{unc}")))
    } else if let Some(disk) = text.strip_prefix("\\\\?\\") {
        Ok(PathBuf::from(disk))
    } else { Ok(path.to_path_buf()) }
}

fn clear_working(root: &Path, working: &Path) -> Result<()> {
    if !working.exists() { return Ok(()); }
    let resolved = fs::canonicalize(working)?;
    ensure!(resolved.parent() == Some(root), "Unsafe driver-cache cleanup path");
    files(working)?; // Refuse junctions/symlinks anywhere before recursive removal.
    fs::remove_dir_all(working)?;
    Ok(())
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
struct Manifest {
    version: u32,
    profile: String,
    files: BTreeMap<String, String>,
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 { break; }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

// Avoid opening a graphics device in the supervisor just to identify a driver.
// Windows keeps NVIDIA's user-mode modules in DriverStore; their installation
// paths, sizes and timestamps change on driver replacement. Include all matching
// packages (including retained older packages) for conservative invalidation.
fn driver_installation_stamp() -> Vec<String> {
    let mut stamp = Vec::new();
    if let Some(windows) = std::env::var_os("SystemRoot") {
        let system = PathBuf::from(windows).join("System32");
        let mut folders = vec![system.clone()];
        if let Ok(entries) = fs::read_dir(system.join("DriverStore/FileRepository")) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name.starts_with("nv") && name.contains(".inf_") { folders.push(entry.path()); }
            }
        }
        for folder in folders {
            for module in ["nvoglv64.dll", "nvldumdx.dll", "nvwgf2umx.dll"] {
                let path = folder.join(module);
                if let Ok(meta) = fs::metadata(&path) {
                    let modified = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok());
                    stamp.push(format!("{}|{}|{modified:?}", path.display(), meta.len()));
                }
            }
        }
    }
    stamp.sort();
    stamp
}

// Refuse junctions as well as symbolic links before recursive copying/removal.
fn checked_metadata(path: &Path) -> Result<fs::Metadata> {
    let meta = fs::symlink_metadata(path)?;
    ensure!(!meta.file_type().is_symlink(), "Cache path is a symbolic link: {}", path.display());
    #[cfg(windows)] {
        use std::os::windows::fs::MetadataExt;
        ensure!(meta.file_attributes() & 0x400 == 0,
            "Cache path is a reparse point: {}", path.display());
    }
    Ok(meta)
}

fn inventory(root: &Path, dir: &Path, files: &mut BTreeMap<String, String>) -> Result<()> {
    ensure!(checked_metadata(dir)?.is_dir(), "Expected cache directory: {}", dir.display());
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let meta = checked_metadata(&path)?;
        if meta.is_dir() {
            inventory(root, &path, files)?;
        } else {
            ensure!(meta.is_file(), "Unexpected cache file type: {}", path.display());
            files.insert(path.strip_prefix(root)?.to_string_lossy().into_owned(), hash_file(&path)?);
        }
    }
    Ok(())
}

fn files(root: &Path) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    inventory(root, root, &mut result)?;
    Ok(result)
}

fn copy_tree(source: &Path, target: &Path) -> Result<()> {
    ensure!(checked_metadata(source)?.is_dir(), "Expected seed directory");
    fs::create_dir(target)?;
    for entry in fs::read_dir(source)? {
        let path = entry?.path();
        let meta = checked_metadata(&path)?;
        let dst = target.join(path.file_name().context("Missing cache filename")?);
        if meta.is_dir() { copy_tree(&path, &dst)?; }
        else {
            ensure!(meta.is_file(), "Unexpected seed file type");
            fs::copy(&path, dst)?;
        }
    }
    Ok(())
}

fn has_shader_pair(root: &Path, entries: &BTreeMap<String, String>) -> bool {
    entries.keys().any(|name| {
        let path = Path::new(name);
        path.extension().is_some_and(|ext| ext == "toc")
            && entries.contains_key(&path.with_extension("bin").to_string_lossy().into_owned())
            && fs::metadata(root.join(path)).is_ok_and(|m| m.len() > 32)
            && fs::metadata(root.join(path.with_extension("bin"))).is_ok_and(|m| m.len() > 0)
    })
}

fn unique_directory(parent: &Path) -> Result<PathBuf> {
    let time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    for attempt in 0..100 {
        let path = parent.join(format!("{}-{time}-{attempt}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    bail!("Could not allocate a unique driver-cache session")
}

fn remove_session(sessions: &Path, session: &Path) -> Result<()> {
    // Only delete a direct child allocated by this supervisor. Validate every
    // descendant before remove_dir_all so a junction cannot redirect cleanup.
    let resolved = fs::canonicalize(session)?;
    let parent = fs::canonicalize(sessions)?;
    ensure!(resolved.parent() == Some(parent.as_path()), "Unsafe session cleanup path");
    files(session)?;
    fs::remove_dir_all(session)?;
    Ok(())
}

fn lock_root(root: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(windows)] {
        use std::os::windows::fs::OpenOptionsExt;
        // OS releases this exclusive handle even after a supervisor crash.
        options.share_mode(0);
    }
    options.open(root.join("driver-cache.lock"))
        .context("Cannot lock cache root. Another reconstruction may be using it; use a separate --kernel-cache-dir")
}

/// Restore the NVIDIA cache at one stable path, then launch inference.
/// A seed is never taken from legacy caches or from a session that reused a seed.
pub fn supervise(cache_root: &Path, profile: String, persistent: bool, refresh: bool) -> Result<ExitStatus> {
    fs::create_dir_all(cache_root)?;
    let root = fs::canonicalize(cache_root)?;
    let _lock = lock_root(&root)?; // Also serialize access to the shared CubeCL DB.
    let executable = std::env::current_exe()?;
    let driver_stamp = driver_installation_stamp();
    if driver_stamp.is_empty() {
        eprintln!("NVIDIA driver installation could not be fingerprinted; after a driver change use --refresh-driver-cache");
    }
    let identity = format!("driver-seed-v1\n{}\n{driver_stamp:?}\n{profile}", hash_file(&executable)?);
    let key = format!("{:x}", Sha256::digest(identity.as_bytes()));
    let managed = root.join("nvidia-driver-snapshots-v1");
    fs::create_dir_all(&managed)?;
    checked_metadata(&managed)?;
    let profile_dir = managed.join(key);
    fs::create_dir_all(&profile_dir)?;
    checked_metadata(&profile_dir)?;
    let sessions = profile_dir.join("sessions");
    fs::create_dir_all(&sessions)?;
    checked_metadata(&sessions)?;
    let seed = profile_dir.join("seed");
    let reused = persistent && !refresh && seed.exists();
    let session = unique_directory(&sessions)?;
    let working = ordinary_path(&root)?.join("nvidia_gl_cache");
    if reused {
        checked_metadata(&seed)?;
        checked_metadata(&seed.join("manifest.json"))?;
        let expected: Manifest = serde_json::from_slice(&fs::read(seed.join("manifest.json"))?)?;
        ensure!(expected.version == 1 && expected.profile == identity,
            "Driver seed identity mismatch; use --refresh-driver-cache");
        let actual = files(&seed.join("cache"))?;
        ensure!(expected.files == actual, "Driver seed was modified; use --refresh-driver-cache");
        clear_working(&root, &working)?;
        copy_tree(&seed.join("cache"), &working)?;
        eprintln!("NVIDIA driver cache: restoring immutable seed {}", seed.display());
    } else {
        if working.exists() {
            // Preserve existing evidence when starting a new profile/refresh.
            files(&working)?;
            fs::rename(&working, session.join("previous-driver-cache"))?;
        }
        fs::create_dir(&working)?;
        eprintln!("NVIDIA driver cache: starting fresh (legacy cache folders are not imported)");
    }
    eprintln!("NVIDIA driver cache fixed working path: {}", working.display());
    let status = Command::new(executable)
        .args(std::env::args_os().skip(1))
        .arg("--internal-cache-worker")
        .env(SESSION_ENV, &working)
        // Override inherited paths: isolation must not silently use a global cache.
        .env("__GL_SHADER_DISK_CACHE", "1")
        .env("__GL_SHADER_DISK_CACHE_PATH", &working)
        .env("__GL_SHADER_DISK_CACHE_SIZE", "2147483648")
        // Match the driver's settings from the successful manual restore runs.
        .env("__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "1")
        .status().context("Could not launch inference worker")?;

    // Waiting for process exit above is essential: driver DLL teardown may write
    // cache files after Rust's main has returned.
    if !status.success() {
        eprintln!("Inference did not succeed; seed unchanged. Driver cache retained at {}", working.display());
        return Ok(status);
    }
    if persistent && !reused {
        let entries = files(&working)?;
        if has_shader_pair(&working, &entries) {
            copy_tree(&working, &session.join("cache"))?;
            let manifest = Manifest { version: 1, profile: identity, files: entries };
            fs::write(session.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
            // Keep the previous seed as evidence during an explicit refresh.
            let previous = if seed.exists() {
                let archive = profile_dir.join(format!("seed-before-{}", session.file_name().unwrap().to_string_lossy()));
                fs::rename(&seed, &archive)?;
                Some(archive)
            } else { None };
            if let Err(error) = fs::rename(&session, &seed) {
                if let Some(previous) = previous { let _ = fs::rename(previous, &seed); }
                return Err(error).context("Could not publish driver seed; completed session preserved");
            }
            eprintln!("NVIDIA driver cache: saved first successful run as {}", seed.display());
            return Ok(status);
        }
        eprintln!("WARNING: no populated NVIDIA shader cache at {}; no backup was saved. On NVIDIA this means driver reuse is NOT working; the next launch may compile again. On other vendors this is expected.", working.display());
    }
    // Leave the live-path cache in place after exit, just like manual runs. It
    // will be replaced from the seed BEFORE the next launch, never merged back.
    if !session.join("previous-driver-cache").exists() {
      if let Err(error) = remove_session(&sessions, &session) {
        eprintln!("Could not remove disposable driver cache {}: {error:#}", session.display());
      }
    }
    Ok(status)
}
