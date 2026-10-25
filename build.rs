use std::{env, fs, path::{Path, PathBuf}, process::Command};

fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", dir.display());
    for entry in fs::read_dir(dir).expect("read source directory") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() { collect(&path, files); }
        else if path.extension().is_some_and(|ext| ext == "rs") { files.push(path); }
    }
}

fn hash_part(hash: &mut u64, bytes: &[u8]) {
    // Stable FNV-1a, length-delimited. This is a cache namespace, not a security digest.
    for byte in (bytes.len() as u64).to_le_bytes().iter().chain(bytes) {
        *hash = (*hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
}

fn main() {
    let mut files = vec![PathBuf::from("Cargo.toml"), PathBuf::from("Cargo.lock"), PathBuf::from("build.rs"),
        PathBuf::from("vendor/cubecl-environment/Cargo.toml"), PathBuf::from("vendor/cubecl-environment/build.rs")];
    collect(Path::new("src"), &mut files);
    collect(Path::new("vendor/cubecl-environment/src"), &mut files);
    files.sort();
    let mut hash = 0xcbf29ce484222325;
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        hash_part(&mut hash, path.to_string_lossy().replace('\\', "/").as_bytes());
        let content = fs::read_to_string(&path).expect("read build input").replace("\r\n", "\n");
        hash_part(&mut hash, content.as_bytes());
    }
    for key in ["TARGET", "PROFILE", "CARGO_ENCODED_RUSTFLAGS", "CARGO_CFG_TARGET_FEATURE"] {
        println!("cargo:rerun-if-env-changed={key}");
        hash_part(&mut hash, env::var(key).unwrap_or_default().as_bytes());
    }
    let mut features: Vec<_> = env::vars().filter(|(k,_)| k.starts_with("CARGO_FEATURE_")).collect();
    features.sort();
    for (key,value) in features { hash_part(&mut hash, key.as_bytes()); hash_part(&mut hash, value.as_bytes()); }
    let compiler = Command::new(env::var_os("RUSTC").expect("RUSTC")).arg("--version").output().expect("rustc version");
    assert!(compiler.status.success(), "could not identify compiler");
    hash_part(&mut hash, &compiler.stdout);
    println!("cargo:rustc-env=UNIMSPS_CACHE_BUILD_ID=v4-shaders-{hash:016x}");
}
