//! Standalone Uni-MS-PS inference. Tensor execution uses Burn 0.22.
pub mod model;
pub mod pipeline;
mod attention;
mod light_attention;
mod norm;
pub mod profile;
mod softmax;
pub mod execution;

mod depthwise;

mod conv_gemm;

/// Persistent GPU kernel, driver shader, and dispatch-strategy cache across runs.
pub mod cache {
    use serde::{Deserialize, Serialize};
    use std::{
        collections::BTreeSet,
        path::{Path, PathBuf},
    };

    #[derive(Default, Serialize, Deserialize)]
    pub struct KernelCache {
        pub version: u32,
        pub warmed_signatures: BTreeSet<String>,
        pub unavailable_flash_shapes: BTreeSet<String>,
    }

    const CACHE_VERSION: u32 = 1;

    pub fn default_cache_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("UNIMSPS_CACHE_DIR") {
            if !dir.trim().is_empty() {
                return PathBuf::from(dir);
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                return parent.join(".unimsps-kernels");
            }
        }
        PathBuf::from(".unimsps-kernels")
    }

    fn set_env_if_unset(key: &str, value: &str) {
        if std::env::var_os(key).is_none() {
            // Edition 2024 marks env mutation unsafe; called single-threaded at startup before device init.
            unsafe {
                std::env::set_var(key, value);
            }
        }
    }

    fn clean_driver_path(path: &Path) -> Option<String> {
        let abs = path.canonicalize().ok()?;
        let s = abs.to_string_lossy();
        // Strip Windows `\?\` extended-length prefix which breaks NVIDIA/CRT driver cache path parsing.
        let cleaned = s.strip_prefix(r"\?").unwrap_or(&s);
        Some(cleaned.to_owned())
    }

    /// Configure GPU driver and runtime persistent disk caches (CUDA PTX/NVRTC JIT cache,
    /// NVIDIA/Mesa/Vulkan/DX12 shader disk caches) before initializing the Burn device.
    pub fn init_environment(cache_dir: &Path, explicit_custom_dir: bool) -> KernelCache {
        let _ = std::fs::create_dir_all(cache_dir);

        // Always enable driver disk caches with a 2 GiB quota and disable automatic cache eviction.
        set_env_if_unset("CUDA_CACHE_DISABLE", "0");
        set_env_if_unset("CUDA_CACHE_MAXSIZE", "2147483648");
        set_env_if_unset("CUDA_FORCE_PTX_JIT", "0");
        set_env_if_unset("__GL_SHADER_DISK_CACHE", "1");
        set_env_if_unset("__GL_SHADER_DISK_CACHE_SIZE", "2147483648");
        set_env_if_unset("__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "1");
        set_env_if_unset("MESA_SHADER_CACHE_MAX_SIZE", "2G");

        // Only redirect driver-level cache paths when `--kernel-cache-dir` is explicitly specified;
        // otherwise keep the OS/driver native cache (%LOCALAPPDATA%\NVIDIA\GLCache / DXCache / ComputeCache)
        // so Vulkan, D3D12, and CUDA reuse already-compiled driver shaders across runs.
        if explicit_custom_dir {
            let cuda_dir = cache_dir.join("cuda_jit");
            let nv_dir = cache_dir.join("nvidia_shaders");
            let vk_dir = cache_dir.join("vulkan_shaders");
            let _ = std::fs::create_dir_all(&cuda_dir);
            let _ = std::fs::create_dir_all(&nv_dir);
            let _ = std::fs::create_dir_all(&vk_dir);

            if let Some(s) = clean_driver_path(&cuda_dir) {
                set_env_if_unset("CUDA_CACHE_PATH", &s);
            }
            if let Some(s) = clean_driver_path(&nv_dir) {
                set_env_if_unset("__GL_SHADER_DISK_CACHE_PATH", &s);
            }
            if let Some(s) = clean_driver_path(&vk_dir) {
                set_env_if_unset("MESA_SHADER_CACHE_DIR", &s);
                set_env_if_unset("AMD_VK_PIPELINE_CACHE_PATH", &s);
                set_env_if_unset("VK_PIPELINE_CACHE_PATH", &s);
            }
        }

        let manifest_path = cache_dir.join("kernel_state.json");
        let cache = std::fs::read(&manifest_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<KernelCache>(&bytes).ok())
            .filter(|c| c.version == CACHE_VERSION)
            .unwrap_or_else(|| KernelCache {
                version: CACHE_VERSION,
                ..Default::default()
            });

        crate::attention::load_unavailable(cache.unavailable_flash_shapes.iter().cloned());
        cache
    }

    impl KernelCache {
        pub fn is_warmed(&self, signature: &str) -> bool {
            self.warmed_signatures.contains(signature)
        }

        pub fn record_and_save(&mut self, cache_dir: &Path, signature: Option<String>) {
            if let Some(sig) = signature {
                self.warmed_signatures.insert(sig);
            }
            for key in crate::attention::snapshot_unavailable() {
                self.unavailable_flash_shapes.insert(key);
            }
            self.version = CACHE_VERSION;
            if std::fs::create_dir_all(cache_dir).is_ok() {
                let manifest_path = cache_dir.join("kernel_state.json");
                if let Ok(bytes) = serde_json::to_vec_pretty(self) {
                    let _ = std::fs::write(manifest_path, bytes);
                }
            }
        }
    }
}
