use anyhow::{Result, ensure};
use burn::tensor::{Device, DeviceKind, FloatDType};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use unimsps::{
    model::{Model, ModelOptions, read_tensor, tensor},
    pipeline::{Input, ProcessOptions, SaveFormats, process_with_options, save_with_formats},
};

#[derive(Parser)]
#[command(about = "Uni-MS-PS normal estimation using Burn 0.22")]
#[command(version = env!("UNIMSPS_GIT_REVISION"))]
struct Cli {
    /// Detailed synchronized CPU/GPU wall timings; writes profile.json. Slows execution.
    #[arg(long, short = 'v', global = true)]
    verbose: bool,
    /// Lightweight patch wall-time and pool trace; adds no GPU waits and writes timing_trace.json.
    #[arg(long, global = true)]
    trace_performance: bool,
    /// Trace patch device timestamps and CPU stage submission times; implies --trace-performance.
    /// Changes command boundaries; use without --verbose to investigate stalls.
    #[arg(long, global = true, conflicts_with = "verbose")]
    trace_gpu: bool,
    /// Maximum complete patches between GPU waits; inner chunks submit asynchronously; 0 is unbounded.
    #[arg(long, default_value_t = 1, global = true)]
    queue_depth: usize,
    #[arg(long, global = true)]
    cpu: bool,
    #[arg(long, conflicts_with = "cpu", global = true)]
    integrated: bool,
    /// Use the NVIDIA CUDA backend (equivalent to --api cuda).
    #[arg(long, conflicts_with_all = ["cpu", "integrated"], global = true)]
    cuda: bool,
    /// Auto selects WGPU on every build unless --cuda is passed.
    #[arg(long, default_value = "auto", value_parser = ["auto", "cuda", "vulkan", "dx12", "metal"], global = true)]
    api: String,
    /// Set-transformer pixel batch size (defaults to 65536 to process entire patches in one batch).
    #[arg(long, default_value_t = 65536, global = true)]
    pixel_batch: usize,
    /// Spatial attention query limit; 0 uses full chunk budget. Unfused scores remain bounded.
    #[arg(long, default_value_t = 0, global = true)]
    query_batch: usize,
    /// Images processed together by the spatial encoder (0 = auto-match images per view, capped at 12).
    #[arg(long, default_value_t = 0, global = true)]
    encoder_batch: usize,
    /// Auto selects FP16 on supporting GPUs, FP32 on CPU/other adapters.
    #[arg(long, default_value = "auto", value_parser = ["auto", "fp16", "fp32"], global = true)]
    precision: String,
    /// RGB input/pyramid cache budget, excluding model weights and normal maps.
    #[arg(long, default_value_t = 512, global = true)]
    image_cache_mib: usize,
    /// Memory budget in MiB for attention score matrix allocation before chunking (default: 256 MiB).
    #[arg(long, default_value_t = 256, global = true)]
    score_budget_mib: usize,
    /// Use Burn's general softmax instead of the dedicated GEMM-attention row kernel.
    #[arg(long, global = true)]
    burn_softmax: bool,
    /// Retain previous matmul+bias and NCHW MLP convolution for timing comparisons.
    #[arg(long, global = true)]
    reference_encoder: bool,
    /// Retain Burn convolution dispatch for spatial reductions and stride-2 embeddings.
    #[arg(long, global = true)]
    burn_convolutions: bool,
    /// Unused allocator cache threshold at scale boundaries; 0 disables trimming (default: 1024 MiB).
    #[arg(long, default_value_t = 1024, global = true)]
    pool_cache_mib: usize,
    /// Force custom subgroup fused spatial attention instead of standard hardware GEMM.
    #[arg(long, global = true)]
    fused_attention: bool,
    /// Legacy flag: Use explicit matmul/softmax spatial attention (now the default).
    #[arg(long, global = true)]
    unfused_attention: bool,
    /// Skip pipeline and GPU kernel pre-warm pass before timed inference.
    #[arg(long, global = true)]
    no_warmup: bool,
    /// Comma-separated normal map formats to save: exr, png, mat, f32, or all (default: exr,png).
    #[arg(long, visible_aliases = ["format", "save-formats", "save_formats"], value_delimiter = ',', default_value = "exr,png", global = true)]
    formats: Vec<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Doctor,
    Infer {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Number of input photographs; -1 uses all. A subset uses lexical filename order.
        #[arg(long, visible_aliases = ["nb-img", "nb_img", "num-images", "num_images", "num-imgs", "num_imgs"], default_value_t = -1, allow_hyphen_values = true)]
        images: isize,
        /// Treat each immediate subdirectory as a separate object, loading the model once.
        #[arg(long)]
        batch: bool,
    },
    Verify {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long, default_value_t = 0.002)]
        tolerance: f32,
    },
}

fn main() -> Result<()> {
    let program_start = std::time::Instant::now();
    let cli = Cli::parse();
    eprintln!("unimsps {} (process-local runtime; Burn 0.22.0 / CubeCL 0.11.0)", env!("UNIMSPS_GIT_REVISION"));
    unimsps::profile::configure(cli.verbose);
    unimsps::trace::configure(cli.trace_performance);
    unimsps::trace::configure_gpu(cli.trace_gpu);
    unimsps::execution::configure(cli.queue_depth);
    unimsps::execution::configure_pool_cache(cli.pool_cache_mib);

    // Vulkan's upstream dependencies enable runtime persistence transitively.
    // Keep RAM throughput reuse, but force upstream's MemoryStorage fallback:
    // an existing regular file cannot be a database directory. The executable
    // itself is never opened for writing; directory creation fails first.
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    let executable = std::env::current_exe()?;
    ensure!(executable.is_file(), "Cannot establish a process-only runtime storage location");
    config.environment.path = CacheConfig::Directory(executable);
    config.compilation.cache = false;
    config.autotune.disable_cache = true;
    config.throughput.disable_cache = false;
    config.environment.records.level = cubecl::records::RecordLevel::Off;
    config.environment.records.keep_sessions = Some(0);
    ensure!(CubeClRuntimeConfig::try_set(config), "CubeCL initialized before runtime configuration");

    let use_cuda = cli.cuda || cli.api == "cuda";
    ensure!(!(cli.integrated && use_cuda), "--integrated cannot select CUDA");
    let effective_api = if cli.cpu {
        "cpu"
    } else if use_cuda {
        "cuda"
    } else {
        cli.api.as_str()
    };

    let device = if cli.cpu {
        Device::flex()
    } else if use_cuda {
        #[cfg(feature = "cuda")]
        {
            Device::cuda(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            anyhow::bail!("CUDA support requires rebuilding with cargo build --release --features cuda");
        }
    } else {
        use burn::tensor::wgpu::WgpuBackend;
        Device::wgpu_options()
            .device_kind(if cli.integrated {
                DeviceKind::IntegratedGpu(0)
            } else {
                DeviceKind::DefaultDevice
            })
            .graphics_api(match cli.api.as_str() {
                "vulkan" => WgpuBackend::Vulkan,
                "dx12" => WgpuBackend::Dx12,
                "metal" => WgpuBackend::Metal,
                _ => {
                    #[cfg(any(target_os = "windows", target_os = "linux"))]
                    {
                        WgpuBackend::Vulkan
                    }
                    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
                    {
                        WgpuBackend::Auto
                    }
                }
            })
            .init()
            .map_err(|e| anyhow::anyhow!("GPU initialization: {e:?}"))?
    };
    eprintln!("Device: {:?} (api: {effective_api})", device.identity());
    eprintln!("Queue depth: {} (0 = asynchronous flush)", cli.queue_depth);

    match cli.command {
        Command::Doctor => {
            let x = tensor(vec![1., 2., 3., 4.], [2, 2], &device);
            println!("Burn 0.22 matmul: {:?}", read_tensor(x.clone().matmul(x)));
        }
        Command::Infer {
            model,
            data,
            out,
            images,
            batch,
        } => {
            ensure!(images == -1 || images > 0, "--images must be -1 (all) or a positive count");
            let save_formats = SaveFormats::parse(&cli.formats)?;
            let image_limit = if images == -1 { None } else { Some(images as usize) };
            let mut objects = if batch {
                std::fs::read_dir(&data)?
                    .map(|e| e.map(|e| e.path()))
                    .collect::<std::io::Result<Vec<_>>>()?
                    .into_iter()
                    .filter(|p| p.is_dir() && p.canonicalize().ok() != out.canonicalize().ok())
                    .collect::<Vec<_>>()
            } else {
                vec![data]
            };
            objects.sort();
            ensure!(!objects.is_empty(), "No object directories found");
            let dtype = match cli.precision.as_str() {
                "fp16" => FloatDType::F16,
                "fp32" => FloatDType::F32,
                _ if !cli.cpu && device.supports_dtype(FloatDType::F16) => FloatDType::F16,
                _ => FloatDType::F32,
            };
            let warmup_lights = if let Some(limit) = image_limit {
                limit.max(1)
            } else if let Some(first_obj) = objects.first() {
                let count = std::fs::read_dir(first_obj)
                    .ok()
                    .map(|entries| {
                        entries
                            .filter_map(|e| e.ok().map(|e| e.path()))
                            .filter(|p| {
                                let name = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                                let ext = p.extension().unwrap_or_default().to_string_lossy().to_lowercase();
                                p.is_file()
                                    && !name.contains("mask")
                                    && !name.contains("normal")
                                    && matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff")
                            })
                            .count()
                    })
                    .unwrap_or(6);
                count.max(1)
            } else {
                6
            };
            let effective_encoder_batch = if cli.encoder_batch == 0 {
                warmup_lights.clamp(1, 12)
            } else {
                cli.encoder_batch.max(1)
            };
            let load_profile = unimsps::profile::phase(&device, || "model/load-and-upload".into());
            let net = Model::load_with_options(&model, device, ModelOptions {
                dtype,
                encoder_batch: effective_encoder_batch,
                pixel_batch: cli.pixel_batch,
                query_batch: cli.query_batch,
                fused_attention: cli.fused_attention && !cli.unfused_attention,
                score_budget_mib: cli.score_budget_mib,
                fused_softmax: !cli.burn_softmax,
                optimized_encoder: !cli.reference_encoder,
                gemm_convolutions: !cli.burn_convolutions,
            })?;
            drop(load_profile);
            eprintln!(
                "Precision: {dtype:?}; encoder batch: {}; pixel batch: {}; score budget: {} MiB; formats: {}; attention: {}",
                effective_encoder_batch,
                net.options.pixel_batch,
                net.options.score_budget_mib,
                save_formats.names().join(","),
                if net.options.fused_attention { "fused subgroup" } else { "hardware GEMM" }
            );
            let preparation_start = std::time::Instant::now();
            if !cli.no_warmup && !cli.cpu {
                eprintln!("Preparing shaders and process-local tuning for {warmup_lights} lights...");
                let warmup_profile = unimsps::profile::phase(&net.device, || "warmup/shaders-and-process-local-tuning".into());
                net.warmup(warmup_lights)?;
                drop(warmup_profile);
            }
            let preparation_seconds = preparation_start.elapsed().as_secs_f64();
            eprintln!("Preparation complete ({preparation_seconds:.2}s).");
            for object in objects {
                let out = if batch {
                    out.join(
                        object
                            .file_name()
                            .ok_or_else(|| anyhow::anyhow!("Object directory needs a name"))?,
                    )
                } else {
                    out.clone()
                };
                unimsps::trace::begin_view();
                let input_profile = unimsps::profile::phase(&net.device, || "input/decode-crop-normalize".into());
                let input = Input::load(&object, image_limit)?;
                drop(input_profile);
                eprintln!(
                    "{}: {} images; padded size {}",
                    object.display(),
                    input.images.len(),
                    input.size
                );
                let start = std::time::Instant::now();
                let inference_profile = unimsps::profile::phase(&net.device, || "inference/total".into());
                let normal = process_with_options(&net, &input,
                    ProcessOptions { image_cache_mib: cli.image_cache_mib })?;
                drop(inference_profile);
                let inference_seconds = start.elapsed().as_secs_f64();
                let restore_profile = unimsps::profile::phase(&net.device, || "output/restore-crop".into());
                let normal = input.restore(&normal);
                drop(restore_profile);
                let save_profile = unimsps::profile::phase(&net.device, || "output/encode-and-save".into());
                save_with_formats(&normal, &out, save_formats)?;
                drop(save_profile);
                let view_seconds = start.elapsed().as_secs_f64();
                let memory_after_view = net.device.memory_pool_usage().map(|u| serde_json::json!({
                    "reserved_bytes":u.bytes_reserved,"in_use_bytes":u.bytes_in_use,
                    "padding_bytes":u.bytes_padding,"allocations":u.number_allocs
                }));
                std::fs::write(
                    out.join("metadata.json"),
                    serde_json::to_vec_pretty(
                        &serde_json::json!({
                            "width":normal.w,"height":normal.h,"formats":save_formats.names(),
                            "normal_layout":"CHW float32 little endian","normal_axes":"upstream xyz, no axis flip",
                            "input_channels":"BGR","files":input.files,"model":model,"seconds":view_seconds,
                            "inference_seconds":inference_seconds,"output_seconds":view_seconds-inference_seconds,
                            "precision":format!("{dtype:?}"),"api":effective_api,"encoder_batch":effective_encoder_batch,
                            "pixel_batch":net.options.pixel_batch,"queue_depth":cli.queue_depth,"image_cache_mib":cli.image_cache_mib,
                            "score_budget_mib":cli.score_budget_mib,"fused_softmax":!cli.burn_softmax,
                            "optimized_encoder":!cli.reference_encoder,"gemm_convolutions":!cli.burn_convolutions,
                            "pool_cache_mib":cli.pool_cache_mib,"burn":"0.22.0","cubecl":"0.11.0","cubek":"0.3.0",
                            "images":input.images.len(),"padded_size":input.size,"git_revision":env!("UNIMSPS_GIT_REVISION"),
                            "preparation_seconds":preparation_seconds,
                            "memory_after_view":memory_after_view,"crop_policy":"upstream exclusive maxima"
                        }),
                    )?,
                )?;
                println!(
                    "Saved normals to {} ({:.2}s)",
                    out.display(),
                    view_seconds
                );
                unimsps::trace::write_report(&out.join("timing_trace.json"),serde_json::json!({
                    "images":input.images.len(),"padded_size":input.size,"queue_depth":cli.queue_depth,
                    "score_budget_mib":cli.score_budget_mib,"pool_cache_mib":cli.pool_cache_mib,
                    "encoder_batch":effective_encoder_batch,"precision":format!("{dtype:?}"),"api":effective_api,
                    "device":format!("{:?}",net.device.identity())}))?;
                unimsps::profile::write_report(&out.join("profile.json"), serde_json::json!({
                    "device": format!("{:?}", net.device.identity()), "api": effective_api,
                    "precision": format!("{dtype:?}"), "images": input.images.len(),
                    "padded_size": input.size, "encoder_batch": effective_encoder_batch,
                    "pixel_batch": net.options.pixel_batch, "query_batch": cli.query_batch,
                    "image_cache_mib": cli.image_cache_mib, "unfused_attention": !net.options.fused_attention,
                    "fused_attention": net.options.fused_attention, "score_budget_mib": net.options.score_budget_mib,
                    "queue_depth": cli.queue_depth, "formats": save_formats.names(),
                    "fused_softmax": net.options.fused_softmax, "optimized_encoder": net.options.optimized_encoder, "gemm_convolutions": net.options.gemm_convolutions, "pool_cache_mib": cli.pool_cache_mib,
                    "object": object, "model": model,
                }))?;
            }
        }
        Command::Verify {
            model,
            fixture,
            tolerance,
        } => {
            ensure!(
                tolerance > 0. && tolerance.is_finite(),
                "Tolerance must be positive"
            );
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(fixture.join("fixture.json"))?)?;
            let shape: Vec<usize> = serde_json::from_value(meta["shape"].clone())?;
            let shape: [usize; 4] = shape
                .try_into()
                .map_err(|_| anyhow::anyhow!("Expected rank 4"))?;
            let read = |p: &str| -> Result<Vec<f32>> {
                let b = std::fs::read(fixture.join(p))?;
                ensure!(b.len() % 4 == 0, "Invalid floats");
                Ok(b.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect())
            };
            let values = read("input.f32")?;
            ensure!(
                values.len() == shape.iter().product::<usize>(),
                "Input length mismatch"
            );
            let expected = read("normal.f32")?;
            let net = Model::load(&model, device, cli.pixel_batch, cli.query_batch)?;
            let actual = read_tensor(net.forward(
                tensor(values, shape, &net.device),
                meta["refinement"].as_bool().unwrap_or(false),
            )?);
            ensure!(actual.len() == expected.len(), "Output length mismatch");
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let mae = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs() as f64)
                .sum::<f64>()
                / actual.len() as f64;
            println!("max_abs={max:.8} mean_abs={mae:.8}");
            ensure!(
                actual.iter().all(|v| v.is_finite()) && max <= tolerance,
                "Parity failed: max={max}, tolerance={tolerance}"
            );
        }
    }
    let total_secs = program_start.elapsed().as_secs_f64();
    eprintln!("Total run time: {total_secs:.2}s");
    println!("Total run time: {total_secs:.2}s");
    Ok(())
}
