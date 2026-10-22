use anyhow::{Result, ensure};
use burn::tensor::{Device, DeviceKind, FloatDType};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use unimsps::{
    model::{Model, ModelOptions, read_tensor, tensor},
    pipeline::{Input, ProcessOptions, process_with_options, save},
};

#[derive(Parser)]
#[command(about = "Uni-MS-PS normal estimation using Burn 0.22")]
struct Cli {
    /// Detailed synchronized CPU/GPU wall timings; writes profile.json. Slows execution.
    #[arg(long, short = 'v', global = true)]
    verbose: bool,
    /// Maximum chunks queued between synchronous GPU waits; 0 uses asynchronous submission.
    #[arg(long, default_value_t = 0, global = true)]
    queue_depth: usize,
    #[arg(long)]
    cpu: bool,
    #[arg(long, conflicts_with = "cpu")]
    integrated: bool,
    /// Auto selects WGPU on every build. CUDA is only used when explicitly requested.
    #[arg(long,default_value="auto",value_parser=["auto","cuda","vulkan","dx12","metal"])]
    api: String,
    /// Set-transformer pixel batch size (defaults to power-of-two 4096 to avoid shape explosion).
    #[arg(long, default_value_t = 4096)]
    pixel_batch: usize,
    /// Spatial attention query limit; 0 uses one fused dispatch. Unfused scores remain bounded.
    #[arg(long, default_value_t = 0)]
    query_batch: usize,
    #[arg(long, default_value_t = 8)]
    encoder_batch: usize,
    /// Auto selects FP16 on supporting GPUs, FP32 on CPU/other adapters.
    #[arg(long, default_value = "auto", value_parser = ["auto", "fp16", "fp32"])]
    precision: String,
    /// RGB input/pyramid cache budget, excluding model weights and normal maps.
    #[arg(long, default_value_t = 512)]
    image_cache_mib: usize,
    /// Use explicit matmul/softmax spatial attention instead of fused SDPA dispatch.
    #[arg(long)]
    unfused_attention: bool,
    /// Skip pipeline and GPU kernel pre-warm pass before timed inference.
    #[arg(long)]
    no_warmup: bool,
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
        #[arg(long, visible_aliases = ["nb-img", "nb_img"], default_value_t = -1, allow_hyphen_values = true)]
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
    let cli = Cli::parse();
    eprintln!("unimsps 0.1.1 (WGPU fused-subgroup spatial/light/norm + static-bucket build)");
    unimsps::profile::configure(cli.verbose);
    unimsps::execution::configure(cli.queue_depth);
    ensure!(!(cli.integrated && cli.api == "cuda"), "--integrated cannot select CUDA");
    let use_cuda = cli.api == "cuda";
    let device = if cli.cpu {
        Device::flex()
    } else if use_cuda {
        #[cfg(feature = "cuda")]
        { Device::cuda(0) }
        #[cfg(not(feature = "cuda"))]
        { anyhow::bail!("CUDA support requires rebuilding with cargo build --release --features cuda"); }
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
                _ => WgpuBackend::Auto,
            })
            .init()
            .map_err(|e| anyhow::anyhow!("GPU initialization: {e:?}"))?
    };
    eprintln!("Device: {:?}", device.identity());
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
            let load_profile = unimsps::profile::phase(&device, || "model/load-and-upload".into());
            let net = Model::load_with_options(&model, device, ModelOptions {
                dtype,
                encoder_batch: cli.encoder_batch,
                pixel_batch: cli.pixel_batch,
                query_batch: cli.query_batch,
                fused_attention: !cli.unfused_attention,
            })?;
            drop(load_profile);
            eprintln!("Precision: {dtype:?}; encoder batch: {}; pixel batch: {}", cli.encoder_batch, net.options.pixel_batch);
            if !cli.no_warmup && !cli.cpu {
                eprintln!("Pre-warming GPU kernels (spatial attention, GEMMs, norms, decoders)...");
                let warmup_profile = unimsps::profile::phase(&net.device, || "warmup/pipeline".into());
                net.warmup(cli.encoder_batch)?;
                drop(warmup_profile);
                eprintln!("Warmup complete.");
            }
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
                let restore_profile = unimsps::profile::phase(&net.device, || "output/restore-crop".into());
                let normal = input.restore(&normal);
                drop(restore_profile);
                let save_profile = unimsps::profile::phase(&net.device, || "output/encode-and-save".into());
                save(&normal, &out)?;
                drop(save_profile);
                std::fs::write(
                    out.join("metadata.json"),
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"width":normal.w,"height":normal.h,"normal_layout":"CHW float32 little endian","normal_axes":"upstream xyz, no axis flip","input_channels":"BGR","files":input.files,"model":model,"seconds":start.elapsed().as_secs_f64(),"precision":format!("{dtype:?}"),"encoder_batch":cli.encoder_batch,"pixel_batch":net.options.pixel_batch,"queue_depth":cli.queue_depth,"image_cache_mib":cli.image_cache_mib,"burn":"0.22.0","crop_policy":"upstream exclusive maxima"}),
                    )?,
                )?;
                println!(
                    "Saved normals to {} ({:.2}s)",
                    out.display(),
                    start.elapsed().as_secs_f64()
                );
                unimsps::profile::write_report(&out.join("profile.json"), serde_json::json!({
                    "device": format!("{:?}", net.device.identity()), "api": cli.api,
                    "precision": format!("{dtype:?}"), "images": input.images.len(),
                    "padded_size": input.size, "encoder_batch": cli.encoder_batch,
                    "pixel_batch": net.options.pixel_batch, "query_batch": cli.query_batch,
                    "image_cache_mib": cli.image_cache_mib, "unfused_attention": cli.unfused_attention,
                    "queue_depth": cli.queue_depth,
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
    Ok(())
}
