use crate::model::{Model, read_tensor, tensor};
use anyhow::{Context, Result, ensure};
use burn::tensor::{Bool, Tensor, TensorData};
use burn_backend::{DType, ExecutionError};
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
    ops::numeric::empty_device_contiguous_dtype,
};
use burn_fusion::{
    Fusion,
    custom::{CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId, TensorIr},
};
use cubecl::prelude::*;
use image::{ImageBuffer, Rgb};
use std::path::{Path, PathBuf};

type Gpu = Fusion<CubeBackend>;

/// Host CHW image; preprocessing keeps the reference's BGR channel convention.
#[derive(Clone)]
pub struct Raster {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}
impl Raster {
    pub fn zeros(c: usize, h: usize, w: usize) -> Self {
        Self {
            c,
            h,
            w,
            data: vec![0.; c * h * w],
        }
    }
    pub fn crop(&self, y: usize, x: usize, h: usize, w: usize) -> Self {
        let mut out = Self::zeros(self.c, h, w);
        for c in 0..self.c {
            for j in 0..h {
                out.data[(c * h + j) * w..(c * h + j + 1) * w].copy_from_slice(
                    &self.data
                        [(c * self.h + y + j) * self.w + x..(c * self.h + y + j) * self.w + x + w],
                );
            }
        }
        out
    }
    /// PyTorch bilinear coordinates, including align_corners=True for normal maps.
    pub fn resize(&self, h: usize, w: usize, align: bool) -> Self {
        let mut out = Self::zeros(self.c, h, w);
        // Fast path for exact 2x pyramid downsampling with align_corners=False:
        // coord(i, 2*dst, dst) = 2*i + 0.5, so dx = dy = 0.5 everywhere.
        if !align && self.h == h * 2 && self.w == w * 2 {
            for c in 0..self.c {
                let src_c = &self.data[c * self.h * self.w..(c + 1) * self.h * self.w];
                let dst_c = &mut out.data[c * h * w..(c + 1) * h * w];
                for y in 0..h {
                    let row0 = &src_c[(2 * y) * self.w..(2 * y + 1) * self.w];
                    let row1 = &src_c[(2 * y + 1) * self.w..(2 * y + 2) * self.w];
                    let dst_row = &mut dst_c[y * w..(y + 1) * w];
                    for x in 0..w {
                        let x2 = 2 * x;
                        dst_row[x] = (row0[x2] * 0.5 + row0[x2 + 1] * 0.5) * 0.5
                            + (row1[x2] * 0.5 + row1[x2 + 1] * 0.5) * 0.5;
                    }
                }
            }
            return out;
        }
        let coord = |i: usize, src: usize, dst: usize| {
            if align {
                if dst == 1 {
                    0.
                } else {
                    i as f32 * (src - 1) as f32 / (dst - 1) as f32
                }
            } else {
                ((i as f32 + 0.5) * src as f32 / dst as f32 - 0.5).max(0.)
            }
        };
        for y in 0..h {
            let fy = coord(y, self.h, h);
            let y0 = (fy.floor() as usize).min(self.h - 1);
            let y1 = (y0 + 1).min(self.h - 1);
            let dy = fy - y0 as f32;
            for x in 0..w {
                let fx = coord(x, self.w, w);
                let x0 = (fx.floor() as usize).min(self.w - 1);
                let x1 = (x0 + 1).min(self.w - 1);
                let dx = fx - x0 as f32;
                for c in 0..self.c {
                    let at = |yy, xx| self.data[(c * self.h + yy) * self.w + xx];
                    out.data[(c * h + y) * w + x] = (at(y0, x0) * (1. - dx) + at(y0, x1) * dx)
                        * (1. - dy)
                        + (at(y1, x0) * (1. - dx) + at(y1, x1) * dx) * dy;
                }
            }
        }
        out
    }
    pub fn normalize(&mut self) {
        let n = self.h * self.w;
        for i in 0..n {
            let d = (0..3)
                .map(|c| self.data[c * n + i].powi(2))
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            for c in 0..3 {
                self.data[c * n + i] /= d;
            }
        }
    }
}
pub struct Input {
    pub images: Vec<Raster>,
    pub mask: Vec<bool>,
    pub size: usize,
    pub original: [usize; 2],
    pub crop: [usize; 4],
    pub padding: [usize; 2],
    pub files: Vec<PathBuf>,
}
impl Input {
    /// Matches the upstream exclusive maximum mask bounding box, including its omitted last row/column.
    pub fn load(path: &Path, limit: Option<usize>) -> Result<Self> {
        ensure!(limit != Some(0), "Image count must be positive");
        let mut files: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let name = p
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_lowercase();
                let ext = p
                    .extension()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_lowercase();
                p.is_file()
                    && !name.contains("mask")
                    && !name.contains("normal")
                    && matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff")
            })
            .collect();
        // Stable order and deterministic subset; the Python command randomly chooses subsets.
        files.sort();
        if let Some(n) = limit {
            files.truncate(n);
        }
        ensure!(!files.is_empty(), "No input images in {}", path.display());
        let first = image::open(&files[0])?;
        let (w, h) = (first.width() as usize, first.height() as usize);
        let mp = path.join("mask.png");
        let mask: Vec<bool> = if mp.exists() {
            let m = image::open(mp)?.to_rgb8();
            ensure!(
                m.width() as usize == w && m.height() as usize == h,
                "Mask dimensions differ"
            );
            m.pixels().map(|p| p[2] > 0).collect()
        } else {
            vec![true; h * w]
        };
        let mut ymin = h;
        let mut xmin = w;
        let mut ymax = 0;
        let mut xmax = 0;
        for y in 0..h {
            for x in 0..w {
                if mask[y * w + x] {
                    ymin = ymin.min(y);
                    xmin = xmin.min(x);
                    ymax = ymax.max(y);
                    xmax = xmax.max(x);
                }
            }
        }
        ensure!(
            ymax > ymin && xmax > xmin,
            "Mask must span at least two rows and columns"
        );
        let ch = ymax - ymin;
        let cw = xmax - xmin;
        let size = ch.max(cw).max(32).next_power_of_two();
        let py = (size - ch) / 2;
        let px = (size - cw) / 2;
        let mut padded_mask = vec![false; size * size];
        for y in 0..ch {
            for x in 0..cw {
                padded_mask[(y + py) * size + x + px] = mask[(y + ymin) * w + x + xmin];
            }
        }
        use rayon::prelude::*;
        let images = files
            .par_iter()
            .map(|file| -> Result<Raster> {
                let img = image::open(file)
                    .with_context(|| format!("Decode {}", file.display()))?
                    .to_rgb32f();
                ensure!(
                    img.width() as usize == w && img.height() as usize == h,
                    "Image dimensions differ: {}",
                    file.display()
                );
                let mut r = Raster::zeros(3, size, size);
                let mut sum = 0f64;
                let mut count = 0usize;
                for y in 0..ch {
                    for x in 0..cw {
                        let p = img.get_pixel((x + xmin) as u32, (y + ymin) as u32);
                        let idx = (y + py) * size + x + px;
                        let is_fg = padded_mask[idx];
                        for c in 0..3 {
                            let v = p[2 - c];
                            r.data[c * size * size + idx] = v;
                            if is_fg {
                                sum += v as f64;
                                count += 1;
                            }
                        }
                    }
                }
                ensure!(
                    sum.is_finite() && sum > 0. && count > 0,
                    "Image has zero/invalid masked intensity: {}",
                    file.display()
                );
                let inv_mean = (count as f64 / sum) as f32;
                for v in &mut r.data {
                    *v *= inv_mean;
                }
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            images,
            mask: padded_mask,
            size,
            original: [h, w],
            crop: [ymin, xmin, ch, cw],
            padding: [py, px],
            files,
        })
    }
    pub fn restore(&self, normal: &Raster) -> Raster {
        let [h, w] = self.original;
        let [y, x, ch, cw] = self.crop;
        let [py, px] = self.padding;
        let mut out = Raster::zeros(3, h, w);
        for c in 0..3 {
            for j in 0..ch {
                for i in 0..cw {
                    out.data[(c * h + y + j) * w + x + i] =
                        normal.data[(c * self.size + py + j) * self.size + px + i];
                }
            }
        }
        out.normalize();
        out
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProcessOptions {
    /// Budget for cached RGB inputs/pyramids, excluding model weights and normal maps.
    /// Larger scales stream RGB patches when they exceed this budget.
    pub image_cache_mib: usize,
}
impl Default for ProcessOptions {
    fn default() -> Self { Self { image_cache_mib: 512 } }
}

enum ImageLevel<'a> {
    Device(Tensor<4>),
    Host(std::borrow::Cow<'a, [Raster]>),
}

fn upload_images(model: &Model, images: &[Raster]) -> Tensor<4> {
    use rayon::prelude::*;
    let _profile = crate::profile::span(&model.device, || format!("input/upload images={} size={}x{}", images.len(), images[0].h, images[0].w));
    let h = images[0].h;
    let w = images[0].w;
    let shape = [images.len(), 3, h, w];
    // One transfer, already converted in parallel on the CPU to the target precision.
    if model.options.dtype == burn::tensor::FloatDType::F16 {
        let values: Vec<half::f16> = images
            .par_iter()
            .flat_map_iter(|im| im.data.iter().map(|&v| half::f16::from_f32(v)))
            .collect();
        Tensor::from_data(TensorData::new(values, shape), model.tensor_options())
    } else {
        let values: Vec<f32> = images.iter().flat_map(|im| im.data.iter().copied()).collect();
        tensor(values, shape, &model.device)
    }
}

#[cube(launch)]
fn upsample2x_align_kernel<F: Float>(
    input: &cubecl::prelude::Tensor<F>,
    output: &mut cubecl::prelude::Tensor<F>,
    #[comptime] channels: usize,
) {
    let idx = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * usize::cast_from(CUBE_DIM)
        + usize::cast_from(UNIT_POS);
    let out_w = output.shape(3);
    let out_h = output.shape(2);
    let in_w = input.shape(3);
    let in_h = input.shape(2);
    let total_pixels = out_h * out_w;
    let total = channels * total_pixels;

    if idx < total {
        let c = idx / total_pixels;
        let p_idx = idx % total_pixels;
        let y = p_idx / out_w;
        let x = p_idx % out_w;

        let fy = (y as f32) * ((in_h - 1) as f32) / ((out_h - 1) as f32);
        let fx = (x as f32) * ((in_w - 1) as f32) / ((out_w - 1) as f32);

        let y0 = usize::min(fy as usize, in_h - 1);
        let y1 = usize::min(y0 + 1, in_h - 1);
        let dy = fy - (y0 as f32);

        let x0 = usize::min(fx as usize, in_w - 1);
        let x1 = usize::min(x0 + 1, in_w - 1);
        let dx = fx - (x0 as f32);

        let in_c_stride = input.stride(1);
        let in_h_stride = input.stride(2);
        let in_base = c * in_c_stride;

        let v00 = f32::cast_from(input[in_base + y0 * in_h_stride + x0]);
        let v01 = f32::cast_from(input[in_base + y0 * in_h_stride + x1]);
        let v10 = f32::cast_from(input[in_base + y1 * in_h_stride + x0]);
        let v11 = f32::cast_from(input[in_base + y1 * in_h_stride + x1]);

        let top = v00 * (1.0 - dx) + v01 * dx;
        let bottom = v10 * (1.0 - dx) + v11 * dx;
        let val = top * (1.0 - dy) + bottom * dy;

        let out_c_stride = output.stride(1);
        let out_h_stride = output.stride(2);
        output[c * out_c_stride + y * out_h_stride + x] = F::cast_from(val);
    }
}

#[derive(Debug)]
struct Upsample2xOp {
    desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for Upsample2xOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([input], [output]) = self.desc.as_fixed();
        let input = handles.get_float_tensor::<CubeBackend>(input);
        let [b, c, h, w] = input.meta.shape().dims::<4>();
        let client = input.client.clone();
        let hw = &client.properties().hardware;
        let total = c * (2 * h) * (2 * w);
        let groups = total.div_ceil(128);
        let gy = groups.div_ceil((hw.max_cube_count.0 as usize).max(1)).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic("Upsample2x dispatch exceeds device grid limits"));
        }
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        let out_shape = [b, c, 2 * h, 2 * w];
        let result = empty_device_contiguous_dtype(
            client.clone(),
            input.device.clone(),
            burn::tensor::Shape::new(out_shape),
            input.dtype,
        );
        match input.dtype {
            DType::F16 => upsample2x_align_kernel::launch::<half::f16>(
                &client, count, dim,
                input.into_tensor_arg(),
                result.clone().into_tensor_arg(),
                c,
            ),
            DType::F32 => upsample2x_align_kernel::launch::<f32>(
                &client, count, dim,
                input.into_tensor_arg(),
                result.clone().into_tensor_arg(),
                c,
            ),
            _ => return Err(ExecutionError::generic("Unsupported upsample dtype")),
        }
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}

fn resize_normal_2x(x: Tensor<4>, h: usize, w: usize) -> Tensor<4> {
    let [b, c, in_h, in_w] = x.dims();
    if h == in_h * 2 && w == in_w * 2 && b == 1 {
        if let Ok(prim) = x.clone().inner().try_into_primitive::<Gpu>() {
            let client = prim.client.clone();
            let out_shape = burn::tensor::Shape::new([b, c, h, w]);
            let out = TensorIr::uninit(client.create_empty_handle(), out_shape, prim.dtype);
            let desc = CustomOpIr::with_scalars(
                "unimsps_upsample2x_align",
                &[prim.into_ir()],
                &[out],
                vec![],
            );
            let [out] = client
                .register(StreamId::current(), OperationIr::Custom(desc.clone()), Upsample2xOp { desc })
                .outputs();
            return Tensor::from_primitive::<Gpu>(out);
        }
    }
    resize_device(x, h, w, true)
}

fn resize_device(x: Tensor<4>, h: usize, w: usize, align_corners: bool) -> Tensor<4> {
    burn::tensor::module::interpolate(x,
        burn::tensor::ops::InterpolateOptions::new(burn::tensor::ops::InterpolateMode::Bilinear)
            .with_align_corners(align_corners).with_output_size([h, w]))
}

fn stage_device(
    model: &Model,
    images: Tensor<4>,
    normal: Option<Tensor<4>>,
    keep_mask: Tensor<4, Bool>,
    host_mask: Option<&[bool]>,
) -> Result<Tensor<4>> {
    let lights = images.dims()[0];
    let (images, refinement) = if let Some(normal) = normal {
        (Tensor::cat(vec![images, normal.cast(model.options.dtype).repeat_dim(0, lights)], 1), true)
    } else {
        let is_6ch = images.dims()[1] == 6;
        (images, is_6ch)
    };
    let raw = model.forward_masked(images, refinement, host_mask)?;
    Ok(raw.mask_fill(keep_mask.bool_not(), 0.))
}

/// Four global stages followed by 256px patches/128px stride.
/// Normals, interpolation, masking and patch folding remain on the device.
/// Only the completed normal map is read back.
pub fn process(model: &Model, input: &Input) -> Result<Raster> {
    process_with_options(model, input, ProcessOptions::default())
}

pub fn process_with_options(model: &Model, input: &Input, options: ProcessOptions) -> Result<Raster> {
    // The inner scope owns all per-view GPU tensors. Reclaim only after they
    // have dropped, including the cached boundary masks and Gaussian weights.
    let result = process_view(model, input, options);
    crate::execution::cleanup_view(&model.device);
    result
}

fn process_view(model: &Model, input: &Input, options: ProcessOptions) -> Result<Raster> {
    use std::borrow::Cow;
    use rayon::prelude::*;

    let prepare_profile = crate::profile::phase(&model.device, || "pyramid/build-inputs-masks-and-weights".into());
    let stages = input.size.ilog2() as usize - 4;
    let budget = (options.image_cache_mib as u64).saturating_mul(1024 * 1024);
    let element_bytes = if model.options.dtype == burn::tensor::FloatDType::F16 { 2u64 } else { 4u64 };
    let rgb_bytes = |size: usize| (input.images.len() as u64)
        .saturating_mul(3).saturating_mul(size as u64)
        .saturating_mul(size as u64).saturating_mul(element_bytes);
    // Build the RGB image pyramid on the host using parallel 2x2 bilinear box downsampling,
    // avoiding 5 multi-scale WGPU bilinear shader compilations (~6.5s) and keeping only the
    // active scale resident in VRAM at a time.
    let mut levels = vec![ImageLevel::Host(Cow::Borrowed(input.images.as_slice()))];
    let mut masks = vec![input.mask.clone()];
    let mut size = input.size;
    while size > 32 {
        let next = match levels.last().unwrap() {
            ImageLevel::Device(x) => ImageLevel::Device(resize_device(x.clone(), size / 2, size / 2, false)),
            ImageLevel::Host(images) => ImageLevel::Host(Cow::Owned(
                images.par_iter().map(|r| r.resize(size / 2, size / 2, false)).collect())),
        };
        levels.push(next);
        let previous = masks.last().unwrap();
        let mut mask = vec![false; size * size / 4];
        for y in 0..size / 2 {
            for x in 0..size / 2 { mask[y * (size / 2) + x] = previous[2 * y * size + 2 * x]; }
        }
        masks.push(mask);
        size /= 2;
    }

    // Cache the nine boundary-mask cases and Gaussian once, instead of rebuilding
    // or transferring a 256x256 mask and weight image for each patch.
    let mut interiors = Vec::new();
    let mut gaussian = None;
    if stages > 4 {
        for row in 0..3 { for col in 0..3 {
            let mut values = vec![false; 256 * 256];
            for y in 0..256 { for x in 0..256 {
                if (y >= 32 || row == 0) && (y < 224 || row == 2)
                    && (x >= 32 || col == 0) && (x < 224 || col == 2) {
                    values[y * 256 + x] = true;
                }
            }}
            interiors.push(Tensor::<4, Bool>::from_data(
                TensorData::new(values, [1, 1, 256, 256]), &model.device));
        }}
        let g: Vec<f64> = (0..256).map(|v| (-0.5 * ((v as f64 - 127.5) / 51.2).powi(2)).exp()).collect();
        let denominator = g.iter().sum::<f64>().powi(2);
        let mut values = Vec::with_capacity(256 * 256);
        for y in 0..256 { for x in 0..256 {
            values.push(half::f16::from_f64(g[y] * g[x] / denominator).to_f32());
        }}
        gaussian = Some(tensor(values, [1, 1, 256, 256], &model.device));
    }

    drop(prepare_profile);
    let mut normal: Option<Tensor<4>> = None;
    let mut final_mask = None;
    for i in 0..stages {
        let size = 32 << i;
        let _scale_profile = crate::profile::phase(&model.device, || format!("scale/{} size={size}", i + 1));
        let active = match levels.pop().context("Missing image scale")? {
            ImageLevel::Host(images) if i < 4 || rgb_bytes(size) <= budget => {
                ImageLevel::Device(upload_images(model, images.as_ref()))
            }
            level => level,
        };
        let host_mask = masks.pop().context("Missing mask scale")?;
        let mask = Tensor::<4, Bool>::from_data(TensorData::new(
            host_mask.clone(), [1, 1, size, size]), &model.device);
        eprintln!("Scale {}/{}: {} x {} ({})", i + 1, stages, size, size,
            if matches!(&active, ImageLevel::Device(_)) { "RGB cached on device" } else { "streaming RGB patches" });
        let prior_profile = crate::profile::span(&model.device, || format!("normal/resize size={size}"));
        let prior = normal.take().map(|n| crate::model::normalize(resize_normal_2x(n, size, size))
            .cast(model.options.dtype));
        drop(prior_profile);
        normal = Some(if i < 4 {
            let ImageLevel::Device(images) = active else { unreachable!() };
            let res = stage_device(model, images, prior, mask.clone(), Some(&host_mask))?;
            crate::execution::trim_memory(&model.device);
            res
        } else {
            let prior = prior.context("Refinement requires previous normals")?;
            let mut out = Tensor::zeros([1, 3, size, size], &model.device);
            let total = ((size - 256) / 128 + 1).pow(2);
            let mut count = 0;
            let mut submitted = 0;
            let mut patch_host_mask = vec![false; 256 * 256];
            let is_final_scale = i + 1 == stages;
            let lights = input.images.len();
            let full_device_images = match &active {
                ImageLevel::Device(images) => Some(Tensor::cat(
                    vec![images.clone(), prior.clone().repeat_dim(0, lights)],
                    1,
                )),
                ImageLevel::Host(_) => None,
            };
            for y in (0..=size - 256).step_by(128) {
                for x in (0..=size - 256).step_by(128) {
                    count += 1;
                    let mut fg_count = 0usize;
                    for py in 0..256 {
                        let src_row = &host_mask[(y + py) * size + x..(y + py) * size + x + 256];
                        let dst_row = &mut patch_host_mask[py * 256..(py + 1) * 256];
                        dst_row.copy_from_slice(src_row);
                        fg_count += src_row.iter().filter(|&&m| m).count();
                    }
                    if fg_count == 0 {
                        let can_skip = if is_final_scale {
                            true
                        } else {
                            // At intermediate patch scales (e.g. 512x512 before 1024x1024), a non-empty
                            // 256x256 patch at the next scale can read `prior` up to 128px outside the
                            // foreground at the current scale. Skip only if that 128px halo is also empty.
                            let y0 = y.saturating_sub(128);
                            let y1 = (y + 384).min(size);
                            let x0 = x.saturating_sub(128);
                            let x1 = (x + 384).min(size);
                            let mut any_near = false;
                            for hy in y0..y1 {
                                if host_mask[hy * size + x0..hy * size + x1].iter().any(|&m| m) {
                                    any_near = true;
                                    break;
                                }
                            }
                            !any_near
                        };
                        if can_skip {
                            eprintln!("  Patch {count}/{total} skipped (empty mask)");
                            continue;
                        }
                    }
                    let _patch_profile = crate::profile::phase(&model.device, || format!("patch/size={size} y={y} x={x} foreground={fg_count}"));
                    let prepare_patch_profile = crate::profile::span(&model.device, || format!("patch/prepare size={size}"));
                    let (images, n_opt) = match &full_device_images {
                        Some(full) => {
                            let patch = full.clone().slice([0..lights, 0..6, y..y + 256, x..x + 256]);
                            (patch, None)
                        }
                        None => {
                            let ImageLevel::Host(images) = &active else { unreachable!() };
                            let patches: Vec<_> = images.par_iter().map(|r| r.crop(y, x, 256, 256)).collect();
                            let imgs = upload_images(model, &patches);
                            let n = prior.clone().slice([0..1, 0..3, y..y + 256, x..x + 256]);
                            (imgs, Some(n))
                        }
                    };
                    let row = if y == 0 { 0 } else if y + 256 == size { 2 } else { 1 };
                    let col = if x == 0 { 0 } else if x + 256 == size { 2 } else { 1 };
                    // Preserve upstream's border-exclusion * background-mask rule.
                    let patch_mask = mask.clone().slice([0..1, 0..1, y..y + 256, x..x + 256])
                        .bool_or(interiors[row * 3 + col].clone());
                    drop(prepare_patch_profile);
                    // At the final scale, only `patch_host_mask` pixels survive `final_mask`, so
                    // `pool_block` only needs the U-Net decoder dependency cone of `patch_host_mask`.
                    // At intermediate patch scales, the 192x192 interior is kept for the next scale's prior.
                    let host_mask_slice = if is_final_scale && fg_count * 10 < 256 * 256 * 9 {
                        Some(patch_host_mask.as_slice())
                    } else {
                        None
                    };
                    let prediction = stage_device(model, images, n_opt, patch_mask, host_mask_slice)?;
                    let blend_profile = crate::profile::span(&model.device, || format!("patch/blend size={size}"));
                    let weighted = prediction * gaussian.as_ref().unwrap().clone();
                    let region = [0..1, 0..3, y..y + 256, x..x + 256];
                    let updated = out.clone().slice(region.clone()) + weighted;
                    out = out.slice_assign(region, updated);
                    drop(blend_profile);
                    // Submit outstanding fusion work and bound pool reservation without per-chunk stalls.
                    model.device.flush().map_err(|e| anyhow::anyhow!("GPU submission failed: {e:?}"))?;
                    submitted += 1;
                    crate::execution::checkpoint_patch(&model.device, submitted);
                    eprintln!("  Patch {count}/{total} {}", if crate::execution::depth() != 0 || crate::profile::enabled() { "completed" } else { "queued" });
                }
            }
            crate::execution::checkpoint_patch(&model.device, submitted);
            crate::execution::trim_memory(&model.device);
            out
        });
        if i + 1 == stages { final_mask = Some(mask); }
    }
    let _readback_profile = crate::profile::phase(&model.device, || "output/normalize-mask-and-readback".into());
    let normal = crate::model::normalize(normal.context("Empty image pyramid")?)
        .mask_fill(final_mask.context("Missing final mask")?.bool_not(), 0.);
    let data = read_tensor(normal);
    Ok(Raster { c: 3, h: input.size, w: input.size, data })
}
#[derive(Clone, Copy, Debug)]
pub struct SaveFormats {
    pub exr: bool,
    pub png: bool,
    pub mat: bool,
    pub f32: bool,
}

impl Default for SaveFormats {
    fn default() -> Self {
        Self {
            exr: true,
            png: true,
            mat: false,
            f32: false,
        }
    }
}

impl SaveFormats {
    pub fn parse(specs: &[String]) -> Result<Self> {
        let mut out = Self {
            exr: false,
            png: false,
            mat: false,
            f32: false,
        };
        for item in specs {
            for token in item.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                match token.to_ascii_lowercase().as_str() {
                    "exr" | ".exr" => out.exr = true,
                    "png" | ".png" => out.png = true,
                    "mat" | ".mat" => out.mat = true,
                    "f32" | ".f32" | "raw" => out.f32 = true,
                    "all" => {
                        out.exr = true;
                        out.png = true;
                        out.mat = true;
                        out.f32 = true;
                    }
                    other => anyhow::bail!(
                        "Unknown normal format '{other}' (supported: exr, png, mat, f32, all)"
                    ),
                }
            }
        }
        ensure!(
            out.exr || out.png || out.mat || out.f32,
            "At least one output format must be specified"
        );
        Ok(out)
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.exr {
            v.push("exr");
        }
        if self.png {
            v.push("png");
        }
        if self.mat {
            v.push("mat");
        }
        if self.f32 {
            v.push("f32");
        }
        v
    }
}

pub fn save(normal: &Raster, out: &Path) -> Result<()> {
    save_with_formats(normal, out, SaveFormats::default())
}

pub fn save_with_formats(normal: &Raster, out: &Path, formats: SaveFormats) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let (h, w) = (normal.h, normal.w);
    let n = h * w;
    ensure!(
        normal.data.iter().all(|v| v.is_finite()),
        "Non-finite output normals"
    );
    let (res_exr, res_png) = rayon::join(
        || -> Result<()> {
            if !formats.exr {
                return Ok(());
            }
            let mut exr = ImageBuffer::<Rgb<f32>, Vec<f32>>::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let p = [normal.data[i], normal.data[n + i], normal.data[2 * n + i]];
                    exr.put_pixel(x as u32, y as u32, Rgb(p));
                }
            }
            exr.save(out.join("normal.exr"))?;
            Ok(())
        },
        || -> Result<()> {
            if !formats.png {
                return Ok(());
            }
            let mut rgb = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let p = [normal.data[i], normal.data[n + i], normal.data[2 * n + i]];
                    rgb.put_pixel(
                        x as u32,
                        y as u32,
                        Rgb(p.map(|v| ((v + 1.) * 127.5).clamp(0., 255.) as u8)),
                    );
                }
            }
            rgb.save(out.join("normal.png"))?;
            Ok(())
        },
    );
    res_exr?;
    res_png?;
    if formats.mat {
        save_mat(normal, &out.join("normal.mat"))?;
    }
    if formats.f32 {
        // Portable raw output is CHW float32; dimensions/convention are in metadata.json.
        let bytes: Vec<u8> = normal.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join("normal.f32"), bytes)?;
    }
    Ok(())
}

// MATLAB v5, uncompressed real single matrix named Normal_est, dimensions H x W x 3.
fn save_mat(normal: &Raster, path: &Path) -> Result<()> {
    fn element(out: &mut Vec<u8>, kind: u32, data: &[u8]) {
        out.extend(kind.to_le_bytes());
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
        while out.len() % 8 != 0 {
            out.push(0);
        }
    }
    ensure!(
        normal.h <= i32::MAX as usize
            && normal.w <= i32::MAX as usize
            && normal.data.len() < (u32::MAX as usize - 256) / 4,
        "Normal map exceeds MAT v5 element limit"
    );
    let mut out = vec![b' '; 116];
    let label = b"MATLAB 5.0 MAT-file, Uni-MS-PS Rust";
    out[..label.len()].copy_from_slice(label);
    out.extend([0u8; 8]);
    out.extend([0, 1, b'I', b'M']);
    let mut matrix = Vec::new();
    element(&mut matrix, 6, &[7, 0, 0, 0, 0, 0, 0, 0]); // mxSINGLE_CLASS
    let dimensions: Vec<u8> = [normal.h as i32, normal.w as i32, 3]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    element(&mut matrix, 5, &dimensions);
    element(&mut matrix, 1, b"Normal_est");
    let mut values = Vec::with_capacity(normal.data.len() * 4);
    for c in 0..3 {
        for x in 0..normal.w {
            for y in 0..normal.h {
                values.extend(normal.data[(c * normal.h + y) * normal.w + x].to_le_bytes());
            }
        }
    }
    element(&mut matrix, 7, &values);
    element(&mut out, 14, &matrix);
    std::fs::write(path, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resize_coordinates() {
        let r = Raster {
            c: 1,
            h: 2,
            w: 2,
            data: vec![0., 2., 4., 6.],
        };
        assert_eq!(r.resize(1, 1, false).data, vec![3.]);
        assert_eq!(
            r.resize(3, 3, true).data,
            vec![0., 1., 2., 2., 3., 4., 4., 5., 6.]
        );
    }
    #[test]
    fn zero_normals_stay_zero() {
        let mut r = Raster::zeros(3, 2, 2);
        r.normalize();
        assert_eq!(r.data, vec![0.; 12]);
    }
}
