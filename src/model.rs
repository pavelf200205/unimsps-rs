use anyhow::{Context, Result, ensure};
use burn::tensor::{
    Device, FloatDType, Tensor, TensorCreationOptions, TensorData, activation, module,
    ops::{ConvOptions, ConvTransposeOptions},
};
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone)]
enum Weight {
    R1(Tensor<1>),
    R2(Tensor<2>),
    R3(Tensor<3>),
    R4(Tensor<4>),
}

struct ActiveMask {
    gather_idx: Tensor<1, burn::tensor::Int>,
    inverse_idx: Tensor<1, burn::tensor::Int>,
    valid_len: usize,
    padded_len: usize,
}

fn effective_pixel_batch(requested: usize) -> usize {
    // Map legacy 5000 / 4096 chunk defaults to 65536 so full patches fit in a single batch.
    if requested == 5000 || requested == 4096 {
        65536
    } else {
        requested.max(1)
    }
}

/// Small stable power-of-two buckets. Never pad tiny coarse stages to 4096.
#[inline]
fn static_bucket_size(valid_pixels: usize, max_batch: usize) -> usize {
    if valid_pixels >= max_batch || max_batch < 256 { return valid_pixels; }
    valid_pixels.max(256).checked_next_power_of_two().unwrap_or(max_batch).min(max_batch).max(valid_pixels)
}

#[derive(Clone, Debug)]
pub struct ModelOptions {
    pub dtype: FloatDType,
    pub encoder_batch: usize,
    pub pixel_batch: usize,
    /// Spatial flash query limit; 0 uses all queries. Unfused scores are always bounded.
    pub query_batch: usize,
    pub fused_attention: bool,
    pub score_budget_mib: usize,
    pub fused_softmax: bool,
    pub optimized_encoder: bool,
    pub gemm_convolutions: bool,
}
impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            dtype: FloatDType::F32,
            encoder_batch: 8,
            pixel_batch: 65536,
            query_batch: 0,
            fused_attention: false,
            score_budget_mib: 256,
            fused_softmax: true,
            optimized_encoder: true,
            gemm_convolutions: true,
        }
    }
}
/// Both pretrained networks. Inference only; no autograd or Python runtime.
pub struct Model {
    weights: BTreeMap<String, Weight>,
    pub device: Device,
    pub options: ModelOptions,
}

pub fn tensor<const D: usize>(values: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(values, shape), device)
}
pub fn read_tensor<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.cast(FloatDType::F32).into_data().try_to_vec::<f32>().expect("FP32 tensor")
}
pub fn normalize(x: Tensor<4>) -> Tensor<4> {
    if let Some(out) = crate::norm::normalize_3ch(x.clone()) {
        return out;
    }
    // Normal-vector reductions must not underflow when activations use FP16.
    let x = x.cast(FloatDType::F32);
    let norm = x.clone().square().sum_dim(1).sqrt().clamp_min(1e-12);
    x / norm
}
impl Model {
    pub fn tensor_options(&self) -> TensorCreationOptions {
        TensorCreationOptions::new(self.device.clone()).with_dtype(self.options.dtype.into())
    }
    pub fn load(
        path: &Path,
        device: Device,
        pixel_batch: usize,
        query_batch: usize,
    ) -> Result<Self> {
        Self::load_with_options(path, device, ModelOptions {
            pixel_batch,
            query_batch,
            ..Default::default()
        })
    }
    pub fn load_with_options(path: &Path, device: Device, mut options: ModelOptions) -> Result<Self> {
        ensure!(
            options.pixel_batch > 0 && options.encoder_batch > 0,
            "Encoder and pixel batch sizes must be positive"
        );
        options.pixel_batch = effective_pixel_batch(options.pixel_batch);
        ensure!(device.supports_dtype(options.dtype), "Selected device does not support {:?}", options.dtype);
        ensure!(path.extension().and_then(|x| x.to_str()) != Some("pth"),
            "PyTorch .pth deserialization is not implemented; use the converted model_uncalibrated.safetensors");
        let bytes = std::fs::read(path).with_context(|| {
            format!(
                "Read {} (convert .pth with tools/export_weights.py first)",
                path.display()
            )
        })?;
        let archive = SafeTensors::deserialize(&bytes)?;
        let schema = expected_shapes();
        ensure!(
            archive.len() == schema.len(),
            "Expected {} tensors, got {}",
            schema.len(),
            archive.len()
        );
        let dtype = options.dtype;
        let upload_1d = |values: Vec<f32>| -> Tensor<1> {
            let n = values.len();
            let create_options = TensorCreationOptions::new(device.clone()).with_dtype(dtype.into());
            if dtype == FloatDType::F16 {
                let values: Vec<half::f16> = values.into_iter().map(half::f16::from_f32).collect();
                Tensor::from_data(TensorData::new(values, [n]), create_options)
            } else {
                Tensor::from_data(TensorData::new(values, [n]), create_options)
            }
        };
        let mut raw_f32: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        for (name, mut shape) in schema {
            let t = archive
                .tensor(&name)
                .with_context(|| format!("Missing {name}"))?;
            ensure!(
                t.shape() == shape && t.dtype() == Dtype::F32,
                "Invalid shape/dtype for {name}: {:?}",
                t.shape()
            );
            let mut values: Vec<f32> = t
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "Non-finite weight: {name}"
            );
            let n = values.len();
            if shape.len() == 2 {
                // Pack before upload: no GPU transpose/cast launches during load.
                let (rows, cols) = (shape[0], shape[1]);
                let mut packed = vec![0f32; n];
                for row in 0..rows {
                    for col in 0..cols { packed[col * rows + row] = values[row * cols + col]; }
                }
                values = packed;
                shape.swap(0, 1);
            }
            raw_f32.insert(name, (shape, values));
        }

        // Precompute dec.0.S_q = S * fc_q.weight + fc_q.bias on the CPU once during load,
        // eliminating all runtime `[B, 1, C]` `dec.0.mab.fc_q` GEMMs across every patch.
        let seed_keys: Vec<String> = raw_f32
            .keys()
            .filter_map(|k| k.strip_suffix(".dec.0.S").map(str::to_owned))
            .collect();
        for prefix in seed_keys {
            let (_, s_vals) = &raw_f32[&format!("{prefix}.dec.0.S")];
            let (w_shape, w_vals) = &raw_f32[&format!("{prefix}.dec.0.mab.fc_q.weight")];
            let (_, b_vals) = &raw_f32[&format!("{prefix}.dec.0.mab.fc_q.bias")];
            let (in_c, out_c) = (w_shape[0], w_shape[1]);
            let mut q_seed = vec![0f32; out_c];
            for col in 0..out_c {
                let mut acc = b_vals[col];
                for row in 0..in_c {
                    acc += s_vals[row] * w_vals[row * out_c + col];
                }
                q_seed[col] = acc;
            }
            raw_f32.insert(format!("{prefix}.dec.0.mab.q_seed"), (vec![1, 1, out_c], q_seed));
        }

        // Pack K and V projections in set-transformer MABs on the CPU before upload.
        let kv_prefixes: Vec<String> = raw_f32
            .keys()
            .filter_map(|name| name.strip_suffix(".fc_k.weight").map(str::to_owned))
            .collect();
        for prefix in kv_prefixes {
            let (k_shape, k_w) = raw_f32.remove(&format!("{prefix}.fc_k.weight")).unwrap();
            let (_, v_w) = raw_f32.remove(&format!("{prefix}.fc_v.weight")).unwrap();
            let (input, output) = (k_shape[0], k_shape[1]);
            let mut kv_w = vec![0f32; input * 2 * output];
            for r in 0..input {
                kv_w[r * 2 * output..r * 2 * output + output]
                    .copy_from_slice(&k_w[r * output..(r + 1) * output]);
                kv_w[r * 2 * output + output..(r + 1) * 2 * output]
                    .copy_from_slice(&v_w[r * output..(r + 1) * output]);
            }
            raw_f32.insert(format!("{prefix}.fc_kv.weight"), (vec![input, 2 * output], kv_w));
            let (_, mut k_b) = raw_f32.remove(&format!("{prefix}.fc_k.bias")).unwrap();
            let (_, v_b) = raw_f32.remove(&format!("{prefix}.fc_v.bias")).unwrap();
            k_b.extend_from_slice(&v_b);
            raw_f32.insert(format!("{prefix}.fc_kv.bias"), (vec![2 * output], k_b));
        }

        // Pack Stage 3 (sr_ratio == 1) spatial `attn.q` [512, 512] and `attn.kv` [512, 1024]
        // into a single `attn.qkv` [512, 1536] GEMM since both project the same `norm1(x)` input.
        let sr1_prefixes: Vec<String> = raw_f32
            .keys()
            .filter_map(|name| name.strip_suffix(".attn.q.weight").map(str::to_owned))
            .filter(|prefix| !raw_f32.contains_key(&format!("{prefix}.attn.sr.weight")))
            .collect();
        for prefix in sr1_prefixes {
            let (q_shape, q_w) = &raw_f32[&format!("{prefix}.attn.q.weight")];
            let (_, kv_w) = &raw_f32[&format!("{prefix}.attn.kv.weight")];
            let c = q_shape[0];
            let mut qkv_w = vec![0f32; c * 3 * c];
            for r in 0..c {
                qkv_w[r * 3 * c..r * 3 * c + c].copy_from_slice(&q_w[r * c..(r + 1) * c]);
                qkv_w[r * 3 * c + c..(r + 1) * 3 * c]
                    .copy_from_slice(&kv_w[r * 2 * c..(r + 1) * 2 * c]);
            }
            raw_f32.insert(format!("{prefix}.attn.qkv.weight"), (vec![c, 3 * c], qkv_w));
        }

        // Prepack targeted dense convolutions with channel-inner patch reduction order.
        // Retain original kernels for the selectable Burn path and unsupported devices.
        if options.gemm_convolutions {
            let names: Vec<String> = raw_f32.keys().filter(|name|
                name.ends_with(".attn.sr.weight") ||
                (2..=4).any(|stage| name.ends_with(&format!(".patch_embed{stage}.proj.weight")))
            ).cloned().collect();
            for name in names {
                let (shape, values) = &raw_f32[&name];
                let output = shape[0];
                let reduction = shape[1] * shape[2] * shape[3];
                let mut packed = vec![0.0_f32; values.len()];
                for o in 0..output {
                    for k in 0..reduction {
                        let channel = k % shape[1];
                        let spatial = k / shape[1];
                        let source = channel * shape[2] * shape[3] + spatial;
                        packed[k * output + o] = values[o * reduction + source];
                    }
                }
                raw_f32.insert(name.replace(".weight", ".gemm_weight"), (vec![reduction, output], packed));
            }
        }

        // Store all weights in their native rank so runtime lookups clone the handle
        // directly without recording thousands of Reshape nodes into Burn's Fusion IR.
        let mut weights = BTreeMap::new();
        for (name, (shape, values)) in raw_f32 {
            let flat = upload_1d(values);
            let w = match shape.as_slice() {
                &[d0] => Weight::R1(flat.reshape([d0])),
                &[d0, d1] => Weight::R2(flat.reshape([d0, d1])),
                &[d0, d1, d2] => Weight::R3(flat.reshape([d0, d1, d2])),
                &[d0, d1, d2, d3] => Weight::R4(flat.reshape([d0, d1, d2, d3])),
                _ => anyhow::bail!("Unsupported weight rank for {name}"),
            };
            weights.insert(name, w);
        }
        Ok(Self {
            weights,
            device,
            options,
        })
    }
    #[inline]
    fn w1(&self, key: &str) -> Tensor<1> {
        let Weight::R1(ref t) = self.weights[key] else { panic!("expected rank-1 weight: {key}") };
        t.clone()
    }
    #[inline]
    fn w2(&self, key: &str) -> Tensor<2> {
        let Weight::R2(ref t) = self.weights[key] else { panic!("expected rank-2 weight: {key}") };
        t.clone()
    }
    #[inline]
    fn w3(&self, key: &str) -> Tensor<3> {
        let Weight::R3(ref t) = self.weights[key] else { panic!("expected rank-3 weight: {key}") };
        t.clone()
    }
    #[inline]
    fn w4(&self, key: &str) -> Tensor<4> {
        let Weight::R4(ref t) = self.weights[key] else { panic!("expected rank-4 weight: {key}") };
        t.clone()
    }
    fn linear(&self, x: Tensor<3>, p: &str) -> Tensor<3> {
        let _profile = crate::profile::span(&self.device, || format!("linear/{p} input={:?}", x.dims()));
        let w = self.w2(&format!("{p}.weight"));
        let [i, o] = w.dims();
        let [b, n, _] = x.dims();
        // In the light transformers b is thousands and n is often only 6-20.
        // Flattening produces one GEMM instead of thousands of tiny batched GEMMs.
        let bias_key = format!("{p}.bias");
        let bias = self.weights.contains_key(&bias_key).then(|| self.w1(&bias_key));
        // Preserve one large GEMM and expose bias through Burn's linear IR.
        let y = if self.options.optimized_encoder {
            module::linear(x.reshape([b * n, i]), w, bias)
        } else {
            let y = x.reshape([b * n, i]).matmul(w);
            if let Some(bias) = bias { y + bias.reshape([1, o]) } else { y }
        };
        y.reshape([b, n, o])
    }
    fn norm(&self, x: Tensor<3>, p: &str) -> Tensor<3> {
        let _profile = crate::profile::span(&self.device, || format!("norm/{p} input={:?}", x.dims()));
        let gamma = self.w1(&format!("{p}.weight"));
        let beta = self.w1(&format!("{p}.bias"));
        if let Some(out) = crate::norm::forward(x.clone(), gamma.clone(), beta.clone()) {
            return out;
        }
        module::layer_norm(x, gamma, Some(beta), 1e-5)
    }
    fn conv(&self, x: Tensor<4>, p: &str, stride: usize, pad: usize, groups: usize) -> Tensor<4> {
        let _profile = crate::profile::span(&self.device, || format!("conv/{p} input={:?} stride={stride} groups={groups}", x.dims()));
        module::conv2d(
            x,
            self.w4(&format!("{p}.weight")),
            Some(self.w1(&format!("{p}.bias"))),
            ConvOptions::new([stride; 2], [pad; 2], [1; 2], groups),
        )
    }
    /// Target dense convolutions without materializing NCHW inputs or outputs.
    fn conv_tokens(&self, x: Tensor<3>, p: &str, h: usize, w: usize, kernel: usize, stride: usize, pad: usize) -> Tensor<3> {
        let [batch, tokens, channels] = x.dims();
        assert_eq!(tokens, h * w);
        let oh = (h + 2 * pad - kernel) / stride + 1;
        let ow = (w + 2 * pad - kernel) / stride + 1;
        let packed_key = format!("{p}.gemm_weight");
        if !self.options.gemm_convolutions || !self.weights.contains_key(&packed_key) || !crate::conv_gemm::supported(&x) {
            let y = self.conv(x.swap_dims(1, 2).reshape([batch, channels, h, w]), p, stride, pad, 1);
            let output = y.dims()[1];
            return y.reshape([batch, output, oh * ow]).swap_dims(1, 2);
        }
        let _profile = crate::profile::span(&self.device, || format!("conv-gemm/{p} B={batch} H={h} W={w} K={kernel} stride={stride}"));
        let weight = self.w2(&packed_key);
        let output = weight.dims()[1];
        let bias = self.w1(&format!("{p}.bias"));
        let rows = batch * oh * ow;
        let bytes = if self.options.dtype == FloatDType::F16 { 2 } else { 4 };
        // Each packed workspace is at most 16 MiB (not a total VRAM cap).
        let limit = (128 * 1024 * 1024 / (channels * kernel * kernel * bytes)).max(1);
        let chunk_rows = (1usize << limit.ilog2()).min(rows);
        crate::profile::event(|| format!("conv/gemm {p} chunk_rows={chunk_rows} reduction={}", channels * kernel * kernel));
        let mut result = if chunk_rows < rows { Some(Tensor::<2>::empty([rows, output], self.tensor_options())) } else { None };
        for start in (0..rows).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(rows);
            let pack_profile = crate::profile::span(&self.device, || format!("conv-pack/{p} rows={}", end - start));
            let patches = crate::conv_gemm::patches(x.clone(), h, w, kernel, stride, pad, start, end - start);
            drop(pack_profile);
            let gemm_profile = crate::profile::span(&self.device, || format!("conv-matmul/{p} rows={}", end - start));
            let value = module::linear(patches, weight.clone(), Some(bias.clone()));
            drop(gemm_profile);
            if result.is_none() { return value.reshape([batch, oh * ow, output]); }
            result = Some(result.take().unwrap().slice_assign([start..end, 0..output], value));
            crate::execution::flush(&self.device);
        }
        result.unwrap().reshape([batch, oh * ow, output])
    }

    /// Consumes `q` `[b, n, c]` and packed `kv` `[b, m, 2 * c]` directly on the fused subgroup
    /// paths, only slicing `kv` into separate `k` and `v` tensors when falling back to CMMA flash or GEMM.
    fn attention_packed(
        &self,
        q: Tensor<3>,
        kv: Tensor<3>,
        heads: usize,
        scale: f32,
        spatial: bool,
    ) -> Tensor<3> {
        let [b, n, c] = q.dims();
        let m = kv.dims()[1];
        let d = c / heads;
        let _profile = crate::profile::span(&self.device, || format!("attention/{} B={b} Q={n} K={m} C={c} heads={heads}", if spatial { "spatial" } else { "light" }));
        if spatial && self.options.fused_attention {
            let q_probe = q.clone().reshape([b, n, heads, d]);
            let use_cmma_flash = n >= 4096 && m >= 64 && crate::attention::supported(&q_probe, &q_probe);
            if !use_cmma_flash && crate::attention::subgroup_supported(&q) {
                let queries = if self.options.query_batch == 0 {
                    n
                } else {
                    self.options.query_batch.min(n)
                };
                if queries >= n {
                    let _chunk_profile = crate::profile::span(&self.device, || format!("attention/chunk subgroup Q=[{b}, {heads}, {n}, {d}] K={m}"));
                    if let Some(out) = crate::attention::spatial_subgroup(q.clone(), kv.clone(), heads) {
                        return out;
                    }
                } else if queries % 2 == 0 {
                    let mut out = Tensor::empty([b, n, c], self.tensor_options());
                    let mut ok = true;
                    for (chunk, start) in (0..n).step_by(queries).enumerate() {
                        let end = (start + queries).min(n);
                        let _chunk_profile = crate::profile::span(&self.device, || format!("attention/chunk subgroup Q=[{b}, {heads}, {}, {d}] K={m}", end - start));
                        if let Some(value) = crate::attention::spatial_subgroup(
                            q.clone().slice([0..b, start..end, 0..c]),
                            kv.clone(),
                            heads,
                        ) {
                            out = out.slice_assign([0..b, start..end, 0..c], value);
                            crate::execution::checkpoint(&self.device, chunk + 1);
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        crate::execution::finish(&self.device);
                        return out;
                    }
                }
            }
        }
        let k = kv.clone().slice([0..b, 0..m, 0..c]);
        let v = kv.slice([0..b, 0..m, c..2 * c]);
        let q = q.reshape([b, n, heads, d]).permute([0, 2, 1, 3]);
        let k = k.reshape([b, m, heads, d]).permute([0, 2, 1, 3]);
        let v = v.reshape([b, m, heads, d]).permute([0, 2, 1, 3]);
        let flash = spatial && n >= 4096 && m >= 64 && self.options.fused_attention && crate::attention::supported(&q, &k);
        let kt = k.clone().swap_dims(2, 3);
        let scaled_kt = kt * scale;
        let compute = |q: Tensor<4>| {
            let _profile = crate::profile::span(&self.device, || format!("attention/chunk {} Q={:?} K={m}", if flash { "flash" } else { "gemm" }, q.dims()));
            if flash {
                if let Some(out) = crate::attention::flash(q.clone(), k.clone(), v.clone()) {
                    return out;
                }
            }
            let qk_profile = crate::profile::span(&self.device, || format!("attention/qk-gemm Q={:?} K={m}", q.dims()));
            let scores = q.matmul(scaled_kt.clone());
            drop(qk_profile);
            let softmax_profile = crate::profile::span(&self.device, || format!("attention/softmax shape={:?}", scores.dims()));
            let probabilities = if self.options.fused_softmax {
                crate::softmax::forward(scores)
            } else { activation::softmax(scores, 3) };
            drop(softmax_profile);
            let _av_profile = crate::profile::span(&self.device, || format!("attention/av-gemm probabilities={:?}", probabilities.dims()));
            probabilities.matmul(v.clone())
        };
        let elem_bytes = if self.options.dtype == FloatDType::F16 { 2 } else { 4 };
        let score_limit = (self.options.score_budget_mib.max(1) * 1024 * 1024usize / (b * heads * m * elem_bytes)).max(1);
        let score_limit = if spatial { 1usize << score_limit.ilog2() } else { score_limit };
        let requested = if !spatial || self.options.query_batch == 0 { n }
            else { self.options.query_batch.min(n) };
        let queries = if flash { requested } else { requested.min(score_limit) };
        if queries >= n {
            return compute(q).permute([0, 2, 1, 3]).reshape([b, n, c]);
        }
        let mut out = Tensor::empty([b, heads, n, d], self.tensor_options());
        for (chunk, start) in (0..n).step_by(queries).enumerate() {
            let end = (start + queries).min(n);
            let value = compute(q.clone().slice([0..b, 0..heads, start..end, 0..d]));
            out = out.slice_assign([0..b, 0..heads, start..end, 0..d], value);
            crate::execution::checkpoint(&self.device, chunk + 1);
        }
        crate::execution::finish(&self.device);
        out.permute([0, 2, 1, 3]).reshape([b, n, c])
    }
    fn block(
        &self,
        x: Tensor<3>,
        p: &str,
        h: usize,
        w: usize,
        heads: usize,
        sr: usize,
    ) -> Tensor<3> {
        let [b, n, c] = x.dims();
        let _block_profile = crate::profile::span(&self.device, || format!("encoder-block/{p} B={b} N={n} C={c}"));
        let z = self.norm(x.clone(), &format!("{p}.norm1"));
        let (q, kv) = if sr > 1 {
            let q = self.linear(z.clone(), &format!("{p}.attn.q"));
            let _reduction_profile = crate::profile::span(&self.device, || format!("encoder/spatial-reduction {p} stride={sr}"));
            let a = self.conv_tokens(z, &format!("{p}.attn.sr"), h, w, sr, sr, 0);
            let z_sr = self.norm(a, &format!("{p}.attn.norm"));
            let kv = self.linear(z_sr, &format!("{p}.attn.kv"));
            (q, kv)
        } else {
            let qkv = self.linear(z, &format!("{p}.attn.qkv"));
            let q = qkv.clone().slice([0..b, 0..n, 0..c]);
            let kv = qkv.slice([0..b, 0..n, c..3 * c]);
            (q, kv)
        };
        let a = self.attention_packed(q, kv, heads, 1.0 / (c as f32 / heads as f32).sqrt(), true);
        let projected = self.linear(a, &format!("{p}.attn.proj"));
        let residual_profile = crate::profile::span(&self.device, || format!("encoder/attention-residual {p}"));
        let x = x + projected;
        drop(residual_profile);
        let z = self.linear(
            self.norm(x.clone(), &format!("{p}.norm2")),
            &format!("{p}.mlp.fc1"),
        );
        let depthwise_profile = crate::profile::span(&self.device, || format!("mlp/depthwise-gelu {p} input={:?}", z.dims()));
        let dw = format!("{p}.mlp.dwconv.dwconv");
        let fused = if self.options.optimized_encoder {
            crate::depthwise::forward(z.clone(), self.w4(&format!("{dw}.weight")), self.w1(&format!("{dw}.bias")), h, w)
        } else { None };
        let z = if let Some(out) = fused {
            out
        } else {
            let conv = self.conv(z.swap_dims(1, 2).reshape([b, 4 * c, h, w]), &dw, 1, 1, 4 * c);
            activation::gelu(conv.reshape([b, 4 * c, n]).swap_dims(1, 2))
        };
        drop(depthwise_profile);
        let projected = self.linear(z, &format!("{p}.mlp.fc2"));
        let _residual_profile = crate::profile::span(&self.device, || format!("encoder/mlp-residual {p}"));
        x + projected
    }
    fn mab(&self, x: Tensor<3>, y: Tensor<3>, p: &str, preprojected_q: bool) -> Tensor<3> {
        let q = if preprojected_q {
            x
        } else {
            let x = if self.weights.contains_key(&format!("{p}.ln0.weight")) {
                self.norm(x, &format!("{p}.ln0"))
            } else {
                x
            };
            self.linear(x, &format!("{p}.fc_q"))
        };
        let c = q.dims()[2];
        let kv = self.linear(y, &format!("{p}.fc_kv"));
        // Set transformer uses sqrt(full dimension), unlike spatial attention.
        // `light_attention::forward` fuses the `q + attn` residual directly in registers.
        let o = if let Some(out) = crate::light_attention::forward(q.clone(), kv.clone(), 4) {
            out
        } else {
            q.clone() + self.attention_packed(q, kv, 4, 1.0 / (c as f32).sqrt(), false)
        };
        let z = if self.weights.contains_key(&format!("{p}.ln1.weight")) {
            self.norm(o.clone(), &format!("{p}.ln1"))
        } else {
            o.clone()
        };
        o + self.linear(
            activation::gelu(self.linear(z, &format!("{p}.fc_o1"))),
            &format!("{p}.fc_o2"),
        )
    }
    fn set_layer(&self, x: Tensor<3>, p: &str, pool: bool) -> Tensor<3> {
        let _profile = crate::profile::span(&self.device, || format!("set-transformer/{p} input={:?}", x.dims()));
        let [b, n, c] = x.dims();
        let batch_limit = effective_pixel_batch(self.options.pixel_batch);
        let q_seed = if pool {
            Some(self.w3(&format!("{p}.dec.0.mab.q_seed")))
        } else {
            None
        };
        let out_n = if pool { 1 } else { n };
        let mut out = if b > batch_limit {
            Some(Tensor::empty([b, out_n, c], self.tensor_options()))
        } else { None };
        for (chunk, start) in (0..b).step_by(batch_limit).enumerate() {
            let end = (start + batch_limit).min(b);
            let valid_pixels = end - start;
            let work_pixels = static_bucket_size(valid_pixels, batch_limit);
            let mut z = if start == 0 && end == b {
                x.clone()
            } else {
                x.clone().slice([start..end, 0..n, 0..c])
            };
            if work_pixels > valid_pixels {
                z = Tensor::cat(vec![z, Tensor::zeros([work_pixels - valid_pixels, n, c], self.tensor_options())], 0);
            }
            for i in 0..3 {
                z = self.mab(z.clone(), z, &format!("{p}.enc.{i}.mab"), false);
            }
            if let Some(ref s_q) = q_seed {
                let seed_q = s_q.clone().repeat_dim(0, work_pixels);
                z = self.mab(seed_q, z, &format!("{p}.dec.0.mab"), true);
            }
            if work_pixels > valid_pixels {
                z = z.slice([0..valid_pixels, 0..out_n, 0..c]);
            }
            if out.is_none() { return z; }
            out = Some(out.take().unwrap().slice_assign([start..end, 0..out_n, 0..c], z));
            crate::execution::checkpoint(&self.device, chunk + 1);
        }
        crate::execution::finish(&self.device);
        out.expect("non-empty light-transformer input")
    }
    fn set_layer_pre_gathered(
        &self,
        x_gathered: Tensor<3>,
        p: &str,
        pool: bool,
        active_mask: Option<&ActiveMask>,
        total_pixels: usize,
    ) -> Tensor<3> {
        let [_, n, c] = x_gathered.dims();
        let _profile = crate::profile::span(&self.device, || format!(
            "masked-set/{p} pixels={total_pixels} active={} padded={} lights={n} C={c}",
            active_mask.map_or(total_pixels, |m| m.valid_len),
            active_mask.map_or(total_pixels, |m| m.padded_len),
        ));
        let out_n = if pool { 1 } else { n };
        if let Some(mask) = active_mask {
            let processed = self.set_layer(x_gathered,p,pool);
            let _scatter_profile = crate::profile::span(&self.device, || format!("layout/inverse-scatter {p}"));
            if let Some(out) = crate::remap::forward(processed.clone(),mask.inverse_idx.clone()) { return out; }
            // Non-Cube backend fallback: only unique valid rows are scattered.
            let processed = processed.slice([0..mask.valid_len,0..out_n,0..c]);
            let indices = mask.gather_idx.clone().slice([0..mask.valid_len]);
            Tensor::zeros([total_pixels,out_n,c],self.tensor_options())
                .select_assign(0,indices,processed,burn::tensor::IndexingUpdateOp::Add)
        } else {
            self.set_layer(x_gathered, p, pool)
        }
    }

    /// Pre-warm and compile all unique GPU kernels (spatial attention, convolutions, norms,
    /// set-transformer GEMMs, and decoder up-blocks) for both Net_first and Net_stage.
    pub fn warmup(&self, lights: usize) -> Result<()> {
        let lights = lights.max(1);

        // 1. Warm Net_first (32x32 global coarse stage)
        let dummy_first = Tensor::zeros([lights, 3, 32, 32], self.tensor_options());
        let mask32 = vec![true; 32 * 32];
        let _ = self.forward_masked(dummy_first, false, Some(&mask32))?;

        // 2. Warm intermediate scales (64x64, 128x128)
        for &sz in &[64, 128] {
            let dummy = Tensor::zeros([lights, 6, sz, sz], self.tensor_options());
            let mask = vec![true; sz * sz];
            let _ = self.forward_masked(dummy, true, Some(&mask))?;
        }

        // 3. Warm Net_stage (256x256 patch refinement stage)
        // A representative mask; individual sparse bucket paths are warmed explicitly below.
        let dummy_stage = Tensor::zeros([lights, 6, 256, 256], self.tensor_options());
        let mut mask256 = vec![false; 256 * 256];
        for y in 32..224 {
            for x in 32..224 {
                mask256[y * 256 + x] = true;
            }
        }
        let _ = self.forward_masked(dummy_stage.clone(), true, Some(&mask256))?;

        // 4. Warm dense Net_stage (full 256x256 patch without masking)
        let _ = self.forward_masked(dummy_stage, true, None)?;

        // Exercise every sparse pool bucket before any timed view. The zero
        // input is shape-only during compile-only replay; light networks stay dense.
        for stage in 0..4 {
            let channels = [64,128,256,512][stage];
            let spatial = [65536,16384,4096,1024][stage];
            for &bucket in &[256,512,1024,2048,4096,8192,16384,32768] {
                if bucket >= spatial || bucket > self.options.pixel_batch { continue; }
                let prefix = format!("Net_stage.pool_block{}",stage+1);
                let dummy = Tensor::zeros([bucket,lights,channels],self.tensor_options());
                let processed = self.set_layer(dummy.clone(),&prefix,true);
                let indices = Tensor::<1,burn::tensor::Int>::from_data(TensorData::new(vec![0_i32;bucket],[bucket]),&self.device);
                let inverse = Tensor::<1,burn::tensor::Int>::from_data(TensorData::new(vec![-1_i32;spatial],[spatial]),&self.device);
                let gather_warm = crate::remap::forward(dummy,indices);
                let scatter_warm = crate::remap::forward(processed,inverse);
                self.device.flush().map_err(|e|anyhow::anyhow!("Sparse warmup submission failed: {e:?}"))?;
                drop((gather_warm,scatter_warm));
            }
        }

        // 5. Ensure all pipeline shader compilations and autotunings complete
        self.device.sync().map_err(|e|anyhow::anyhow!("Warmup GPU completion failed: {e:?}"))?;
        Ok(())
    }

    pub fn forward(&self, images: Tensor<4>, refinement: bool) -> Result<Tensor<4>> {
        self.forward_masked(images, refinement, None)
    }

    /// Images [lights, channels, height, width]; dimensions must be multiples of 8, >=32.
    /// Processes one object/patch, with bounded batches of input lights and optional foreground mask.
    pub fn forward_masked(
        &self,
        mut images: Tensor<4>,
        refinement: bool,
        host_mask: Option<&[bool]>,
    ) -> Result<Tensor<4>> {
        let [lights, channels, h, w] = images.dims();
        let _profile = crate::profile::span(&self.device, || format!("network/{} input={lights}x{channels}x{h}x{w}", if refinement { "refinement" } else { "first" }));
        ensure!(
            lights > 0 && channels == if refinement { 6 } else { 3 },
            "Invalid input channels/light count"
        );
        ensure!(
            h >= 32 && w >= 32 && h % 8 == 0 && w % 8 == 0,
            "Dimensions must be multiples of 8 and >=32"
        );
        let root = if refinement { "Net_stage" } else { "Net_first" };
        images = images.cast(self.options.dtype);
        let batch_limit = effective_pixel_batch(self.options.pixel_batch);
        let mut current_mask: Option<Vec<bool>> = host_mask.map(|m| m.to_vec());
        let mut pools = Vec::new();
        let mut deepest = None;
        for i in 0..4 {
            let _profile = crate::profile::span(&self.device, || format!("encoder-stage/{root}/{} input={:?}", i + 1, images.dims()));
            let p = format!("{root}.patch_embed{}", i + 1);
            let [_, ic, ih, iw] = images.dims();
            let c = [64, 128, 256, 512][i];
            let oh = if i == 0 { ih } else { ih / 2 };
            let ow = if i == 0 { iw } else { iw / 2 };
            let mask_profile = crate::profile::span(&self.device, || format!("mask/prepare-stage {}", i + 1));
            if i == 0 {
                if let Some(prev) = current_mask.take() {
                    // Exact backward dependency cone for `pools[0]` through `output_block` (3x3 conv, pad 1):
                    // output at K0 only reads `pools[0]` within Chebyshev distance 1 of K0.
                    let mut d0 = vec![false; oh * ow];
                    for y in 0..oh {
                        let y0 = y.saturating_sub(1);
                        let y1 = (y + 1).min(oh - 1);
                        for x in 0..ow {
                            let x0 = x.saturating_sub(1);
                            let x1 = (x + 1).min(ow - 1);
                            let mut any = false;
                            for yy in y0..=y1 {
                                for xx in x0..=x1 {
                                    if prev[yy * ow + xx] {
                                        any = true;
                                        break;
                                    }
                                }
                                if any { break; }
                            }
                            d0[y * ow + x] = any;
                        }
                    }
                    current_mask = Some(d0);
                }
            } else if let Some(prev) = current_mask.take() {
                // Exact backward dependency cone for `pools[i]` through `up_block` (ConvTranspose2d 4x4, stride 2, pad 1):
                // output pixel y at 2x resolution depends on input u where y in [2*u - 1, 2*u + 2].
                let mut next_mask = vec![false; oh * ow];
                for u in 0..oh {
                    let y0 = (2 * u).saturating_sub(1);
                    let y1 = (2 * u + 2).min(ih - 1);
                    for v in 0..ow {
                        let x0 = (2 * v).saturating_sub(1);
                        let x1 = (2 * v + 2).min(iw - 1);
                        let mut any = false;
                        for yy in y0..=y1 {
                            for xx in x0..=x1 {
                                if prev[yy * iw + xx] {
                                    any = true;
                                    break;
                                }
                            }
                            if any { break; }
                        }
                        next_mask[u * ow + v] = any;
                    }
                }
                current_mask = Some(next_mask);
            }
            let active_mask = current_mask.as_ref().and_then(|m| {
                let keep_count = m.iter().filter(|&&k|k).count();
                if keep_count == 0 || keep_count * 10 >= m.len() * 9 { return None; }
                let valid_len = keep_count;
                let full_chunks = (valid_len / batch_limit) * batch_limit;
                let rem = valid_len - full_chunks;
                let padded_len = full_chunks + if rem == 0 { 0 } else { static_bucket_size(rem,batch_limit) };
                if padded_len >= m.len() { return None; }
                let mut gather = Vec::with_capacity(padded_len);
                let mut inverse = vec![-1_i32;m.len()];
                for (pixel,&keep) in m.iter().enumerate() {
                    if keep {
                        inverse[pixel] = gather.len() as i32;
                        gather.push(pixel as i32);
                    }
                }
                gather.resize(padded_len,0);
                Some(ActiveMask {
                    gather_idx: Tensor::<1,burn::tensor::Int>::from_data(TensorData::new(gather,[padded_len]),&self.device),
                    inverse_idx: Tensor::<1,burn::tensor::Int>::from_data(TensorData::new(inverse,[m.len()]),&self.device),
                    valid_len,padded_len,
                })
            });

            drop(mask_profile);
            let mut encoded_chunks = Vec::with_capacity((lights + self.options.encoder_batch - 1) / self.options.encoder_batch);
            for light in (0..lights).step_by(self.options.encoder_batch) {
                let end = (light + self.options.encoder_batch).min(lights);
                let batch = end - light;
                let input_batch = if light == 0 && end == lights {
                    images.clone()
                } else {
                    images.clone().slice([light..end, 0..ic, 0..ih, 0..iw])
                };
                let (z, hh, ww) = if i > 0 {
                    let tokens = input_batch.reshape([batch, ic, ih * iw]).swap_dims(1, 2);
                    (self.conv_tokens(tokens, &format!("{p}.proj"), ih, iw, 3, 2, 1), oh, ow)
                } else {
                    // Keep the cheap first embedding on Burn's existing path.
                    let z = self.conv(input_batch, &format!("{p}.proj"), 1, 1, 1);
                    let [_, c, hh, ww] = z.dims();
                    (z.reshape([batch, c, hh * ww]).swap_dims(1, 2), hh, ww)
                };
                let mut z = self.norm(z, &format!("{p}.norm"));
                for j in 0..[3, 4, 6, 3][i] {
                    z = self.block(
                        z,
                        &format!("{root}.block{}.{j}", i + 1),
                        hh,
                        ww,
                        [1, 2, 4, 8][i],
                        [8, 4, 2, 1][i],
                    );
                }
                encoded_chunks.push(self.norm(z, &format!("{root}.norm{}", i + 1)));
                crate::execution::flush(&self.device);
            }
            let encoded = if encoded_chunks.len() == 1 {
                encoded_chunks.pop().unwrap()
            } else {
                Tensor::cat(encoded_chunks, 0)
            };
            let gather_profile = crate::profile::span(&self.device, || format!("layout/foreground-gather stage={}", i + 1));
            let x = encoded.permute([1, 0, 2]);
            // `pool_block` outputs only enter the local convolutional U-Net decoder, so gathering
            // the exact decoder backward cone `active_mask` is 100% lossless (0.0 error on kept pixels).
            let x_pool = if let Some(ref mask) = active_mask {
                crate::remap::forward(x.clone(),mask.gather_idx.clone())
                    .unwrap_or_else(||x.clone().select(0,mask.gather_idx.clone()))
            } else {
                x.clone()
            };
            drop(gather_profile);
            let pool = self
                .set_layer_pre_gathered(x_pool, &format!("{root}.pool_block{}", i + 1), true, active_mask.as_ref(), oh * ow)
                .reshape([1, oh, ow, c])
                .permute([0, 3, 1, 2]);
            if i < 3 {
                // `light_block` feeds into subsequent encoder stages (`block2`, `block3`, `block4`)
                // which perform global spatial self-attention over all spatial tokens in the patch.
                images = self
                    .set_layer(x, &format!("{root}.light_block{}", i + 1), false)
                    .permute([1, 2, 0])
                    .reshape([lights, c, oh, ow]);
                pools.push(pool);
            } else {
                deepest = Some(pool);
            }
        }
        let mut x = deepest.unwrap();
        let _decoder_profile = crate::profile::span(&self.device, || format!("decoder/{root} input={:?}", x.dims()));
        for i in 0..3 {
            let _profile = crate::profile::span(&self.device, || format!("decoder/up{} input={:?}", i + 1, x.dims()));
            let p = format!("{root}.up_block{}.0", i + 1);
            x = module::conv_transpose2d(
                x,
                self.w4(&format!("{p}.weight")),
                Some(self.w1(&format!("{p}.bias"))),
                ConvTransposeOptions::new([2; 2], [1; 2], [0; 2], [1; 2], 1),
            );
            x = activation::leaky_relu(x, 0.1);
            x = Tensor::cat(vec![x, pools[2 - i].clone()], 1);
        }
        Ok(normalize(self.conv(
            x,
            &format!("{root}.output_block"),
            1,
            1,
            1,
        )))
    }
}

fn expected_shapes() -> BTreeMap<String, Vec<usize>> {
    fn linear(s: &mut BTreeMap<String, Vec<usize>>, p: &str, i: usize, o: usize, b: bool) {
        s.insert(format!("{p}.weight"), vec![o, i]);
        if b {
            s.insert(format!("{p}.bias"), vec![o]);
        }
    }
    fn norm(s: &mut BTreeMap<String, Vec<usize>>, p: &str, c: usize) {
        s.insert(format!("{p}.weight"), vec![c]);
        s.insert(format!("{p}.bias"), vec![c]);
    }
    fn conv(s: &mut BTreeMap<String, Vec<usize>>, p: &str, i: usize, o: usize, k: usize) {
        s.insert(format!("{p}.weight"), vec![o, i, k, k]);
        s.insert(format!("{p}.bias"), vec![o]);
    }
    fn mab(s: &mut BTreeMap<String, Vec<usize>>, p: &str, c: usize, ff: usize, ln: bool) {
        for l in ["fc_q", "fc_k", "fc_v"] {
            linear(s, &format!("{p}.{l}"), c, c, true);
        }
        linear(s, &format!("{p}.fc_o1"), c, ff, true);
        linear(s, &format!("{p}.fc_o2"), ff, c, true);
        if ln {
            for l in ["ln0", "ln1"] {
                norm(s, &format!("{p}.{l}"), c);
            }
        }
    }
    let mut s = BTreeMap::new();
    for (root, input) in [("Net_first", 3), ("Net_stage", 6)] {
        for i in 0..4 {
            let c = [64, 128, 256, 512][i];
            let sr = [8, 4, 2, 1][i];
            let p = format!("{root}.patch_embed{}", i + 1);
            conv(
                &mut s,
                &format!("{p}.proj"),
                if i == 0 { input } else { c / 2 },
                c,
                3,
            );
            norm(&mut s, &format!("{p}.norm"), c);
            norm(&mut s, &format!("{root}.norm{}", i + 1), c);
            for j in 0..[3, 4, 6, 3][i] {
                let p = format!("{root}.block{}.{j}", i + 1);
                for n in ["norm1", "norm2"] {
                    norm(&mut s, &format!("{p}.{n}"), c);
                }
                linear(&mut s, &format!("{p}.attn.q"), c, c, false);
                linear(&mut s, &format!("{p}.attn.kv"), c, 2 * c, false);
                linear(&mut s, &format!("{p}.attn.proj"), c, c, true);
                if sr > 1 {
                    conv(&mut s, &format!("{p}.attn.sr"), c, c, sr);
                    norm(&mut s, &format!("{p}.attn.norm"), c);
                }
                linear(&mut s, &format!("{p}.mlp.fc1"), c, 4 * c, true);
                linear(&mut s, &format!("{p}.mlp.fc2"), 4 * c, c, true);
                conv(&mut s, &format!("{p}.mlp.dwconv.dwconv"), 1, 4 * c, 3);
            }
            for kind in if i < 3 {
                vec!["light", "pool"]
            } else {
                vec!["pool"]
            } {
                let p = format!("{root}.{kind}_block{}", i + 1);
                for j in 0..3 {
                    mab(&mut s, &format!("{p}.enc.{j}.mab"), c, 2 * c, true);
                }
                if kind == "pool" {
                    s.insert(format!("{p}.dec.0.S"), vec![1, 1, c]);
                    mab(&mut s, &format!("{p}.dec.0.mab"), c, 512, false);
                }
            }
        }
        for (i, (ic, oc)) in [(512, 512), (768, 256), (384, 128)].into_iter().enumerate() {
            let p = format!("{root}.up_block{}.0", i + 1);
            s.insert(format!("{p}.weight"), vec![ic, oc, 4, 4]);
            s.insert(format!("{p}.bias"), vec![oc]);
        }
        conv(&mut s, &format!("{root}.output_block"), 192, 3, 3);
    }
    s
}
