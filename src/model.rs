use anyhow::{Context, Result, ensure};
use burn::tensor::{
    Device, FloatDType, Tensor, TensorCreationOptions, TensorData, activation, module,
    ops::{ConvOptions, ConvTransposeOptions},
};
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeMap, path::Path};

struct Weight {
    shape: Vec<usize>,
    value: Tensor<1>,
}
#[derive(Clone, Debug)]
pub struct ModelOptions {
    pub dtype: FloatDType,
    pub encoder_batch: usize,
    pub pixel_batch: usize,
    /// Spatial flash query limit; 0 uses all queries. Unfused scores are always bounded.
    pub query_batch: usize,
    pub fused_attention: bool,
}
impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            dtype: FloatDType::F32,
            encoder_batch: 8,
            pixel_batch: 5000,
            query_batch: 0,
            fused_attention: true,
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
    pub fn load_with_options(path: &Path, device: Device, options: ModelOptions) -> Result<Self> {
        ensure!(
            options.pixel_batch > 0 && options.encoder_batch > 0,
            "Encoder and pixel batch sizes must be positive"
        );
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
        let mut weights = BTreeMap::new();
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
            let dtype = options.dtype;
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
            let create_options = TensorCreationOptions::new(device.clone()).with_dtype(dtype.into());
            let value = if dtype == FloatDType::F16 {
                let values: Vec<half::f16> = values.into_iter().map(half::f16::from_f32).collect();
                Tensor::from_data(TensorData::new(values, [n]), create_options)
            } else {
                Tensor::from_data(TensorData::new(values, [n]), create_options)
            };
            weights.insert(
                name,
                Weight {
                    shape,
                    value,
                },
            );
        }
        // K and V always consume the same (unnormalized) light tokens. Pack
        // their projections once, then use a single large GEMM per MAB.
        let kv_prefixes: Vec<String> = weights.keys()
            .filter_map(|name| name.strip_suffix(".fc_k.weight").map(str::to_owned))
            .collect();
        for prefix in kv_prefixes {
            let k = weights.remove(&format!("{prefix}.fc_k.weight")).unwrap();
            let v = weights.remove(&format!("{prefix}.fc_v.weight")).unwrap();
            let (input, output) = (k.shape[0], k.shape[1]);
            let value = Tensor::cat(vec![k.value.reshape([input, output]), v.value.reshape([input, output])], 1)
                .reshape([input * 2 * output]);
            weights.insert(format!("{prefix}.fc_kv.weight"), Weight { shape: vec![input, 2 * output], value });
            let k = weights.remove(&format!("{prefix}.fc_k.bias")).unwrap();
            let v = weights.remove(&format!("{prefix}.fc_v.bias")).unwrap();
            weights.insert(format!("{prefix}.fc_kv.bias"), Weight {
                shape: vec![2 * output], value: Tensor::cat(vec![k.value, v.value], 0),
            });
        }
        Ok(Self {
            weights,
            device,
            options,
        })
    }
    fn w<const D: usize>(&self, key: &str) -> Tensor<D> {
        let w = &self.weights[key];
        w.value
            .clone()
            .reshape(<[usize; D]>::try_from(w.shape.as_slice()).expect("validated rank"))
    }
    fn linear(&self, x: Tensor<3>, p: &str) -> Tensor<3> {
        let _profile = crate::profile::span(&self.device, || format!("linear/{p} input={:?}", x.dims()));
        let w = self.w::<2>(&format!("{p}.weight"));
        let [i, o] = w.dims();
        let [b, n, _] = x.dims();
        // In the light transformers b is thousands and n is often only 6-20.
        // Flattening produces one GEMM instead of thousands of tiny batched GEMMs.
        let y = x.reshape([b * n, i]).matmul(w);
        let y = if self.weights.contains_key(&format!("{p}.bias")) {
            y + self.w::<1>(&format!("{p}.bias")).reshape([1, o])
        } else {
            y
        };
        y.reshape([b, n, o])
    }
    fn norm(&self, x: Tensor<3>, p: &str) -> Tensor<3> {
        let _profile = crate::profile::span(&self.device, || format!("norm/{p} input={:?}", x.dims()));
        module::layer_norm(
            x,
            self.w(&format!("{p}.weight")),
            Some(self.w(&format!("{p}.bias"))),
            1e-5,
        )
    }
    fn conv(&self, x: Tensor<4>, p: &str, stride: usize, pad: usize, groups: usize) -> Tensor<4> {
        let _profile = crate::profile::span(&self.device, || format!("conv/{p} input={:?} stride={stride} groups={groups}", x.dims()));
        module::conv2d(
            x,
            self.w(&format!("{p}.weight")),
            Some(self.w(&format!("{p}.bias"))),
            ConvOptions::new([stride; 2], [pad; 2], [1; 2], groups),
        )
    }
    // Spatial attention uses an explicit flash launch. Short light attention
    // retains the reference's full-width scale and inexpensive small score matrices.
    fn attention(
        &self,
        q: Tensor<3>,
        k: Tensor<3>,
        v: Tensor<3>,
        heads: usize,
        scale: f32,
        spatial: bool,
    ) -> Tensor<3> {
        let [b, n, c] = q.dims();
        let m = k.dims()[1];
        let d = c / heads;
        let _profile = crate::profile::span(&self.device, || format!("attention/{} B={b} Q={n} K={m} C={c} heads={heads}", if spatial { "spatial" } else { "light" }));
        if !spatial {
            if let Some(out) = crate::light_attention::forward(q.clone(), k.clone(), v.clone(), heads) {
                return out;
            }
        }
        let q = q.reshape([b, n, heads, d]).permute([0, 2, 1, 3]);
        let k = k.reshape([b, m, heads, d]).permute([0, 2, 1, 3]);
        let v = v.reshape([b, m, heads, d]).permute([0, 2, 1, 3]);
        // Only use flash when sequence lengths are large enough for CMMA tiles (m >= 64, n >= 4096)
        let flash = spatial && n >= 4096 && m >= 64 && self.options.fused_attention && crate::attention::supported(&q, &k);
        let kt = k.clone().swap_dims(2, 3);
        let compute = |q: Tensor<4>| {
            let _profile = crate::profile::span(&self.device, || format!("attention/chunk {} Q={:?} K={m}", if flash { "flash" } else { "gemm" }, q.dims()));
            if flash {
                if let Some(out) = crate::attention::flash(q.clone(), k.clone(), v.clone()) {
                    return out;
                }
            }
            let scores = (q * scale).matmul(kt.clone());
            activation::softmax(scores, 3).matmul(v.clone())
        };
        let elem_bytes = if self.options.dtype == FloatDType::F16 { 2 } else { 4 };
        let score_limit = (64 * 1024 * 1024usize / (b * heads * m * elem_bytes)).max(1);
        // Stable power-of-two GEMM tiles avoid awkward 2730-query dispatches
        // (six FP16 images at 256px), while retaining a bounded score buffer.
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
        let z = self.norm(x.clone(), &format!("{p}.norm1"));
        let q = self.linear(z.clone(), &format!("{p}.attn.q"));
        let z = if sr > 1 {
            let a = self.conv(
                z.swap_dims(1, 2).reshape([b, c, h, w]),
                &format!("{p}.attn.sr"),
                sr,
                0,
                1,
            );
            let [_, _, hh, ww] = a.dims();
            self.norm(
                a.reshape([b, c, hh * ww]).swap_dims(1, 2),
                &format!("{p}.attn.norm"),
            )
        } else {
            z
        };
        let kv = self.linear(z, &format!("{p}.attn.kv"));
        let m = kv.dims()[1];
        let k = kv.clone().slice([0..b, 0..m, 0..c]);
        let v = kv.slice([0..b, 0..m, c..2 * c]);
        let a = self.attention(q, k, v, heads, 1.0 / (c as f32 / heads as f32).sqrt(), true);
        let x = x + self.linear(a, &format!("{p}.attn.proj"));
        let z = self.linear(
            self.norm(x.clone(), &format!("{p}.norm2")),
            &format!("{p}.mlp.fc1"),
        );
        let z = self.conv(
            z.swap_dims(1, 2).reshape([b, 4 * c, h, w]),
            &format!("{p}.mlp.dwconv.dwconv"),
            1,
            1,
            4 * c,
        );
        let z = activation::gelu(z.reshape([b, 4 * c, n]).swap_dims(1, 2));
        x + self.linear(z, &format!("{p}.mlp.fc2"))
    }
    fn mab(&self, x: Tensor<3>, y: Tensor<3>, p: &str) -> Tensor<3> {
        let x = if self.weights.contains_key(&format!("{p}.ln0.weight")) {
            self.norm(x, &format!("{p}.ln0"))
        } else {
            x
        };
        let q = self.linear(x, &format!("{p}.fc_q"));
        let c = q.dims()[2];
        let kv = self.linear(y, &format!("{p}.fc_kv"));
        let [batch, lights, _] = kv.dims();
        let k = kv.clone().slice([0..batch, 0..lights, 0..c]);
        let v = kv.slice([0..batch, 0..lights, c..2 * c]);
        // Set transformer uses sqrt(full dimension), unlike spatial attention.
        let o = q.clone() + self.attention(q, k, v, 4, 1.0 / (c as f32).sqrt(), false);
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
        let seed_weight = if pool {
            Some(self.w::<3>(&format!("{p}.dec.0.S")))
        } else {
            None
        };
        let out_n = if pool { 1 } else { n };
        let mut out = if b > self.options.pixel_batch {
            Some(Tensor::empty([b, out_n, c], self.tensor_options()))
        } else { None };
        for (chunk, start) in (0..b).step_by(self.options.pixel_batch).enumerate() {
            let end = (start + self.options.pixel_batch).min(b);
            let valid_pixels = end - start;
            // Pixels are independent in the set transformer. Bucket only the
            // last partial batch to reduce mask-dependent kernel/tuning shapes;
            // discard padded rows afterward. Never exceed the requested batch.
            let work_pixels = valid_pixels.div_ceil(256).saturating_mul(256)
                .min(self.options.pixel_batch);
            let mut z = if start == 0 && end == b {
                x.clone()
            } else {
                x.clone().slice([start..end, 0..n, 0..c])
            };
            if work_pixels > valid_pixels {
                z = Tensor::cat(vec![z, Tensor::zeros([work_pixels - valid_pixels, n, c], self.tensor_options())], 0);
            }
            for i in 0..3 {
                z = self.mab(z.clone(), z, &format!("{p}.enc.{i}.mab"));
            }
            if let Some(ref s) = seed_weight {
                let seed = s.clone().repeat_dim(0, work_pixels);
                z = self.mab(seed, z, &format!("{p}.dec.0.mab"));
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
    fn set_layer_masked(
        &self,
        x: Tensor<3>,
        p: &str,
        pool: bool,
        active_idx: Option<&Tensor<1, burn::tensor::Int>>,
        total_pixels: usize,
    ) -> Tensor<3> {
        let [_, n, c] = x.dims();
        let _profile = crate::profile::span(&self.device, || format!("masked-set/{p} pixels={total_pixels} active={} lights={n} C={c}", active_idx.map_or(total_pixels, |idx| idx.dims()[0])));
        let out_n = if pool { 1 } else { n };
        if let Some(idx) = active_idx {
            let gathered = x.select(0, idx.clone());
            let processed = self.set_layer(gathered, p, pool);
            let zeros = Tensor::zeros([total_pixels, out_n, c], self.tensor_options());
            zeros.select_assign(0, idx.clone(), processed, burn::tensor::IndexingUpdateOp::Add)
        } else {
            self.set_layer(x, p, pool)
        }
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
            if i > 0 {
                if let Some(prev) = current_mask.take() {
                    // Downsample mask by 2x (max-pooling 3x3 stride 2 pad 1 to match conv3x3 stride 2)
                    let mut next_mask = vec![false; oh * ow];
                    for y in 0..oh {
                        for x in 0..ow {
                            let sy = 2 * y;
                            let sx = 2 * x;
                            let mut any = false;
                            for dy in sy.saturating_sub(1)..=(sy + 1).min(ih - 1) {
                                for dx in sx.saturating_sub(1)..=(sx + 1).min(iw - 1) {
                                    if prev[dy * iw + dx] {
                                        any = true;
                                        break;
                                    }
                                }
                                if any { break; }
                            }
                            next_mask[y * ow + x] = any;
                        }
                    }
                    current_mask = Some(next_mask);
                }
            }
            let active_idx = current_mask.as_ref().and_then(|m| {
                let indices: Vec<i32> = m
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &keep)| if keep { Some(idx as i32) } else { None })
                    .collect();
                // Only use gather/scatter when it saves at least 10% of pixels and is non-empty
                if !indices.is_empty() && indices.len() * 10 < m.len() * 9 {
                    let len = indices.len();
                    Some(Tensor::<1, burn::tensor::Int>::from_data(
                        TensorData::new(indices, [len]),
                        &self.device,
                    ))
                } else {
                    None
                }
            });

            let mut encoded_chunks = Vec::with_capacity((lights + self.options.encoder_batch - 1) / self.options.encoder_batch);
            for light in (0..lights).step_by(self.options.encoder_batch) {
                let end = (light + self.options.encoder_batch).min(lights);
                let batch = end - light;
                let input_batch = if light == 0 && end == lights {
                    images.clone()
                } else {
                    images.clone().slice([light..end, 0..ic, 0..ih, 0..iw])
                };
                let z = self.conv(
                    input_batch,
                    &format!("{p}.proj"),
                    if i == 0 { 1 } else { 2 },
                    1,
                    1,
                );
                let [_, c, hh, ww] = z.dims();
                let mut z = self.norm(
                    z.reshape([batch, c, hh * ww]).swap_dims(1, 2),
                    &format!("{p}.norm"),
                );
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
                crate::execution::finish(&self.device);
            }
            let encoded = if encoded_chunks.len() == 1 {
                encoded_chunks.pop().unwrap()
            } else {
                Tensor::cat(encoded_chunks, 0)
            };
            let x = encoded.permute([1, 0, 2]);
            let pool = self
                .set_layer_masked(x.clone(), &format!("{root}.pool_block{}", i + 1), true, active_idx.as_ref(), oh * ow)
                .reshape([1, oh, ow, c])
                .permute([0, 3, 1, 2]);
            if i < 3 {
                images = self
                    .set_layer_masked(x, &format!("{root}.light_block{}", i + 1), false, active_idx.as_ref(), oh * ow)
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
                self.w(&format!("{p}.weight")),
                Some(self.w(&format!("{p}.bias"))),
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
