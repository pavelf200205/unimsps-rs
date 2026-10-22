//! Short light sequences: one subgroup per query/head, FP32 online softmax.
//! No cooperative matrices, fixed warp width, CUDA APIs or attention-score buffer.
use burn::tensor::Tensor as BurnTensor;
use burn_backend::{DType, ExecutionError, ops::{AttentionModuleOptions, FloatTensorOps, ModuleOps}};
use burn_cubecl::{CubeBackend, fusion::{CubeFusionHandle, FusionCubeRuntime}, ops::numeric::empty_device_contiguous_dtype};
use burn_fusion::{Fusion, custom::{CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, ScalarIr, StreamId, TensorIr}};
use cubecl::prelude::*;

type Gpu = Fusion<CubeBackend>;

#[cube(launch)]
fn short_attention<F: Float>(
    q: &Tensor<F>, k: &Tensor<F>, v: &Tensor<F>, out: &mut Tensor<F>,
    scale: f32, #[comptime] heads: usize, #[comptime] width: usize,
    #[comptime] slots: usize, #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    // A workgroup owns a fixed number of rows regardless of the subgroup width
    // chosen by the driver. Wider subgroups process additional rows in sequence.
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < q.shape(0) * q.shape(1) * heads;
        // Keep every subgroup participating in the collective, including padding
        // rows, so drivers need not support divergent subgroup control flow.
        let head = row % heads;
        let query = (row / heads) % q.shape(1);
        let batch = row / (heads * q.shape(1));
        let mut queries = Array::<f32>::new(slots);
        let mut accum = Array::<f32>::new(slots);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            queries[j] = 0.0;
            accum[j] = 0.0;
            if valid && channel < width {
                queries[j] = f32::cast_from(q[batch * q.stride(0) + query * q.stride(1)
                    + (head * width + channel) * q.stride(2)]);
            }
        }
        // Explicit constructors make these mutable runtime values in CubeCL's
        // expansion. A negated literal can otherwise remain a host-side f32.
        let mut maximum = f32::new(-3.402823e38_f32);
        let mut denominator = f32::new(0.0_f32);
        for key in 0..k.shape(1) {
            let mut dot = f32::new(0.0_f32);
            #[unroll]
            for j in 0..slots {
                let channel = lane + j * plane;
                if valid && channel < width {
                    dot += queries[j] * f32::cast_from(k[batch * k.stride(0) + key * k.stride(1)
                        + (head * width + channel) * k.stride(2)]);
                }
            }
            let score = plane_sum(dot) * scale;
            let next_maximum = f32::max(maximum, score);
            let correction = f32::exp(maximum - next_maximum);
            let probability = f32::exp(score - next_maximum);
            denominator = denominator * correction + probability;
            #[unroll]
            for j in 0..slots {
                let channel = lane + j * plane;
                if valid && channel < width {
                    let value = f32::cast_from(v[batch * v.stride(0) + key * v.stride(1)
                        + (head * width + channel) * v.stride(2)]);
                    accum[j] = accum[j] * correction + probability * value;
                }
            }
            maximum = next_maximum;
        }
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            if valid && channel < width {
                out[row * width + channel] = F::cast_from(accum[j] / denominator);
            }
        }
    }
}

#[derive(Debug)]
struct LightOp { desc: CustomOpIr }

impl Operation<FusionCubeRuntime> for LightOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([q, k, v], [output]) = self.desc.as_fixed();
        let [q, k, v] = [q, k, v].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let ScalarIr::UInt(heads) = self.desc.scalars[0] else { unreachable!() };
        let heads = heads as usize;
        let [batch, queries, channels] = q.meta.shape().dims::<3>();
        let width = channels / heads;
        let scale = 1.0 / (channels as f32).sqrt();
        let client = q.client.clone();
        let props = client.properties();
        let hw = &props.hardware;
        let can_launch = props.features.plane.contains(cubecl::ir::features::Plane::Ops)
            && hw.plane_size_min >= 2 && hw.plane_size_max <= 128
            && hw.max_units_per_cube >= 128 && hw.max_cube_dim.0 >= 128;
        let result = if can_launch {
            crate::profile::event(|| format!("light/subgroup B={batch} Q={queries} C={channels} subgroup={}..{} cmma_configs={}", hw.plane_size_min, hw.plane_size_max, props.features.matmul.cmma.len()));
            static NOTICE: std::sync::Once = std::sync::Once::new();
            NOTICE.call_once(|| eprintln!("Light attention: fused subgroup kernel (variable subgroup width, FP32 accumulation)"));
            let slots = width.div_ceil(hw.plane_size_min as usize);
            let rows_per_group = 128 / hw.plane_size_min as usize;
            let groups = (batch * queries * heads).div_ceil(rows_per_group);
            let gx = groups.min(hw.max_cube_count.0 as usize).max(1);
            let gy = groups.div_ceil(gx);
            if gy > hw.max_cube_count.1 as usize {
                return Err(ExecutionError::generic("Light attention dispatch exceeds device grid limits"));
            }
            let result = empty_device_contiguous_dtype(client.clone(), q.device.clone(), q.meta.shape().clone(), q.dtype);
            let count = CubeCount::Static(gx as u32, gy as u32, 1);
            let dim = CubeDim::new_1d(128);
            match q.dtype {
                DType::F16 => short_attention::launch::<half::f16>(&client, count, dim,
                    q.into_tensor_arg(), k.into_tensor_arg(), v.into_tensor_arg(), result.clone().into_tensor_arg(),
                    scale, heads, width, slots, rows_per_group),
                DType::F32 => short_attention::launch::<f32>(&client, count, dim,
                    q.into_tensor_arg(), k.into_tensor_arg(), v.into_tensor_arg(), result.clone().into_tensor_arg(),
                    scale, heads, width, slots, rows_per_group),
                _ => return Err(ExecutionError::generic("Unsupported light attention dtype")),
            }
            result
        } else {
            crate::profile::event(|| format!("light/Burn-fallback B={batch} Q={queries} C={channels}"));
            // Devices without subgroup operations retain the normal Burn path.
            let keys = k.meta.shape[1];
            let q = CubeBackend::float_swap_dims(CubeBackend::float_reshape(q, [batch, queries, heads, width].into()), 1, 2);
            let k = CubeBackend::float_swap_dims(CubeBackend::float_reshape(k, [batch, keys, heads, width].into()), 1, 2);
            let v = CubeBackend::float_swap_dims(CubeBackend::float_reshape(v, [batch, keys, heads, width].into()), 1, 2);
            let out = CubeBackend::attention(q, k, v, None, None, AttentionModuleOptions {
                scale: Some(scale as f64), softcap: None, is_causal: false,
            });
            CubeBackend::float_reshape(CubeBackend::float_swap_dims(out, 1, 2), [batch, queries, channels].into())
        };
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}

pub(crate) fn forward(q: BurnTensor<3>, k: BurnTensor<3>, v: BurnTensor<3>, heads: usize) -> Option<BurnTensor<3>> {
    if k.dims()[1] > 32 { return None; }
    let q = q.inner().try_into_primitive::<Gpu>().ok()?;
    let k = k.inner().try_into_primitive::<Gpu>().ok()?;
    let v = v.inner().try_into_primitive::<Gpu>().ok()?;
    let client = q.client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), q.shape.clone(), q.dtype);
    let desc = CustomOpIr::with_scalars("unimsps_short_light_attention",
        &[q.into_ir(), k.into_ir(), v.into_ir()], &[out], vec![ScalarIr::UInt(heads as u64)]);
    let [out] = client.register(StreamId::current(), OperationIr::Custom(desc.clone()), LightOp { desc }).outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}
