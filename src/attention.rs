//! Spatial attention dispatch:
//! 1. Uses BlackboxAccelerated flash attention when cooperative matrix (CMMA) tiles are supported.
//! 2. Uses a fused 2-queries-per-subgroup WGPU online-softmax kernel (`spatial_subgroup_attention`)
//!    consuming packed KV `[B, K, 2*C]` directly when CMMA is unavailable, eliminating quadratic score
//!    buffers, 4D permutes, KV slice kernels, and thousands of chunked GEMM dispatches.
use burn::tensor::Tensor as BurnTensor;
use burn_backend::{
    DType, ExecutionError,
    ops::AttentionModuleOptions,
};
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
    kernel::attention::flash_attention,
    ops::numeric::empty_device_contiguous_dtype,
};
use burn_fusion::{
    Fusion,
    custom::{CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, ScalarIr, StreamId, TensorIr},
};
use cubecl::prelude::*;
use cubek::attention::forward::{
    launch::{BlueprintStrategy, Strategy},
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};
use std::{collections::HashSet, sync::{Mutex, OnceLock}};

type Gpu = Fusion<CubeBackend>;

// Availability depends on the device AND problem shape. A small-stage failure
// must not disable acceleration for all later stages or another GPU.
fn unavailable() -> &'static Mutex<HashSet<String>> {
    static CACHE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashSet::new()))
}

fn problem_key(q: &BurnTensor<4>, k: &BurnTensor<4>) -> String {
    format!("{:?}/{:?}/{:?}/{:?}", q.device().identity(), q.dtype(), q.dims(), k.dims())
}

#[cube(launch)]
fn spatial_subgroup_attention_4x<F: Float>(
    q: &Tensor<F>,
    kv: &Tensor<F>,
    out: &mut Tensor<F>,
    scale: f32,
    #[comptime] heads: usize,
    #[comptime] width: usize,
    #[comptime] slots: usize,
    #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let q_quads = q.shape(1) / 4;
    // Each subgroup processes 4 adjacent queries in the same (batch, head),
    // quadrupling K/V load reuse and halving threadblock dispatch count.
    for tile in 0..rows_per_group / planes {
        let quad_row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = quad_row < q.shape(0) * q_quads * heads;
        let quad_idx = quad_row % q_quads;
        let query0 = quad_idx * 4;
        let query1 = query0 + 1;
        let query2 = query0 + 2;
        let query3 = query0 + 3;
        let head = (quad_row / q_quads) % heads;
        let batch = quad_row / (q_quads * heads);
        let mut queries0 = Array::<f32>::new(slots);
        let mut queries1 = Array::<f32>::new(slots);
        let mut queries2 = Array::<f32>::new(slots);
        let mut queries3 = Array::<f32>::new(slots);
        let mut accum0 = Array::<f32>::new(slots);
        let mut accum1 = Array::<f32>::new(slots);
        let mut accum2 = Array::<f32>::new(slots);
        let mut accum3 = Array::<f32>::new(slots);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            queries0[j] = 0.0;
            queries1[j] = 0.0;
            queries2[j] = 0.0;
            queries3[j] = 0.0;
            accum0[j] = 0.0;
            accum1[j] = 0.0;
            accum2[j] = 0.0;
            accum3[j] = 0.0;
            if valid && channel < width {
                let c_off = (head * width + channel) * q.stride(2);
                let b_off = batch * q.stride(0);
                let q_stride = q.stride(1);
                queries0[j] = f32::cast_from(q[b_off + query0 * q_stride + c_off]);
                queries1[j] = f32::cast_from(q[b_off + query1 * q_stride + c_off]);
                queries2[j] = f32::cast_from(q[b_off + query2 * q_stride + c_off]);
                queries3[j] = f32::cast_from(q[b_off + query3 * q_stride + c_off]);
            }
        }
        let mut maximum0 = f32::new(-3.402823e38_f32);
        let mut maximum1 = f32::new(-3.402823e38_f32);
        let mut maximum2 = f32::new(-3.402823e38_f32);
        let mut maximum3 = f32::new(-3.402823e38_f32);
        let mut denominator0 = f32::new(0.0_f32);
        let mut denominator1 = f32::new(0.0_f32);
        let mut denominator2 = f32::new(0.0_f32);
        let mut denominator3 = f32::new(0.0_f32);
        for key in 0..kv.shape(1) {
            let mut dot0 = f32::new(0.0_f32);
            let mut dot1 = f32::new(0.0_f32);
            let mut dot2 = f32::new(0.0_f32);
            let mut dot3 = f32::new(0.0_f32);
            let kv_base = batch * kv.stride(0) + key * kv.stride(1);
            #[unroll]
            for j in 0..slots {
                let channel = lane + j * plane;
                if valid && channel < width {
                    let k_val = f32::cast_from(kv[kv_base + (head * width + channel) * kv.stride(2)]);
                    dot0 += queries0[j] * k_val;
                    dot1 += queries1[j] * k_val;
                    dot2 += queries2[j] * k_val;
                    dot3 += queries3[j] * k_val;
                }
            }
            let score0 = plane_sum(dot0) * scale;
            let score1 = plane_sum(dot1) * scale;
            let score2 = plane_sum(dot2) * scale;
            let score3 = plane_sum(dot3) * scale;

            let next_max0 = f32::max(maximum0, score0);
            let next_max1 = f32::max(maximum1, score1);
            let next_max2 = f32::max(maximum2, score2);
            let next_max3 = f32::max(maximum3, score3);

            let corr0 = f32::exp(maximum0 - next_max0);
            let corr1 = f32::exp(maximum1 - next_max1);
            let corr2 = f32::exp(maximum2 - next_max2);
            let corr3 = f32::exp(maximum3 - next_max3);

            let prob0 = f32::exp(score0 - next_max0);
            let prob1 = f32::exp(score1 - next_max1);
            let prob2 = f32::exp(score2 - next_max2);
            let prob3 = f32::exp(score3 - next_max3);

            denominator0 = denominator0 * corr0 + prob0;
            denominator1 = denominator1 * corr1 + prob1;
            denominator2 = denominator2 * corr2 + prob2;
            denominator3 = denominator3 * corr3 + prob3;

            #[unroll]
            for j in 0..slots {
                let channel = lane + j * plane;
                if valid && channel < width {
                    let v_val = f32::cast_from(kv[kv_base + ((heads + head) * width + channel) * kv.stride(2)]);
                    accum0[j] = accum0[j] * corr0 + prob0 * v_val;
                    accum1[j] = accum1[j] * corr1 + prob1 * v_val;
                    accum2[j] = accum2[j] * corr2 + prob2 * v_val;
                    accum3[j] = accum3[j] * corr3 + prob3 * v_val;
                }
            }
            maximum0 = next_max0;
            maximum1 = next_max1;
            maximum2 = next_max2;
            maximum3 = next_max3;
        }
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            if valid && channel < width {
                let out_c = (head * width + channel) * out.stride(2);
                let out_b = batch * out.stride(0);
                let out_stride = out.stride(1);
                out[out_b + query0 * out_stride + out_c] = F::cast_from(accum0[j] / denominator0);
                out[out_b + query1 * out_stride + out_c] = F::cast_from(accum1[j] / denominator1);
                out[out_b + query2 * out_stride + out_c] = F::cast_from(accum2[j] / denominator2);
                out[out_b + query3 * out_stride + out_c] = F::cast_from(accum3[j] / denominator3);
            }
        }
    }
}

#[cube(launch)]
fn spatial_subgroup_attention<F: Float>(
    q: &Tensor<F>,
    kv: &Tensor<F>,
    out: &mut Tensor<F>,
    scale: f32,
    #[comptime] heads: usize,
    #[comptime] width: usize,
    #[comptime] slots: usize,
    #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let q_pairs = q.shape(1) / 2;
    // Each subgroup processes 2 adjacent queries in the same (batch, head),
    // reusing each K/V load across both queries in registers while all subgroups
    // in the workgroup broadcast the exact same K/V cache lines.
    for tile in 0..rows_per_group / planes {
        let pair_row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = pair_row < q.shape(0) * q_pairs * heads;
        let pair_idx = pair_row % q_pairs;
        let query0 = pair_idx * 2;
        let query1 = query0 + 1;
        let head = (pair_row / q_pairs) % heads;
        let batch = pair_row / (q_pairs * heads);
        let mut queries0 = Array::<f32>::new(slots);
        let mut queries1 = Array::<f32>::new(slots);
        let mut accum0 = Array::<f32>::new(slots);
        let mut accum1 = Array::<f32>::new(slots);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            queries0[j] = 0.0;
            queries1[j] = 0.0;
            accum0[j] = 0.0;
            accum1[j] = 0.0;
            if valid && channel < width {
                let c_off = (head * width + channel) * q.stride(2);
                let b_off = batch * q.stride(0);
                queries0[j] = f32::cast_from(q[b_off + query0 * q.stride(1) + c_off]);
                queries1[j] = f32::cast_from(q[b_off + query1 * q.stride(1) + c_off]);
            }
        }
        let mut maximum0 = f32::new(-3.402823e38_f32);
        let mut maximum1 = f32::new(-3.402823e38_f32);
        let mut denominator0 = f32::new(0.0_f32);
        let mut denominator1 = f32::new(0.0_f32);
        for key in 0..kv.shape(1) {
            let mut dot0 = f32::new(0.0_f32);
            let mut dot1 = f32::new(0.0_f32);
            let kv_base = batch * kv.stride(0) + key * kv.stride(1);
            #[unroll]
            for j in 0..slots {
                let channel = lane + j * plane;
                if valid && channel < width {
                    let k_val = f32::cast_from(kv[kv_base + (head * width + channel) * kv.stride(2)]);
                    dot0 += queries0[j] * k_val;
                    dot1 += queries1[j] * k_val;
                }
            }
            let score0 = plane_sum(dot0) * scale;
            let score1 = plane_sum(dot1) * scale;
            let next_max0 = f32::max(maximum0, score0);
            let next_max1 = f32::max(maximum1, score1);
            let corr0 = f32::exp(maximum0 - next_max0);
            let corr1 = f32::exp(maximum1 - next_max1);
            let prob0 = f32::exp(score0 - next_max0);
            let prob1 = f32::exp(score1 - next_max1);
            denominator0 = denominator0 * corr0 + prob0;
            denominator1 = denominator1 * corr1 + prob1;
            #[unroll]
            for j in 0..slots {
                let channel = lane + j * plane;
                if valid && channel < width {
                    let v_val = f32::cast_from(kv[kv_base + ((heads + head) * width + channel) * kv.stride(2)]);
                    accum0[j] = accum0[j] * corr0 + prob0 * v_val;
                    accum1[j] = accum1[j] * corr1 + prob1 * v_val;
                }
            }
            maximum0 = next_max0;
            maximum1 = next_max1;
        }
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            if valid && channel < width {
                let out_c = (head * width + channel) * out.stride(2);
                let out_b = batch * out.stride(0);
                let out_stride = out.stride(1);
                let out_index0 = out_b + query0 * out_stride + out_c;
                let out_index1 = out_b + query1 * out_stride + out_c;
                out[out_index0] = F::cast_from(accum0[j] / denominator0);
                out[out_index1] = F::cast_from(accum1[j] / denominator1);
            }
        }
    }
}

#[derive(Debug)]
struct SpatialSubgroupOp {
    desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for SpatialSubgroupOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([q, kv], [output]) = self.desc.as_fixed();
        let [q, kv] = [q, kv].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let ScalarIr::UInt(heads) = self.desc.scalars[0] else { unreachable!() };
        let heads = heads as usize;
        let [batch, queries, channels] = q.meta.shape().dims::<3>();
        let keys = kv.meta.shape().dims::<3>()[1];
        let width = channels / heads;
        let scale = 1.0 / (width as f32).sqrt();
        let client = q.client.clone();
        let props = client.properties();
        let hw = &props.hardware;
        crate::profile::event(|| format!(
            "spatial/subgroup B={batch} Q={queries} K={keys} C={channels} heads={heads} subgroup={}..{}",
            hw.plane_size_min, hw.plane_size_max
        ));
        let slots = width.div_ceil(hw.plane_size_min as usize);
        let rows_per_group = 128 / hw.plane_size_min as usize;
        let dim = CubeDim::new_1d(128);
        let result = empty_device_contiguous_dtype(client.clone(), q.device.clone(), q.meta.shape().clone(), q.dtype);

        if queries % 4 == 0 {
            static NOTICE_4X: std::sync::Once = std::sync::Once::new();
            NOTICE_4X.call_once(|| eprintln!(
                "Spatial attention: fused 4x-query subgroup online-softmax kernel (packed KV, FP32 accumulation)"
            ));
            let q_quads = queries / 4;
            let groups = (batch * q_quads * heads).div_ceil(rows_per_group);
            let max_gx = (hw.max_cube_count.0 as usize).max(1);
            let gy = groups.div_ceil(max_gx).max(1);
            let gx = groups.div_ceil(gy).max(1);
            if gy > hw.max_cube_count.1 as usize {
                return Err(ExecutionError::generic("Spatial subgroup attention dispatch exceeds device grid limits"));
            }
            let count = CubeCount::Static(gx as u32, gy as u32, 1);
            match q.dtype {
                DType::F16 => spatial_subgroup_attention_4x::launch::<half::f16>(
                    &client, count, dim,
                    q.into_tensor_arg(), kv.into_tensor_arg(),
                    result.clone().into_tensor_arg(),
                    scale, heads, width, slots, rows_per_group,
                ),
                DType::F32 => spatial_subgroup_attention_4x::launch::<f32>(
                    &client, count, dim,
                    q.into_tensor_arg(), kv.into_tensor_arg(),
                    result.clone().into_tensor_arg(),
                    scale, heads, width, slots, rows_per_group,
                ),
                _ => return Err(ExecutionError::generic("Unsupported spatial attention dtype")),
            }
        } else {
            static NOTICE: std::sync::Once = std::sync::Once::new();
            NOTICE.call_once(|| eprintln!(
                "Spatial attention: fused 2x-query subgroup online-softmax kernel (packed KV, FP32 accumulation)"
            ));
            let q_pairs = queries / 2;
            let groups = (batch * q_pairs * heads).div_ceil(rows_per_group);
            let max_gx = (hw.max_cube_count.0 as usize).max(1);
            let gy = groups.div_ceil(max_gx).max(1);
            let gx = groups.div_ceil(gy).max(1);
            if gy > hw.max_cube_count.1 as usize {
                return Err(ExecutionError::generic("Spatial subgroup attention dispatch exceeds device grid limits"));
            }
            let count = CubeCount::Static(gx as u32, gy as u32, 1);
            match q.dtype {
                DType::F16 => spatial_subgroup_attention::launch::<half::f16>(
                    &client, count, dim,
                    q.into_tensor_arg(), kv.into_tensor_arg(),
                    result.clone().into_tensor_arg(),
                    scale, heads, width, slots, rows_per_group,
                ),
                DType::F32 => spatial_subgroup_attention::launch::<f32>(
                    &client, count, dim,
                    q.into_tensor_arg(), kv.into_tensor_arg(),
                    result.clone().into_tensor_arg(),
                    scale, heads, width, slots, rows_per_group,
                ),
                _ => return Err(ExecutionError::generic("Unsupported spatial attention dtype")),
            }
        }
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}

#[derive(Debug)]
struct FlashOp {
    desc: CustomOpIr,
    problem: String,
}

impl Operation<FusionCubeRuntime> for FlashOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([q, k, v], [out]) = self.desc.as_fixed();
        let [q, k, v] = [q, k, v].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let options = AttentionModuleOptions { scale: None, softcap: None, is_causal: false };

        let accelerated = flash_attention(
            q.clone(), k.clone(), v.clone(), None, None, options,
            Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(
                BlackboxAcceleratedStrategy { num_planes: 4, seq_q: 1, seq_kv: 1 },
            )),
        ).or_else(|_| {
            flash_attention(
                q.clone(), k.clone(), v.clone(), None, None, options,
                Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(
                    BlackboxAcceleratedStrategy { num_planes: 2, seq_q: 1, seq_kv: 1 },
                )),
            )
        });

        let result = match accelerated {
            Ok(result) => {
                crate::profile::event(|| format!("spatial/accelerated-flash {}", self.problem));
                static NOTICE: std::sync::Once = std::sync::Once::new();
                NOTICE.call_once(|| eprintln!("Spatial attention: accelerated flash"));
                result
            }
            Err(accelerated_error) => {
                crate::profile::event(|| format!("spatial/unit-flash-after-failure {}", self.problem));
                unavailable().lock().unwrap().insert(self.problem.clone());
                let result = flash_attention(
                    q, k, v, None, None, options,
                    Strategy::Unit(BlueprintStrategy::Inferred(())),
                )
                .map_err(|unit_error| ExecutionError::generic(format!(
                    "Flash attention unavailable: accelerated={accelerated_error:?}; unit={unit_error:?}"
                )))?;
                static NOTICE: std::sync::Once = std::sync::Once::new();
                NOTICE.call_once(|| eprintln!(
                    "Spatial attention: accelerated flash unavailable for this shape ({accelerated_error:?}); caching fallback"
                ));
                result
            }
        };
        handles.register_float_tensor::<CubeBackend>(&out.id, result);
        Ok(())
    }
}

pub(crate) fn supported(q: &BurnTensor<4>, k: &BurnTensor<4>) -> bool {
    let Ok(primitive) = q.clone().inner().try_into_primitive::<Gpu>() else { return false; };
    let client = primitive.client.device().client();
    if client.properties().features.matmul.cmma.is_empty() {
        return false;
    }
    !unavailable().lock().unwrap().contains(&problem_key(q, k))
}

pub(crate) fn subgroup_supported(q: &BurnTensor<3>) -> bool {
    if q.dims()[1] % 2 != 0 {
        return false;
    }
    let Ok(primitive) = q.clone().inner().try_into_primitive::<Gpu>() else { return false; };
    let client = primitive.client.device().client();
    let props = client.properties();
    let hw = &props.hardware;
    props.features.plane.contains(cubecl::ir::features::Plane::Ops)
        && hw.plane_size_min >= 2
        && hw.plane_size_max <= 128
        && hw.max_units_per_cube >= 128
        && hw.max_cube_dim.0 >= 128
}

pub(crate) fn spatial_subgroup(
    q: BurnTensor<3>,
    kv: BurnTensor<3>,
    heads: usize,
) -> Option<BurnTensor<3>> {
    if !subgroup_supported(&q) {
        return None;
    }
    let q = q.inner().try_into_primitive::<Gpu>().ok()?;
    let kv = kv.inner().try_into_primitive::<Gpu>().ok()?;
    let client = q.client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), q.shape.clone(), q.dtype);
    let desc = CustomOpIr::with_scalars(
        "unimsps_spatial_subgroup_attention",
        &[q.into_ir(), kv.into_ir()],
        &[out],
        vec![ScalarIr::UInt(heads as u64)],
    );
    let [out] = client.register(
        StreamId::current(),
        OperationIr::Custom(desc.clone()),
        SpatialSubgroupOp { desc },
    ).outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}

pub(crate) fn flash(q: BurnTensor<4>, k: BurnTensor<4>, v: BurnTensor<4>) -> Option<BurnTensor<4>> {
    // Do not re-check the cache here: queued execution can update it between
    // dispatch planning and registration. The caller has already sized its work.
    let problem = problem_key(&q, &k);
    let q = q.inner().try_into_primitive::<Gpu>().ok()?;
    let k = k.inner().try_into_primitive::<Gpu>().ok()?;
    let v = v.inner().try_into_primitive::<Gpu>().ok()?;
    let client = q.client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), q.shape.clone(), q.dtype);
    let desc = CustomOpIr::with_scalars(
        "unimsps_spatial_flash", &[q.into_ir(), k.into_ir(), v.into_ir()], &[out], vec![],
    );
    let [out] = client.register(StreamId::current(), OperationIr::Custom(desc.clone()), FlashOp { desc, problem }).outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}
