//! Single-pass subgroup LayerNorm over the channel axis of [B, N, C] tensors.
//! Replaces Burn's multi-pass reduction + elementwise LayerNorm (which triggered
//! 878 shape compilations/autotunings and ~11s in profile.json) with 4 shape-agnostic
//! subgroup kernels (C = 64, 128, 256, 512) accumulating mean and variance in FP32 registers.
use burn::tensor::Tensor as BurnTensor;
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

type Gpu = Fusion<CubeBackend>;

#[cube(launch)]
fn subgroup_layer_norm<F: Float>(
    x: &Tensor<F>,
    gamma: &Tensor<F>,
    beta: &Tensor<F>,
    out: &mut Tensor<F>,
    inv_c: f32,
    #[comptime] channels: usize,
    #[comptime] slots: usize,
    #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < x.shape(0) * x.shape(1);
        let batch = row / x.shape(1);
        let seq = row % x.shape(1);
        let mut vals = Array::<f32>::new(slots);
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for j in 0..slots {
            let ch = lane + j * plane;
            vals[j] = 0.0;
            if valid && ch < channels {
                let v = f32::cast_from(x[batch * x.stride(0) + seq * x.stride(1) + ch * x.stride(2)]);
                vals[j] = v;
                sum += v;
            }
        }
        let mean = plane_sum(sum) * inv_c;
        let mut var_sum = f32::new(0.0_f32);
        #[unroll]
        for j in 0..slots {
            let ch = lane + j * plane;
            if valid && ch < channels {
                let diff = vals[j] - mean;
                var_sum += diff * diff;
            }
        }
        let inv_std = 1.0_f32 / f32::sqrt(plane_sum(var_sum) * inv_c + 1e-5_f32);
        #[unroll]
        for j in 0..slots {
            let ch = lane + j * plane;
            if valid && ch < channels {
                let g = f32::cast_from(gamma[ch * gamma.stride(0)]);
                let b = f32::cast_from(beta[ch * beta.stride(0)]);
                out[row * channels + ch] = F::cast_from((vals[j] - mean) * inv_std * g + b);
            }
        }
    }
}

#[derive(Debug)]
struct NormOp {
    desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for NormOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([x, gamma, beta], [output]) = self.desc.as_fixed();
        let [x, gamma, beta] = [x, gamma, beta].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let [batch, seq, channels] = x.meta.shape().dims::<3>();
        let client = x.client.clone();
        let props = client.properties();
        let hw = &props.hardware;
        static NOTICE: std::sync::Once = std::sync::Once::new();
        NOTICE.call_once(|| eprintln!("LayerNorm: fused single-pass subgroup kernel (FP32 register stats)"));
        let slots = channels.div_ceil(hw.plane_size_min as usize);
        let rows_per_group = 128 / hw.plane_size_min as usize;
        let groups = (batch * seq).div_ceil(rows_per_group);
        let max_gx = (hw.max_cube_count.0 as usize).max(1);
        let gy = groups.div_ceil(max_gx).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic("LayerNorm dispatch exceeds device grid limits"));
        }
        let result = empty_device_contiguous_dtype(
            client.clone(),
            x.device.clone(),
            x.meta.shape().clone(),
            x.dtype,
        );
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        let inv_c = 1.0f32 / (channels as f32);
        match x.dtype {
            DType::F16 => subgroup_layer_norm::launch::<half::f16>(
                &client,
                count,
                dim,
                x.into_tensor_arg(),
                gamma.into_tensor_arg(),
                beta.into_tensor_arg(),
                result.clone().into_tensor_arg(),
                inv_c,
                channels,
                slots,
                rows_per_group,
            ),
            DType::F32 => subgroup_layer_norm::launch::<f32>(
                &client,
                count,
                dim,
                x.into_tensor_arg(),
                gamma.into_tensor_arg(),
                beta.into_tensor_arg(),
                result.clone().into_tensor_arg(),
                inv_c,
                channels,
                slots,
                rows_per_group,
            ),
            _ => return Err(ExecutionError::generic("Unsupported LayerNorm dtype")),
        }
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}

pub(crate) fn forward(
    x: BurnTensor<3>,
    gamma: BurnTensor<1>,
    beta: BurnTensor<1>,
) -> Option<BurnTensor<3>> {
    let channels = x.dims()[2];
    let x = x.inner().try_into_primitive::<Gpu>().ok()?;
    let gamma = gamma.inner().try_into_primitive::<Gpu>().ok()?;
    let beta = beta.inner().try_into_primitive::<Gpu>().ok()?;
    let client = x.client.clone();
    let compute_client = client.device().client();
    let props = compute_client.properties();
    let hw = &props.hardware;
    let can_launch = props.features.plane.contains(cubecl::ir::features::Plane::Ops)
        && hw.plane_size_min >= 2
        && hw.plane_size_max <= 128
        && hw.max_units_per_cube >= 128
        && hw.max_cube_dim.0 >= 128
        && channels <= 1024;
    if !can_launch {
        return None;
    }
    let out = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
    let desc = CustomOpIr::with_scalars(
        "unimsps_subgroup_layer_norm",
        &[x.into_ir(), gamma.into_ir(), beta.into_ir()],
        &[out],
        vec![],
    );
    let [out] = client
        .register(StreamId::current(), OperationIr::Custom(desc.clone()), NormOp { desc })
        .outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}
