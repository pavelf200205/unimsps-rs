#![allow(unstable_name_collisions)]
//! Direct 3x3 depthwise convolution on [batch, spatial token, channel].
//! No NCHW materialization; FP32 accumulation and native output precision.
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
fn token_depthwise<F: Float>(x: &Tensor<F>, weight: &Tensor<F>, bias: &Tensor<F>, out: &mut Tensor<F>, #[comptime] height: usize, #[comptime] width: usize, #[comptime] channels: usize) {
    let index = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if index < out.len() {
        let ch = index % channels;
        let pixel = index / channels;
        let batch = pixel / (height * width);
        let y = (pixel / width) % height;
        let xx = pixel % width;
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for ky in 0..3usize {
            #[unroll]
            for kx in 0..3usize {
                // Offset by one before subtracting to avoid unsigned underflow.
                let yy = y + ky;
                let sx = xx + kx;
                if yy >= 1 && yy <= height && sx >= 1 && sx <= width {
                    let token = (yy - 1) * width + sx - 1;
                    let a = f32::cast_from(x[batch * x.stride(0) + token * x.stride(1) + ch * x.stride(2)]);
                    let w = f32::cast_from(weight[ch * weight.stride(0) + ky * weight.stride(2) + kx * weight.stride(3)]);
                    sum += a * w;
                }
            }
        }
        // Round convolution to its native dtype before the activation, as in
        // the unfused path. Evaluate the erf GELU (not the tanh approximation).
        let conv = F::cast_from(sum + f32::cast_from(bias[ch * bias.stride(0)]));
        let value = f32::cast_from(conv);
        out[index] = F::cast_from(0.5_f32 * value * (1.0_f32 + f32::erf(value * 0.7071067811865476_f32)));
    }
}

#[derive(Debug)]
struct DepthwiseOp { desc: CustomOpIr, height: usize, width: usize }
impl Operation<FusionCubeRuntime> for DepthwiseOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([x, weight, bias], [output]) = self.desc.as_fixed();
        let [x, weight, bias] = [x, weight, bias].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let [batch, tokens, channels] = x.meta.shape().dims::<3>();
        crate::profile::event(|| format!("mlp/token-depthwise-gelu B={batch} N={tokens} C={channels}"));
        let client = x.client.clone();
        let hw = &client.properties().hardware;
        let groups = (batch * tokens * channels).div_ceil(128);
        let gy = groups.div_ceil((hw.max_cube_count.0 as usize).max(1)).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize { return Err(ExecutionError::generic("Depthwise dispatch exceeds device grid limits")); }
        let result = empty_device_contiguous_dtype(client.clone(), x.device.clone(), x.meta.shape().clone(), x.dtype);
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        match x.dtype {
            DType::F16 => token_depthwise::launch::<half::f16>(&client, count, dim, x.into_tensor_arg(), weight.into_tensor_arg(), bias.into_tensor_arg(), result.clone().into_tensor_arg(), self.height, self.width, channels),
            DType::F32 => token_depthwise::launch::<f32>(&client, count, dim, x.into_tensor_arg(), weight.into_tensor_arg(), bias.into_tensor_arg(), result.clone().into_tensor_arg(), self.height, self.width, channels),
            _ => return Err(ExecutionError::generic("Unsupported depthwise dtype")),
        }
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}

pub(crate) fn forward(x: BurnTensor<3>, weight: BurnTensor<4>, bias: BurnTensor<1>, height: usize, width: usize) -> Option<BurnTensor<3>> {
    let [_, tokens, channels] = x.dims();
    if tokens != height * width || weight.dims() != [channels, 1, 3, 3] || bias.dims() != [channels] { return None; }
    let x = x.inner().try_into_primitive::<Gpu>().ok()?;
    let weight = weight.inner().try_into_primitive::<Gpu>().ok()?;
    let bias = bias.inner().try_into_primitive::<Gpu>().ok()?;
    let client = x.client.clone();
    let compute = client.device().client();
    let hw = &compute.properties().hardware;
    if hw.max_units_per_cube < 128 || hw.max_cube_dim.0 < 128 { return None; }
    let output = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
    let desc = CustomOpIr::with_scalars("unimsps_token_depthwise", &[x.into_ir(), weight.into_ir(), bias.into_ir()], &[output], vec![]);
    let [out] = client.register(StreamId::current(), OperationIr::Custom(desc.clone()), DepthwiseOp { desc, height, width }).outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}
