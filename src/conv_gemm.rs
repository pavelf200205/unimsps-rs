//! Bounded im2col on token tensors, followed by the backend's native GEMM.
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
fn pack<F: Float>(x: &Tensor<F>, out: &mut Tensor<F>, start: usize,
    #[comptime] height: usize, #[comptime] width: usize, #[comptime] channels: usize,
    #[comptime] kernel: usize, #[comptime] stride: usize, #[comptime] pad: usize,
    #[comptime] output_h: usize, #[comptime] output_w: usize) {
    let index = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if index < out.len() {
        let reduction = channels * kernel * kernel;
        let row = start + index / reduction;
        let k = index % reduction;
        let ch = k % channels;
        let ky = k / (channels * kernel);
        let kx = (k / channels) % kernel;
        let batch = row / (output_h * output_w);
        let y = (row / output_w) % output_h * stride + ky;
        let xx = row % output_w * stride + kx;
        let mut value = F::new(0.0_f32);
        if y >= pad && y < height + pad && xx >= pad && xx < width + pad {
            let token = (y - pad) * width + xx - pad;
            value = x[batch * x.stride(0) + token * x.stride(1) + ch * x.stride(2)];
        }
        out[index] = value;
    }
}
#[derive(Debug)]
struct PackOp { desc: CustomOpIr, height: usize, width: usize, kernel: usize, stride: usize, pad: usize, start: usize, rows: usize }
impl Operation<FusionCubeRuntime> for PackOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([input], [output]) = self.desc.as_fixed();
        let x = handles.get_float_tensor::<CubeBackend>(input);
        let channels = x.meta.shape().dims::<3>()[2];
        let client = x.client.clone();
        let hw = &client.properties().hardware;
        let count_elements = self.rows * channels * self.kernel * self.kernel;
        let groups = count_elements.div_ceil(128);
        let gy = groups.div_ceil((hw.max_cube_count.0 as usize).max(1)).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize { return Err(ExecutionError::generic("Convolution packing exceeds device grid limits")); }
        let shape = burn_backend::Shape::new([self.rows, channels * self.kernel * self.kernel]);
        let result = empty_device_contiguous_dtype(client.clone(), x.device.clone(), shape, x.dtype);
        let oh = (self.height + 2 * self.pad - self.kernel) / self.stride + 1;
        let ow = (self.width + 2 * self.pad - self.kernel) / self.stride + 1;
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        match x.dtype {
            DType::F16 => pack::launch::<half::f16>(&client, count, dim, x.into_tensor_arg(), result.clone().into_tensor_arg(), self.start, self.height, self.width, channels, self.kernel, self.stride, self.pad, oh, ow),
            DType::F32 => pack::launch::<f32>(&client, count, dim, x.into_tensor_arg(), result.clone().into_tensor_arg(), self.start, self.height, self.width, channels, self.kernel, self.stride, self.pad, oh, ow),
            _ => return Err(ExecutionError::generic("Unsupported convolution packing dtype")),
        }
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}
pub(crate) fn supported(x: &BurnTensor<3>) -> bool {
    let Ok(x) = x.clone().inner().try_into_primitive::<Gpu>() else { return false; };
    let compute = x.client.device().client();
    let hw = &compute.properties().hardware;
    hw.max_units_per_cube >= 128 && hw.max_cube_dim.0 >= 128
}
pub(crate) fn patches(x: BurnTensor<3>, height: usize, width: usize, kernel: usize, stride: usize, pad: usize, start: usize, rows: usize) -> BurnTensor<2> {
    let channels = x.dims()[2];
    let x = x.inner().try_into_primitive::<Gpu>().expect("checked convolution backend");
    let client = x.client.clone();
    let shape = burn_backend::Shape::new([rows, channels * kernel * kernel]);
    let output = TensorIr::uninit(client.create_empty_handle(), shape, x.dtype);
    let desc = CustomOpIr::with_scalars("unimsps_conv_pack", &[x.into_ir()], &[output], vec![]);
    let op = PackOp { desc: desc.clone(), height, width, kernel, stride, pad, start, rows };
    let [out] = client.register(StreamId::current(), OperationIr::Custom(desc), op).outputs();
    BurnTensor::from_primitive::<Gpu>(out)
}
