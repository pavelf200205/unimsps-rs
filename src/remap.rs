//! Parallel row remapping for token-layout gather and inverse scatter.
//! One output element per lane: no atomics, serial scatter, or padding writes.
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
fn remap<F: Float, I: Int>(x: &Tensor<F>, indices: &Tensor<I>, out: &mut Tensor<F>) {
    let index = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if index < out.len() {
        let channels = out.shape(2);
        let lights = out.shape(1);
        let pixel = index / (lights * channels);
        let light = (index / channels) % lights;
        let channel = index % channels;
        let source = i32::cast_from(indices[pixel * indices.stride(0)]);
        let mut value = F::new(0.0_f32);
        if source >= 0 {
            let row = usize::cast_from(source);
            if row < x.shape(0) {
                value = x[row * x.stride(0) + light * x.stride(1) + channel * x.stride(2)];
            }
        }
        out[index] = value;
    }
}
#[derive(Debug)]
struct RemapOp { desc: CustomOpIr }
impl Operation<FusionCubeRuntime> for RemapOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([input, map], [output]) = self.desc.as_fixed();
        let x = handles.get_float_tensor::<CubeBackend>(input);
        let map = handles.get_int_tensor::<CubeBackend>(map);
        let [_, lights, channels] = x.meta.shape().dims::<3>();
        let rows = map.meta.shape().dims::<1>()[0];
        let client = x.client.clone();
        let hw = &client.properties().hardware;
        let groups = (rows * lights * channels).div_ceil(128);
        let gy = groups.div_ceil((hw.max_cube_count.0 as usize).max(1)).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize { return Err(ExecutionError::generic("Row remap exceeds device grid limits")); }
        let result = empty_device_contiguous_dtype(client.clone(),x.device.clone(),burn_backend::Shape::new([rows,lights,channels]),x.dtype);
        let count = CubeCount::Static(gx as u32,gy as u32,1);
        let dim = CubeDim::new_1d(128);
        match (x.dtype,map.dtype) {
            (DType::F16,DType::I32) => remap::launch::<half::f16,i32>(&client,count,dim,x.into_tensor_arg(),map.into_tensor_arg(),result.clone().into_tensor_arg()),
            (DType::F32,DType::I32) => remap::launch::<f32,i32>(&client,count,dim,x.into_tensor_arg(),map.into_tensor_arg(),result.clone().into_tensor_arg()),
            (DType::F16,DType::I64) => remap::launch::<half::f16,i64>(&client,count,dim,x.into_tensor_arg(),map.into_tensor_arg(),result.clone().into_tensor_arg()),
            (DType::F32,DType::I64) => remap::launch::<f32,i64>(&client,count,dim,x.into_tensor_arg(),map.into_tensor_arg(),result.clone().into_tensor_arg()),
            _ => return Err(ExecutionError::generic("Unsupported remap dtype")),
        }
        handles.register_float_tensor::<CubeBackend>(&output.id,result);
        Ok(())
    }
}
pub(crate) fn forward(x: BurnTensor<3>, map: BurnTensor<1,burn::tensor::Int>) -> Option<BurnTensor<3>> {
    let [_,lights,channels] = x.dims();
    let rows = map.dims()[0];
    let x = x.inner().try_into_primitive::<Gpu>().ok()?;
    let map = map.inner().try_into_primitive::<Gpu>().ok()?;
    if !matches!(x.dtype,DType::F16|DType::F32) || !matches!(map.dtype,DType::I32|DType::I64) { return None; }
    let client = x.client.clone();
    let compute = client.device().client();
    let hw = &compute.properties().hardware;
    if hw.max_units_per_cube < 128 || hw.max_cube_dim.0 < 128 { return None; }
    let out = TensorIr::uninit(client.create_empty_handle(),burn_backend::Shape::new([rows,lights,channels]),x.dtype);
    let desc = CustomOpIr::with_scalars("unimsps_row_remap",&[x.into_ir(),map.into_ir()],&[out],vec![]);
    let [out] = client.register(StreamId::current(),OperationIr::Custom(desc.clone()),RemapOp { desc }).outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}
