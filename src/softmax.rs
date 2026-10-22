//! Row softmax for GEMM attention. Keep GEMMs native; fuse only the reduction.
use burn::tensor::{Tensor as BurnTensor, activation};
use burn_backend::{DType, ExecutionError};
use burn_cubecl::{CubeBackend, fusion::{CubeFusionHandle, FusionCubeRuntime}, ops::numeric::empty_device_contiguous_dtype};
use burn_fusion::{Fusion, custom::{CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId, TensorIr}};
use cubecl::prelude::*;

type Gpu = Fusion<CubeBackend>;

#[cube(launch)]
fn row_softmax<F: Float>(
    x: &Tensor<F>, out: &mut Tensor<F>,
    #[comptime] columns: usize, #[comptime] slots: usize,
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
        let query = row % x.shape(1);
        let mut values = Array::<f32>::new(slots);
        let mut maximum = f32::new(-3.402823e38_f32);
        #[unroll]
        for j in 0..slots {
            let column = lane + j * plane;
            values[j] = -3.402823e38_f32;
            if valid && column < columns {
                values[j] = f32::cast_from(x[batch * x.stride(0) + query * x.stride(1) + column * x.stride(2)]);
            }
            maximum = f32::max(maximum, values[j]);
        }
        maximum = plane_max(maximum);
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for j in 0..slots {
            let column = lane + j * plane;
            let mut value = f32::new(0.0_f32);
            if valid && column < columns { value = f32::exp(values[j] - maximum); }
            values[j] = value;
            sum += value;
        }
        let denominator = plane_sum(sum);
        #[unroll]
        for j in 0..slots {
            let column = lane + j * plane;
            if valid && column < columns {
                out[row * columns + column] = F::cast_from(values[j] / denominator);
            }
        }
    }
}

#[derive(Debug)]
struct SoftmaxOp { desc: CustomOpIr }
impl Operation<FusionCubeRuntime> for SoftmaxOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([input], [output]) = self.desc.as_fixed();
        let x = handles.get_float_tensor::<CubeBackend>(input);
        let [batch, queries, columns] = x.meta.shape().dims::<3>();
        let client = x.client.clone();
        let props = client.properties();
        let hw = &props.hardware;
        let slots = columns.div_ceil(hw.plane_size_min as usize);
        let rows_per_group = 128 / hw.plane_size_min as usize;
        let groups = (batch * queries).div_ceil(rows_per_group);
        let gy = groups.div_ceil(hw.max_cube_count.0 as usize).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic("Softmax dispatch exceeds device grid limits"));
        }
        // Read the whole row into registers before writing it. Reuse the score
        // buffer only if no other consumer owns it and its layout is contiguous.
        let reuse = x.can_mut() && **x.meta.strides() == [queries * columns, columns, 1];
        let result = if reuse { x.clone() } else {
            empty_device_contiguous_dtype(client.clone(), x.device.clone(), x.meta.shape().clone(), x.dtype)
        };
        crate::profile::event(|| format!("softmax/subgroup columns={columns} in_place={reuse}"));
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        match x.dtype {
            DType::F16 => row_softmax::launch::<half::f16>(&client, count, dim,
                x.into_tensor_arg(), result.clone().into_tensor_arg(), columns, slots, rows_per_group),
            DType::F32 => row_softmax::launch::<f32>(&client, count, dim,
                x.into_tensor_arg(), result.clone().into_tensor_arg(), columns, slots, rows_per_group),
            _ => return Err(ExecutionError::generic("Unsupported softmax dtype")),
        }
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}

pub(crate) fn forward(x: BurnTensor<4>) -> BurnTensor<4> {
    let shape = x.dims();
    let [batch, heads, queries, columns] = shape;
    let Ok(probe) = x.clone().inner().try_into_primitive::<Gpu>() else { return activation::softmax(x, 3); };
    let compute_client = probe.client.device().client();
    let props = compute_client.properties();
    let hw = &props.hardware;
    let supported = props.features.plane.contains(cubecl::ir::features::Plane::Ops)
        && hw.plane_size_min >= 2 && hw.plane_size_max <= 128
        && hw.max_units_per_cube >= 128 && hw.max_cube_dim.0 >= 128
        && columns > 0 && columns <= 1024
        && columns.div_ceil(hw.plane_size_min.max(1) as usize) <= 64;
    drop(probe);
    if !supported { return activation::softmax(x, 3); }
    let x = x.reshape([batch * heads, queries, columns]).inner().try_into_primitive::<Gpu>().unwrap();
    let client = x.client.clone();
    let output = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
    let desc = CustomOpIr::with_scalars("unimsps_row_softmax", &[x.into_ir()], &[output], vec![]);
    let [out] = client.register(StreamId::current(), OperationIr::Custom(desc.clone()), SoftmaxOp { desc }).outputs();
    BurnTensor::<3>::from_primitive::<Gpu>(out).reshape(shape)
}
