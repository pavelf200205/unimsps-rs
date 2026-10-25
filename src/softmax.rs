//! Row softmax for GEMM attention. Operates directly on 4D `[B, H, Q, K]` score tensors
//! without intermediate Reshape ops or clones, guaranteeing in-place buffer reuse.
use burn::tensor::{Tensor as BurnTensor, activation};
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
fn row_softmax<F: Float>(
    x: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] columns: usize,
    #[comptime] slots: usize,
    #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let heads = x.shape(1);
    let queries = x.shape(2);
    let total_rows = x.shape(0) * heads * queries;
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < total_rows;
        let query = row % queries;
        let head = (row / queries) % heads;
        let batch = row / (queries * heads);
        let row_base = batch * x.stride(0) + head * x.stride(1) + query * x.stride(2);
        let col_stride = x.stride(3);
        let mut values = Array::<f32>::new(slots);
        let mut maximum = f32::new(-3.402823e38_f32);
        #[unroll]
        for j in 0..slots {
            let column = lane + j * plane;
            values[j] = -3.402823e38_f32;
            if valid && column < columns {
                values[j] = f32::cast_from(x[row_base + column * col_stride]);
            }
            maximum = f32::max(maximum, values[j]);
        }
        maximum = plane_max(maximum);
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for j in 0..slots {
            let column = lane + j * plane;
            let mut value = f32::new(0.0_f32);
            if valid && column < columns {
                value = f32::exp(values[j] - maximum);
            }
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
struct SoftmaxOp {
    desc: CustomOpIr,
}

impl Operation<FusionCubeRuntime> for SoftmaxOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle>) -> Result<(), ExecutionError> {
        let ([input], [output]) = self.desc.as_fixed();
        let x = handles.get_float_tensor::<CubeBackend>(input);
        let [batch, heads, queries, columns] = x.meta.shape().dims::<4>();
        let client = x.client.clone();
        let props = client.properties();
        let hw = &props.hardware;
        let slots = columns.div_ceil(hw.plane_size_min as usize);
        let rows_per_group = 128 / hw.plane_size_min as usize;
        let groups = (batch * heads * queries).div_ceil(rows_per_group);
        let gy = groups.div_ceil(hw.max_cube_count.0 as usize).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic("Softmax dispatch exceeds device grid limits"));
        }
        // Read the whole row into registers before writing it. Reuse the score
        // buffer directly when exclusively owned and contiguous.
        let reuse = x.can_mut()
            && **x.meta.strides() == [heads * queries * columns, queries * columns, columns, 1];
        let result = if reuse {
            x.clone()
        } else {
            empty_device_contiguous_dtype(
                client.clone(),
                x.device.clone(),
                x.meta.shape().clone(),
                x.dtype,
            )
        };
        crate::profile::event(|| format!("softmax/subgroup columns={columns} in_place={reuse}"));
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        match x.dtype {
            DType::F16 => row_softmax::launch::<half::f16>(
                &client,
                count,
                dim,
                x.into_tensor_arg(),
                result.clone().into_tensor_arg(),
                columns,
                slots,
                rows_per_group,
            ),
            DType::F32 => row_softmax::launch::<f32>(
                &client,
                count,
                dim,
                x.into_tensor_arg(),
                result.clone().into_tensor_arg(),
                columns,
                slots,
                rows_per_group,
            ),
            _ => return Err(ExecutionError::generic("Unsupported softmax dtype")),
        }
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}

pub(crate) fn forward(x: BurnTensor<4>) -> BurnTensor<4> {
    let [_, _, _, columns] = x.dims();
    if columns == 0 || columns > 1024 {
        return activation::softmax(x, 3);
    }
    let Ok(prim) = x.inner().try_into_primitive::<Gpu>() else {
        unreachable!("CubeBackend tensor expected");
    };
    let client = prim.client.clone();
    let compute_client = client.device().client();
    let props = compute_client.properties();
    let hw = &props.hardware;
    let supported = props.features.plane.contains(cubecl::ir::features::Plane::Ops)
        && hw.plane_size_min >= 2
        && hw.plane_size_max <= 128
        && hw.max_units_per_cube >= 128
        && hw.max_cube_dim.0 >= 128
        && columns.div_ceil(hw.plane_size_min.max(1) as usize) <= 64;
    if !supported {
        return activation::softmax(BurnTensor::from_primitive::<Gpu>(prim), 3);
    }
    let output = TensorIr::uninit(client.create_empty_handle(), prim.shape.clone(), prim.dtype);
    let desc = CustomOpIr::with_scalars("unimsps_row_softmax", &[prim.into_ir()], &[output], vec![]);
    let [out] = client
        .register(StreamId::current(), OperationIr::Custom(desc.clone()), SoftmaxOp { desc })
        .outputs();
    BurnTensor::<4>::from_primitive::<Gpu>(out)
}
