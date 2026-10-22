//! Direct flash attention: uses BlackboxAccelerated flash when supported for the problem size,
//! and switches to tiled GEMM attention if cooperative matrix tiles are unavailable on the device.
use burn::tensor::Tensor;
use burn_backend::{ExecutionError, ops::AttentionModuleOptions};
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
    kernel::attention::flash_attention,
};
use burn_fusion::{
    Fusion,
    custom::{CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId, TensorIr},
};
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

fn problem_key(q: &Tensor<4>, k: &Tensor<4>) -> String {
    format!("{:?}/{:?}/{:?}/{:?}", q.device().identity(), q.dtype(), q.dims(), k.dims())
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
                    "Spatial attention: accelerated flash unavailable for this shape ({accelerated_error:?}); caching a shape-specific GEMM fallback"
                ));
                result
            }
        };
        handles.register_float_tensor::<CubeBackend>(&out.id, result);
        Ok(())
    }
}

pub(crate) fn supported(q: &Tensor<4>, k: &Tensor<4>) -> bool {
    let Ok(primitive) = q.clone().inner().try_into_primitive::<Gpu>() else { return false; };
    let client = primitive.client.device().client();
    if client.properties().features.matmul.cmma.is_empty() {
        static NOTICE: std::sync::Once = std::sync::Once::new();
        NOTICE.call_once(|| eprintln!("Spatial attention: tiled GEMM (device exposes no cooperative-matrix configurations)"));
        return false;
    }
    !unavailable().lock().unwrap().contains(&problem_key(q, k))
}

pub(crate) fn flash(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Option<Tensor<4>> {
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
    Some(Tensor::from_primitive::<Gpu>(out))
}
