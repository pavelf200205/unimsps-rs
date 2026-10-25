//! Standalone Uni-MS-PS inference. Tensor execution uses Burn 0.22.
pub mod model;
pub mod pipeline;
mod attention;
mod light_attention;
mod norm;
pub mod profile;
mod softmax;
pub mod execution;

mod depthwise;

mod conv_gemm;

pub mod cache;
mod remap;

pub mod trace;
