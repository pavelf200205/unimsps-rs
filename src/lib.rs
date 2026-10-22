//! Standalone Uni-MS-PS inference. Tensor execution uses Burn 0.22.
pub mod model;
pub mod pipeline;
mod attention;
mod light_attention;
pub mod profile;
pub mod execution;
