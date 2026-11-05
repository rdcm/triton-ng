//! Async gRPC client of [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server):
//! inference (unary and streaming), model and server metadata, statistics, model repository
//! control, shared memory regions, logging and trace settings, over the `GRPCInferenceService`
//! of Triton. See the [repository README](https://github.com/rdcm/triton-ng#using-the-grpc-client)
//! for examples.
#![deny(missing_docs)]

mod client;
mod datatype;
mod error;
#[rustfmt::skip]
mod generated;
mod infer;
mod types;

pub use client::{InferStream, TritonClient, TritonClientConfig};
pub use datatype::Datatype;
pub use error::{Error, Result};
pub use infer::{
    InferInput, InferOptions, InferOutput, InferResponse, InputData, OutputData, Parameter,
};
pub use tonic::transport::{Certificate, ClientTlsConfig, Identity};
pub use types::{
    CudaMemoryRegion, LogSettingValue, MemoryUsage, ModelConfig, ModelIndex, ModelInfo,
    ModelStatistics, ServerInfo, SharedMemoryRegion, TensorMetadata,
};
