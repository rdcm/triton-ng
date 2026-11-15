#[allow(clippy::enum_variant_names)]
#[allow(
    clippy::disallowed_macros,
    reason = "prost::Oneof generates panic and unreachable branches; this module is generated from protobuf"
)]
#[path = "inference.rs"]
pub mod inference;

#[path = "grpc.health.v1.rs"]
pub mod healthcheck;
