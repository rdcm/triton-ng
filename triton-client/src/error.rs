#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid URL: {0}")]
    InvalidUrl(String),

    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    #[error("decode error: {0}")]
    Decode(String),
}

pub type Result<T> = std::result::Result<T, Error>;
