/// Everything a call of the client can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The URL of [`crate::TritonClientConfig`] cannot be parsed.
    #[error("invalid URL: {0}")]
    InvalidUrl(String),

    /// The connection to the server failed.
    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    /// The server answered the call with a gRPC status other than OK.
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    /// The response cannot be decoded: an unknown datatype, a tensor whose bytes do not fit its shape and type.
    #[error("decode error: {0}")]
    Decode(String),

    /// The server reports an inference failure in the response of a stream.
    #[error("inference error: {0}")]
    Inference(String),
}

/// `Result` with the [`Error`] of the client.
pub type Result<T> = std::result::Result<T, Error>;
