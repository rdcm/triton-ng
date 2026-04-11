#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("config error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("client error: {0}")]
    Client(#[from] triton_client::Error),

    #[error("model has no inputs")]
    ModelHasNoInputs,

    #[error("model has no outputs")]
    ModelHasNoOutputs,
}

pub type Result<T> = std::result::Result<T, Error>;
