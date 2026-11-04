use crate::{ErrorCode, TritonError};

/// Why an inference failed, kept as plain data until the future resolves to
/// a [`TritonError`].
#[derive(Debug, PartialEq)]
pub(crate) struct Failure {
    pub(crate) code: ErrorCode,
    pub(crate) message: String,
}

impl From<TritonError> for Failure {
    fn from(error: TritonError) -> Self {
        Self {
            code: error.code(),
            message: error.to_string(),
        }
    }
}

impl From<Failure> for TritonError {
    fn from(failure: Failure) -> Self {
        TritonError::with_code(failure.code, failure.message)
    }
}
