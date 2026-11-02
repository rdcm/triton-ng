use crate::utils::cstr_to_string;

/// The error of every fallible call in triton-ng and of the backend hooks:
/// any error type that is `Send + Sync`, boxed. Being `Send + Sync`, it can
/// be held across `.await` in `model_instance_execute`, whose future must be
/// `Send`. `?` converts `String`, `&str`, [`TritonError`] and most library
/// errors into it.
///
/// # Examples
///
/// ```ignore
/// fn model_initialize(model: &Model) -> Result<(), Error> {
///     let vocab = String::from_utf8(model.load_file("vocab.txt")?)?;
///     if vocab.is_empty() {
///         return Err("vocab.txt is empty".into());
///     }
///     model.set_state(State { vocab })?;
///     Ok(())
/// }
/// ```
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// What kind of failure a [`TritonError`] is, the `TRITONSERVER_Error_Code`
/// Triton reports to clients. The gRPC frontend turns it into the status code
/// of the call (`InvalidArg` into `INVALID_ARGUMENT`, `Unsupported` into
/// `UNIMPLEMENTED`, the others into the status of the same name), so clients
/// can retry `Unavailable` and give up on `InvalidArg`.
///
/// # Examples
///
/// ```ignore
/// match ctx.server().infer_async(inference).await {
///     Ok(result) => answer(request, result)?,
///     Err(e) if e.code() == ErrorCode::Unavailable => request.respond_error_with_code(e.code(), "encoder is overloaded")?,
///     Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    Unknown,
    Internal,
    NotFound,
    InvalidArg,
    Unavailable,
    Unsupported,
    AlreadyExists,
    Cancelled,
}

impl ErrorCode {
    pub(crate) fn from_sys(code: triton_ng_sys::TRITONSERVER_Error_Code) -> Self {
        match code {
            triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL => {
                Self::Internal
            }
            triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_NOT_FOUND => {
                Self::NotFound
            }
            triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INVALID_ARG => {
                Self::InvalidArg
            }
            triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNAVAILABLE => {
                Self::Unavailable
            }
            triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNSUPPORTED => {
                Self::Unsupported
            }
            triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_ALREADY_EXISTS => {
                Self::AlreadyExists
            }
            triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_CANCELLED => {
                Self::Cancelled
            }
            _ => Self::Unknown,
        }
    }

    pub(crate) fn to_sys(self) -> triton_ng_sys::TRITONSERVER_Error_Code {
        match self {
            Self::Unknown => triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNKNOWN,
            Self::Internal => {
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL
            }
            Self::NotFound => {
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_NOT_FOUND
            }
            Self::InvalidArg => {
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INVALID_ARG
            }
            Self::Unavailable => {
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNAVAILABLE
            }
            Self::Unsupported => {
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_UNSUPPORTED
            }
            Self::AlreadyExists => {
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_ALREADY_EXISTS
            }
            Self::Cancelled => {
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_CANCELLED
            }
        }
    }
}

/// An error from Triton or from triton-ng: its [`ErrorCode`] and message.
/// Errors of a BLS call keep the code the called model or Triton gave them.
///
/// Owns the `TRITONSERVER_Error` it may carry and deletes it on drop. `Send + Sync`.
#[derive(Debug)]
pub struct TritonError {
    ptr: *mut triton_ng_sys::TRITONSERVER_Error,
    code: ErrorCode,
    message: String,
}

// SAFETY: a TRITONSERVER_Error is an immutable code and message owned by this
// value; reading it and deleting it are not tied to the thread that made it.
unsafe impl Send for TritonError {}
unsafe impl Sync for TritonError {}

impl TritonError {
    pub(crate) unsafe fn new(ptr: *mut triton_ng_sys::TRITONSERVER_Error) -> Self {
        if ptr.is_null() {
            return TritonError {
                ptr,
                code: ErrorCode::Unknown,
                message: "Unknown Triton error".to_string(),
            };
        }
        // SAFETY: ptr not null
        let (code, msg_ptr) = unsafe {
            (
                triton_ng_sys::TRITONSERVER_ErrorCode(ptr),
                triton_ng_sys::TRITONSERVER_ErrorMessage(ptr),
            )
        };
        let message = if msg_ptr.is_null() {
            "Unknown Triton error".to_string()
        } else {
            unsafe { cstr_to_string(msg_ptr) }
        };

        TritonError {
            ptr,
            code: ErrorCode::from_sys(code),
            message,
        }
    }

    /// An [`ErrorCode::Internal`] error with `message`.
    pub fn from_message(message: impl Into<String>) -> Self {
        Self::with_code(ErrorCode::Internal, message)
    }

    /// An error with `code` and `message`. Returned from a backend hook, it
    /// reaches Triton with its code.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_initialize(model: &Model) -> Result<(), Error> {
    ///     let vocab = model
    ///         .config_parameter("vocab")?
    ///         .ok_or_else(|| TritonError::with_code(ErrorCode::InvalidArg, "missing 'vocab' config parameter"))?;
    ///     // ...
    ///     Ok(())
    /// }
    /// ```
    pub fn with_code(code: ErrorCode, message: impl Into<String>) -> Self {
        TritonError {
            ptr: std::ptr::null_mut(),
            code,
            message: message.into(),
        }
    }

    /// What kind of failure this is.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Err(e) = ctx.server().infer_async(inference).await {
    ///     request.respond_error_with_code(e.code(), &e.to_string())?;
    /// }
    /// ```
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// The code of `error`: its own when it is a [`TritonError`], `Internal`
    /// for any other error.
    pub(crate) fn code_of(error: &Error) -> ErrorCode {
        error
            .downcast_ref::<TritonError>()
            .map_or(ErrorCode::Internal, TritonError::code)
    }

    /// Creates a Triton-side error object with `code`, deleted on drop.
    pub(crate) fn in_triton(code: ErrorCode, message: &str) -> Self {
        let message = message.replace('\0', "?");
        let Ok(c_message) = std::ffi::CString::new(message.as_str()) else {
            return Self::with_code(code, message);
        };
        let ptr =
            unsafe { triton_ng_sys::TRITONSERVER_ErrorNew(code.to_sys(), c_message.as_ptr()) };
        TritonError { ptr, code, message }
    }

    /// The Triton error object, null when the error exists only on the Rust side.
    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONSERVER_Error {
        self.ptr
    }
}

impl std::fmt::Display for TritonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TritonError {}

impl Drop for TritonError {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONSERVER_ErrorDelete(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ErrorCode;

    const CODES: [ErrorCode; 8] = [
        ErrorCode::Unknown,
        ErrorCode::Internal,
        ErrorCode::NotFound,
        ErrorCode::InvalidArg,
        ErrorCode::Unavailable,
        ErrorCode::Unsupported,
        ErrorCode::AlreadyExists,
        ErrorCode::Cancelled,
    ];

    #[test]
    fn codes_round_trip_through_triton_values() {
        for code in CODES {
            assert_eq!(ErrorCode::from_sys(code.to_sys()), code);
        }
        assert_eq!(ErrorCode::from_sys(u32::MAX), ErrorCode::Unknown);
    }

    #[test]
    fn triton_values_are_distinct() {
        let mut values = CODES.map(ErrorCode::to_sys);
        values.sort();
        values
            .windows(2)
            .for_each(|pair| assert_ne!(pair[0], pair[1]));
    }
}
