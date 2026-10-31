use crate::utils::cstr_to_string;

pub type Error = Box<dyn std::error::Error>;

#[derive(Debug)]
pub struct TritonError {
    ptr: *mut triton_ng_sys::TRITONSERVER_Error,
    message: String,
}

impl TritonError {
    pub(crate) unsafe fn new(ptr: *mut triton_ng_sys::TRITONSERVER_Error) -> Self {
        let message = if ptr.is_null() {
            "Unknown Triton error".to_string()
        } else {
            // SAFETY: ptr not null
            unsafe {
                let msg_ptr = triton_ng_sys::TRITONSERVER_ErrorMessage(ptr);
                if msg_ptr.is_null() {
                    "Unknown Triton error".to_string()
                } else {
                    cstr_to_string(msg_ptr)
                }
            }
        };

        TritonError { ptr, message }
    }

    pub fn from_message(message: impl Into<String>) -> Self {
        TritonError {
            ptr: std::ptr::null_mut(),
            message: message.into(),
        }
    }

    /// Creates a Triton-side INTERNAL error object, deleted on drop.
    pub(crate) fn internal(message: &str) -> Self {
        Self::with_code(
            triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
            message,
        )
    }

    /// Creates a Triton-side CANCELLED error object, deleted on drop.
    pub(crate) fn cancelled(message: &str) -> Self {
        Self::with_code(
            triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_CANCELLED,
            message,
        )
    }

    fn with_code(code: triton_ng_sys::TRITONSERVER_Error_Code, message: &str) -> Self {
        let message = message.replace('\0', "?");
        let Ok(c_message) = std::ffi::CString::new(message.as_str()) else {
            return Self::from_message(message);
        };
        let ptr = unsafe { triton_ng_sys::TRITONSERVER_ErrorNew(code, c_message.as_ptr()) };
        TritonError { ptr, message }
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
