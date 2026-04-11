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
