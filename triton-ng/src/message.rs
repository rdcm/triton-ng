use crate::TritonError;
use std::ffi::c_char;
use triton_ng_macros::triton_call;

/// A JSON message from Triton (model metadata, config).
///
/// Owns the `TRITONSERVER_Message` and deletes it on drop. `Send + Sync`.
pub(crate) struct Message {
    ptr: *mut triton_ng_sys::TRITONSERVER_Message,
}

// SAFETY: a TRITONSERVER_Message is an immutable serialized JSON document
// owned by this value; serializing only reads it.
unsafe impl Send for Message {}
unsafe impl Sync for Message {}

impl Message {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONSERVER_Message) -> Self {
        Self { ptr }
    }

    pub(crate) fn to_json(&self) -> Result<String, TritonError> {
        let mut base: *const c_char = std::ptr::null();
        let mut byte_size: usize = 0;
        triton_call!(triton_ng_sys::TRITONSERVER_MessageSerializeToJson(
            self.ptr,
            &mut base,
            &mut byte_size,
        ))?;

        if byte_size == 0 {
            return Ok(String::new());
        }
        if base.is_null() {
            return Err(TritonError::from_message(
                "message serialized to a null buffer",
            ));
        }

        let bytes = unsafe { std::slice::from_raw_parts(base as *const u8, byte_size) };
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

impl Drop for Message {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { triton_ng_sys::TRITONSERVER_MessageDelete(self.ptr) };
        }
    }
}
