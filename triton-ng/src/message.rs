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

    /// A new message holding a copy of `json` (core `r25.09` builds a
    /// `std::string` of it in `TRITONSERVER_MessageNewFromSerializedJson`).
    pub(crate) fn from_json(json: &str) -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONSERVER_MessageNewFromSerializedJson(
            &mut _,
            json.as_ptr().cast(),
            json.len(),
        ))?;
        if ptr.is_null() {
            return Err(TritonError::from_message(
                "TRITONSERVER_MessageNewFromSerializedJson returned null",
            ));
        }
        Ok(Self::from_ptr(ptr))
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONSERVER_Message {
        self.ptr
    }

    pub(crate) fn to_json(&self) -> Result<String, TritonError> {
        unsafe { serialize_to_json(self.ptr) }
    }
}

/// Serializes a `TRITONSERVER_Message` to JSON without taking it: used both
/// by [`Message`] and for messages Triton keeps, like the backend config.
///
/// # Safety
///
/// `ptr` must be a live `TRITONSERVER_Message` for the whole call.
pub(crate) unsafe fn serialize_to_json(
    ptr: *mut triton_ng_sys::TRITONSERVER_Message,
) -> Result<String, TritonError> {
    let mut base: *const c_char = std::ptr::null();
    let mut byte_size: usize = 0;
    triton_call!(triton_ng_sys::TRITONSERVER_MessageSerializeToJson(
        ptr,
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

impl Drop for Message {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { triton_ng_sys::TRITONSERVER_MessageDelete(self.ptr) };
        }
    }
}
