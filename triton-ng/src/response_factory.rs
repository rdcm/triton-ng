use crate::error::TritonError;
use crate::request::Request;
use triton_ng_macros::triton_call;

/// Response factory of a request, used to send flags without a response.
///
/// `TRITONBACKEND_ResponseFactoryNew` allocates a new `shared_ptr` to the
/// factory of the request and `TRITONBACKEND_ResponseFactoryDelete` deletes
/// only that `shared_ptr` (core r25.09, backend_model.cc), so the factory
/// stays valid while this value lives, even after the request is released.
///
/// Owns the `TRITONBACKEND_ResponseFactory` and deletes it on drop. Neither
/// `Send` nor `Sync`: it lives only inside [`crate::ResponseSender::finish`].
pub(crate) struct ResponseFactory {
    ptr: *mut triton_ng_sys::TRITONBACKEND_ResponseFactory,
}

impl ResponseFactory {
    pub(crate) fn new(request: &Request) -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseFactoryNew(
            &mut _,
            request.as_ptr()
        ))?;
        if ptr.is_null() {
            return Err(TritonError::from_message("response factory is null"));
        }
        Ok(Self { ptr })
    }

    /// Sends `flags` without a response: the response callback of the request
    /// is called with a null response.
    pub(crate) fn send_flags(&self, flags: u32) -> Result<(), TritonError> {
        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseFactorySendFlags(
            self.ptr, flags
        ))
    }
}

impl Drop for ResponseFactory {
    fn drop(&mut self) {
        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseFactoryDelete(self.ptr)) {
            crate::log::error(&format!("failed to delete response factory: {e}"));
        }
    }
}
