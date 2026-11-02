use triton_ng_macros::triton_call;

/// Releases a `TRITONBACKEND_Request` to Triton when the last owner lets it
/// go. A [`crate::Request`] holds one, and so does every BLS request traced
/// as its child: the trace of a request is released with the request, and a
/// child trace that outlives it reports into freed memory of the frontend
/// (the frontend frees the trace data it shares with the children as soon as
/// no tracked trace is left).
///
/// Owns the request: releases it on drop. `Send + Sync`.
pub(crate) struct RequestRelease {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
}

// SAFETY: the pointer is only used once, in `Drop`, by the last owner, and
// Triton lets a request be released from any thread (tritonbackend.h,
// TRITONBACKEND_RequestRelease).
unsafe impl Send for RequestRelease {}
unsafe impl Sync for RequestRelease {}

impl RequestRelease {
    pub(crate) fn new(ptr: *mut triton_ng_sys::TRITONBACKEND_Request) -> Self {
        Self { ptr }
    }
}

impl Drop for RequestRelease {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        // On error the request stays with the backend, nothing else can be done with it.
        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_RequestRelease(
            self.ptr,
            triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL,
        )) {
            crate::log::error(&format!("failed to release request: {e}"));
        }
    }
}
