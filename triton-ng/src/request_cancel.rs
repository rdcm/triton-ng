use crate::error::TritonError;
use std::sync::{Arc, Mutex, PoisonError};
use triton_ng_macros::triton_call;

/// The right to cancel a submitted BLS request, shared by its
/// [`crate::InferenceFuture`] or [`crate::InferenceStream`] and its release
/// callback.
///
/// `TRITONSERVER_InferenceRequestCancel` needs a live request, and the
/// release callback deletes it, so the callback calls
/// [`RequestCancel::released`] first: from then on cancelling does nothing.
/// Both run under one mutex, so the request is never deleted during a
/// cancellation.
///
/// Borrows the `TRITONSERVER_InferenceRequest`, whose deletion is up to the
/// release callback. `Send + Sync`.
pub(crate) struct RequestCancel {
    /// Null once the request is released.
    request: Mutex<*mut triton_ng_sys::TRITONSERVER_InferenceRequest>,
}

// SAFETY: the pointer is only used under the mutex, and the release callback
// nulls it under the same mutex before it deletes the request. Triton lets a
// request be cancelled from any thread (InferenceRequest::Cancel only sets a
// flag of its response factory).
unsafe impl Send for RequestCancel {}
unsafe impl Sync for RequestCancel {}

impl RequestCancel {
    pub(crate) fn new(request: *mut triton_ng_sys::TRITONSERVER_InferenceRequest) -> Arc<Self> {
        Arc::new(Self {
            request: Mutex::new(request),
        })
    }

    /// Asks Triton to cancel the request. Does nothing once it is released.
    pub(crate) fn cancel(&self) -> Result<(), TritonError> {
        self.with_live(|request| {
            triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestCancel(request))
        })
    }

    /// Runs `f` on the request under the mutex, so that it is not released
    /// meanwhile, unless it is released already.
    fn with_live<E>(
        &self,
        f: impl FnOnce(*mut triton_ng_sys::TRITONSERVER_InferenceRequest) -> Result<(), E>,
    ) -> Result<(), E> {
        let request = self.request.lock().unwrap_or_else(PoisonError::into_inner);
        if request.is_null() {
            return Ok(());
        }
        f(*request)
    }

    /// Called by the release callback before it deletes the request.
    pub(crate) fn released(&self) {
        *self.request.lock().unwrap_or_else(PoisonError::into_inner) = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::RequestCancel;

    fn live_calls(cancel: &RequestCancel) -> usize {
        let mut calls = 0;
        cancel
            .with_live(|_| {
                calls += 1;
                Ok::<(), ()>(())
            })
            .unwrap();
        calls
    }

    #[test]
    fn request_is_used_until_it_is_released() {
        let mut request = 0u8;
        let cancel = RequestCancel::new((&raw mut request).cast());

        assert_eq!(live_calls(&cancel), 1);
        cancel.released();
        assert_eq!(live_calls(&cancel), 0);
    }
}
