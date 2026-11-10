use crate::batch_statistics::BatchStatistics;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use triton_ng_macros::triton_call;

/// Releases a `TRITONBACKEND_Request` to Triton when the last owner lets it
/// go. A [`crate::Request`] holds one, and so does every BLS request traced
/// as its child: the trace of a request is released with the request, and a
/// child trace that outlives it reports into freed memory of the frontend
/// (the frontend frees the trace data it shares with the children as soon as
/// no tracked trace is left).
///
/// A request is released for good, or, after
/// [`RequestRelease::reschedule`], handed back to Triton's iterative sequence
/// batcher, which enqueues it again. The flag is kept here because the
/// release may happen after the `Request` is gone.
///
/// Before the release, the request is reported to Triton's statistics as
/// done, successfully when its final response went out without an error.
///
/// Owns the request: releases it on drop. `Send + Sync`.
pub(crate) struct RequestRelease {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
    reschedule: AtomicBool,
    succeeded: AtomicBool,
    statistics: Arc<BatchStatistics>,
}

// SAFETY: the pointer is only used once, in `Drop`, by the last owner, and
// Triton lets a request be released from any thread (tritonbackend.h,
// TRITONBACKEND_RequestRelease).
unsafe impl Send for RequestRelease {}
unsafe impl Sync for RequestRelease {}

impl RequestRelease {
    pub(crate) fn new(
        ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
        statistics: Arc<BatchStatistics>,
    ) -> Self {
        Self {
            ptr,
            reschedule: AtomicBool::new(false),
            succeeded: AtomicBool::new(false),
            statistics,
        }
    }

    /// Records that the final response of the request went out without an
    /// error, for the statistics.
    pub(crate) fn succeeded(&self) {
        self.succeeded.store(true, Ordering::Release);
    }

    /// The statistics of the batch the request belongs to.
    pub(crate) fn statistics(&self) -> &BatchStatistics {
        &self.statistics
    }

    /// Releases the request with `TRITONSERVER_REQUEST_RELEASE_RESCHEDULE`
    /// instead. Only for models with `iterative_sequence`: the core ignores
    /// the flag otherwise and the request never gets a final response, see
    /// `Request::reschedule`.
    pub(crate) fn reschedule(&self) {
        self.reschedule.store(true, Ordering::Release);
    }
}

impl Drop for RequestRelease {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let reschedule = self.reschedule.load(Ordering::Acquire);
        // A rescheduled request is not done, it comes back for another step.
        if !reschedule
            && let Err(e) = self
                .statistics
                .report_request(self.ptr, self.succeeded.load(Ordering::Acquire))
        {
            crate::log::error(&format!("failed to report request statistics: {e}"));
        }
        let flags = if reschedule {
            triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_RESCHEDULE
        } else {
            triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL
        };
        // On error the request stays with the backend, nothing else can be done with it.
        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_RequestRelease(self.ptr, flags,))
        {
            crate::log::error(&format!("failed to release request: {e}"));
        }
    }
}
