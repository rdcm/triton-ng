use crate::TritonError;
use crate::request::Request;
use crate::request_release::RequestRelease;
use std::sync::Arc;
use triton_ng_macros::triton_call;

/// The trace of a BLS request, spawned from the trace of the [`Request`] it
/// is made for, so that Triton reports it as a child of that request.
///
/// Owns the `TRITONSERVER_InferenceTrace` until it is handed to
/// `TRITONSERVER_ServerInferAsync`, see [`ChildTrace::into_raw`], and deletes
/// it on drop before that (the pointer is null after the handover). A spawned trace has reported nothing yet, so
/// deleting it is all its release takes (`InferenceTrace` has no destructor
/// logic in core `r25.09`, `TRITONSERVER_InferenceTraceDelete` is a plain
/// `delete`; the trace data of the frontend stays with the parent).
///
/// Holds the parent request unreleased: the frontend frees the trace data
/// shared by a trace and its children with the parent, see [`RequestRelease`].
/// `Send`, not `Sync`.
pub(crate) struct ChildTrace {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceTrace,
    parent: Arc<RequestRelease>,
}

// SAFETY: a spawned trace is used by nobody but its owner until it is handed
// to Triton; deleting it does not depend on the thread. `ChildTrace` is not
// `Sync`, the parent guard is `Send + Sync`.
unsafe impl Send for ChildTrace {}

impl ChildTrace {
    /// Spawns a child of the trace of `request`, `None` when the request is
    /// not traced.
    pub(crate) fn spawn(request: &Request) -> Result<Option<Self>, TritonError> {
        let parent = request.trace()?;
        if parent.is_null() {
            return Ok(None);
        }
        let mut ptr = std::ptr::null_mut();
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceTraceSpawnChildTrace(
            parent, &mut ptr
        ))?;
        if ptr.is_null() {
            return Err(TritonError::from_message(
                "Triton spawned a null child trace",
            ));
        }
        Ok(Some(Self {
            ptr,
            parent: request.release_guard(),
        }))
    }

    /// The guard of the parent request, see [`RequestRelease`].
    pub(crate) fn parent(&self) -> Arc<RequestRelease> {
        Arc::clone(&self.parent)
    }

    /// Hands the trace over for `TRITONSERVER_ServerInferAsync` and returns
    /// the guard of its parent request, which must outlive everything that
    /// may still report into the trace: the BLS request and its responses.
    ///
    /// Triton owns the trace once the submission succeeds. When it fails,
    /// Triton has either released the trace already or never taken it, and
    /// the error does not tell which, so the trace is leaked rather than
    /// deleted twice.
    pub(crate) fn into_raw(
        mut self,
    ) -> (
        *mut triton_ng_sys::TRITONSERVER_InferenceTrace,
        Arc<RequestRelease>,
    ) {
        let ptr = std::mem::replace(&mut self.ptr, std::ptr::null_mut());
        (ptr, Arc::clone(&self.parent))
    }
}

impl Drop for ChildTrace {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Err(e) = triton_call!(triton_ng_sys::TRITONSERVER_InferenceTraceDelete(self.ptr)) {
            crate::log::error(&format!("failed to delete child trace: {e}"));
        }
    }
}
