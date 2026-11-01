use crate::inference_future::Completion;
use crate::inference_request::InputBuffers;
use crate::inference_response::InferenceResponse;
use crate::message::Message;
use crate::response_allocator::{OutputPlan, ResponseAllocator};
use crate::utils::cstring_from_str;
use crate::{InferenceFuture, InferenceRequest, InferenceResult, OutputTensor, TritonError};
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use triton_ng_macros::triton_call;

/// What the response callback of one BLS request needs, handed to Triton as
/// `response_userp` and freed with the final response.
pub(crate) struct InferenceContext {
    completion: Arc<Completion>,
    allocator: ResponseAllocator,
    outputs: OutputPlan,
    pending_outputs: Vec<OutputTensor>,
    error: Option<String>,
}

/// The Triton server the backend runs in, used for BLS: inference of other
/// models from a backend. Batches get it from [`crate::Context::server`].
///
/// Borrows the `TRITONSERVER_Server`: Triton owns it, so there is nothing to
/// drop. The server stays valid while any model is loaded, and a model stays
/// loaded while any of its [`crate::Request`]s is alive. `Send + Sync + Clone`.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///     let state = ctx.state::<State>()?;
///     let mut inference = InferenceRequest::new(ctx.server(), &state.encoder, -1)?;
///     // ... add the inputs of all `requests` as one batch ...
///     let result = ctx.server().infer_async(inference).await?;
///     // ... answer every request from its rows of `result` ...
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct Server {
    ptr: *mut triton_ng_sys::TRITONSERVER_Server,
}

// SAFETY: the pointer is the TRITONSERVER_Server, which the frontends call
// from many threads at once. TRITONSERVER_ServerDelete waits for in-flight
// requests and for every model to unload before freeing it (InferenceServer::Stop),
// and a model is not unloaded while any of its requests is alive, so the
// pointer outlives every `Request` the backend holds. Only an expired
// --exit-timeout-secs on shutdown breaks this, when the process is exiting.
unsafe impl Send for Server {}
unsafe impl Sync for Server {}

impl Server {
    pub(crate) fn from_ptr(
        ptr: *mut triton_ng_sys::TRITONSERVER_Server,
    ) -> Result<Self, TritonError> {
        if ptr.is_null() {
            return Err(TritonError::from_message(
                "TRITONBACKEND_ModelServer returned null",
            ));
        }
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONSERVER_Server {
        self.ptr
    }

    pub fn api_version(&self) -> Result<(u32, u32), TritonError> {
        let mut major: u32 = 0;
        let mut minor: u32 = 0;

        triton_call!(triton_ng_sys::TRITONSERVER_ApiVersion(
            &mut major, &mut minor
        ))?;

        Ok((major, minor))
    }

    pub fn is_ready(&self) -> Result<bool, TritonError> {
        let mut ready = false;

        triton_call!(triton_ng_sys::TRITONSERVER_ServerIsReady(
            self.ptr, &mut ready
        ))?;

        Ok(ready)
    }

    pub fn is_model_ready(&self, model_name: &str, version: i64) -> Result<bool, TritonError> {
        let model_name_cstr = cstring_from_str(model_name)?;
        let mut ready = false;

        triton_call!(triton_ng_sys::TRITONSERVER_ServerModelIsReady(
            self.ptr,
            model_name_cstr.as_ptr(),
            version,
            &mut ready
        ))?;

        Ok(ready)
    }

    pub fn model_metadata(&self, model_name: &str, version: i64) -> Result<String, TritonError> {
        let model_name_cstr = cstring_from_str(model_name)?;
        let metadata = Message::from_ptr(triton_call!(
            triton_ng_sys::TRITONSERVER_ServerModelMetadata(
                self.ptr,
                model_name_cstr.as_ptr(),
                version,
                &mut _,
            )
        )?);

        metadata.to_json()
    }

    /// Submits inference and returns a future of its result, see
    /// [`InferenceFuture`]. Submission errors are reported by the future.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut inference = InferenceRequest::new(&server, "encoder", -1)?;
    /// inference.add_input_with_data("AUDIO", DataType::Fp32, &[1, samples.len() as i64], bytes)?;
    /// inference.add_requested_output("EMBEDDING")?;
    ///
    /// let result = server.infer_async(inference).await?;
    /// let embedding = result.output("EMBEDDING")?.as_fp32_vec()?;
    /// ```
    ///
    /// Takes ownership of `request`. The request is freed by the
    /// `inference_request_release` callback, which Triton calls only *after* the
    /// backend has fully released all references to it. This prevents the race
    /// where `InferenceRequest::Drop` frees the object while Triton's backend
    /// thread still holds a pointer (which causes a SIGSEGV in
    /// `TRITONBACKEND_RequestRelease`).
    ///
    /// The input buffers owned by the request are freed by the same callback:
    /// Triton may read them until the request is released, which can happen
    /// after the final response has already been delivered.
    pub fn infer_async(&self, request: InferenceRequest) -> InferenceFuture {
        let completion = Completion::new();
        match self.submit(request, Arc::clone(&completion)) {
            Ok(()) => InferenceFuture::new(completion),
            Err(e) => InferenceFuture::new(Completion::failed(e.to_string())),
        }
    }

    fn submit(
        &self,
        request: InferenceRequest,
        completion: Arc<Completion>,
    ) -> Result<(), TritonError> {
        let allocator = ResponseAllocator::new()?;

        let (raw, buffers, outputs) = request.into_raw_parts();
        let buffers_ptr = Box::into_raw(Box::new(buffers)) as *mut c_void;

        // Register the release callback before handing the request to Triton.
        if let Err(e) = triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetReleaseCallback(
                raw,
                Some(inference_request_release),
                buffers_ptr,
            )
        ) {
            // Triton never saw this request, release everything ourselves.
            unsafe { inference_request_release(raw, RELEASE_ALL, buffers_ptr) };
            return Err(e);
        }

        let context = Box::new(InferenceContext {
            completion,
            allocator,
            outputs: OutputPlan::new(outputs),
            pending_outputs: Vec::new(),
            error: None,
        });
        let context_ptr = Box::into_raw(context) as *mut c_void;

        if let Err(e) = triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetResponseCallback(
                raw,
                (*context_ptr.cast::<InferenceContext>()).allocator.as_ptr(),
                ptr::from_ref(&(*context_ptr.cast::<InferenceContext>()).outputs)
                    .cast_mut()
                    .cast(),
                Some(inference_response_complete),
                context_ptr,
            )
        ) {
            unsafe {
                drop(Box::from_raw(context_ptr as *mut InferenceContext));
                inference_request_release(raw, RELEASE_ALL, buffers_ptr);
            }
            return Err(e);
        }

        // Transfer ownership to Triton. The release callback now owns deletion.
        if let Err(e) = triton_call!(triton_ng_sys::TRITONSERVER_ServerInferAsync(
            self.ptr,
            raw,
            ptr::null_mut(),
        )) {
            // Triton guarantees it will NOT call neither the release nor the
            // response callback on failure, so we must clean up manually.
            unsafe {
                drop(Box::from_raw(context_ptr as *mut InferenceContext));
                inference_request_release(raw, RELEASE_ALL, buffers_ptr);
            }
            return Err(e);
        }

        Ok(())
    }
}

/// Called by Triton after the backend has fully released the inference request.
/// At this point it is safe to delete the request object.
///
/// `userp` is the boxed [`InputBuffers`] referenced by the request, they are
/// freed only after the request itself is deleted.
///
/// The header says RELEASE_ALL should always be set today, but recommends
/// checking explicitly in case future versions add new flags.
unsafe extern "C" fn inference_request_release(
    request: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
    flags: u32,
    userp: *mut std::os::raw::c_void,
) {
    if flags & RELEASE_ALL == 0 {
        return;
    }

    if !request.is_null() {
        unsafe { triton_ng_sys::TRITONSERVER_InferenceRequestDelete(request) };
    }

    if !userp.is_null() {
        unsafe { drop(Box::from_raw(userp as *mut InputBuffers)) };
    }
}

const RELEASE_ALL: u32 =
    triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL;

/// Called by Triton for each response (and once more with a null response when
/// RESPONSE_COMPLETE_FINAL is set, to signal end-of-stream).
///
/// For one-to-one models this is called exactly once with FINAL set.
/// For decoupled/streaming models it may be called multiple times; we
/// accumulate outputs across all non-final callbacks and only send the
/// collected result (and drop the context) when FINAL arrives.
unsafe extern "C" fn inference_response_complete(
    response_ptr: *mut triton_ng_sys::TRITONSERVER_InferenceResponse,
    flags: u32,
    userp: *mut std::os::raw::c_void,
) {
    use triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL as FINAL;
    let is_final = flags & FINAL != 0;

    if userp.is_null() {
        // No context to report to: still free the response.
        if !response_ptr.is_null() {
            drop(InferenceResponse::from_ptr(response_ptr));
        }
        return;
    }

    // Borrow context without taking ownership — we only own it on the final call.
    let context = unsafe { &mut *(userp as *mut InferenceContext) };

    if !response_ptr.is_null() {
        match InferenceResponse::from_ptr(response_ptr) {
            Err(e) => {
                context.error.get_or_insert_with(|| e.to_string());
            }
            Ok(response) => {
                if let Some(err) = response.error() {
                    context.error.get_or_insert_with(|| err.to_string());
                } else {
                    match response.outputs() {
                        Ok(outputs) => context.pending_outputs.extend(outputs),
                        Err(e) => {
                            context.error.get_or_insert_with(|| e.to_string());
                        }
                    }
                }
            }
        }
    }

    if is_final {
        let context = unsafe { Box::from_raw(userp as *mut InferenceContext) };
        let result = match context.error {
            Some(error) => InferenceResult::failed(error),
            None => InferenceResult {
                outputs: context.pending_outputs,
                error: None,
            },
        };
        context.completion.complete(result);
    }
}
