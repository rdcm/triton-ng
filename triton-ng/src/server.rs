use crate::inference_request::InferenceRequest;
use crate::inference_response::InferenceResponse;
use crate::request::decode_le;
use crate::response_allocator::ResponseAllocator;
use crate::types::DataType;
use crate::utils::{Message, cstring_from_str};
use crate::{Error, TritonError};
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};
use triton_ng_macros::triton_call;

pub struct OutputTensor {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub datatype: crate::types::DataType,
}

impl OutputTensor {
    pub fn as_fp32_vec(&self) -> Result<Vec<f32>, Error> {
        self.expect_datatype(DataType::Fp32)?;
        decode_le(&self.data, f32::from_le_bytes)
    }

    pub fn as_i32_vec(&self) -> Result<Vec<i32>, Error> {
        self.expect_datatype(DataType::Int32)?;
        decode_le(&self.data, i32::from_le_bytes)
    }

    pub fn as_i64_vec(&self) -> Result<Vec<i64>, Error> {
        self.expect_datatype(DataType::Int64)?;
        decode_le(&self.data, i64::from_le_bytes)
    }

    /// Integer tensor widened to `i64`, accepts both INT32 and INT64.
    pub fn as_index_vec(&self) -> Result<Vec<i64>, Error> {
        match self.datatype {
            DataType::Int32 => Ok(decode_le(&self.data, i32::from_le_bytes)?
                .into_iter()
                .map(i64::from)
                .collect()),
            _ => self.as_i64_vec(),
        }
    }

    fn expect_datatype(&self, expected: DataType) -> Result<(), Error> {
        if self.datatype != expected {
            return Err(format!(
                "output '{}' has datatype {:?}, expected {:?}",
                self.name, self.datatype, expected
            )
            .into());
        }
        Ok(())
    }
}

pub struct InferenceResult {
    pub outputs: Vec<OutputTensor>,
    pub error: Option<String>,
}

impl InferenceResult {
    /// Returns the output tensor named `name`.
    pub fn output(&self, name: &str) -> Result<&OutputTensor, Error> {
        self.outputs
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| format!("missing output '{name}'").into())
    }
}

impl InferenceResult {
    fn failed(message: impl Into<String>) -> Self {
        Self {
            outputs: vec![],
            error: Some(message.into()),
        }
    }
}

pub struct InferenceContext {
    completion: Arc<Completion>,
    allocator: ResponseAllocator,
    pending_outputs: Vec<OutputTensor>,
    error: Option<String>,
}

/// Slot shared by the response callback, which fills it once, and
/// [`InferenceFuture`], which takes the result out of it.
struct Completion {
    state: Mutex<CompletionState>,
}

#[derive(Default)]
struct CompletionState {
    result: Option<InferenceResult>,
    waker: Option<Waker>,
    taken: bool,
}

impl Completion {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CompletionState::default()),
        })
    }

    fn failed(message: impl Into<String>) -> Arc<Self> {
        let completion = Self::new();
        completion.complete(InferenceResult::failed(message));
        completion
    }

    fn lock(&self) -> MutexGuard<'_, CompletionState> {
        // The state stays consistent even if a holder panicked: every update is a single assignment.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Stores the result and wakes the waiting task. Only the first result is kept.
    fn complete(&self, result: InferenceResult) {
        let waker = {
            let mut state = self.lock();
            if state.result.is_some() || state.taken {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        // Woken outside the lock: an executor may poll the future right away.
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn take(state: &mut CompletionState) -> Option<Result<InferenceResult, String>> {
        if let Some(result) = state.result.take() {
            state.taken = true;
            return Some(match result.error {
                Some(error) => Err(error),
                None => Ok(result),
            });
        }
        if state.taken {
            return Some(Err("inference result was already taken".to_string()));
        }
        None
    }
}

/// Inference running in Triton, resolves to its result.
///
/// The request is already submitted when the future is created, so it runs
/// whether the future is polled or not. Dropping the future does not cancel
/// the request: Triton finishes it and the result is discarded.
///
/// The future holds no Triton pointers, it is `Send + 'static` and works
/// with any executor. Triton wakes it from its own threads.
pub struct InferenceFuture {
    completion: Arc<Completion>,
}

impl InferenceFuture {
    fn poll_result(&self, cx: &mut Context<'_>) -> Poll<Result<InferenceResult, String>> {
        let mut state = self.completion.lock();
        if let Some(result) = Completion::take(&mut state) {
            return Poll::Ready(result);
        }
        match &state.waker {
            Some(waker) if waker.will_wake(cx.waker()) => {}
            _ => state.waker = Some(cx.waker().clone()),
        }
        Poll::Pending
    }
}

impl Future for InferenceFuture {
    type Output = Result<InferenceResult, TritonError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.poll_result(cx).map_err(TritonError::from_message)
    }
}

/// The Triton server the backend runs in, used for BLS: inference of other
/// models from a backend. Batches get it from [`crate::Context::server`].
///
/// `Server` is `Send + Sync + Clone`. The server stays valid while any model
/// is loaded, and a model stays loaded while any of its [`crate::Request`]s
/// is alive.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context<State>, requests: Vec<Request>) -> Result<(), Error> {
///     let mut inference = InferenceRequest::new(ctx.server(), &ctx.state().encoder, -1)?;
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
            Ok(()) => InferenceFuture { completion },
            Err(e) => InferenceFuture {
                completion: Completion::failed(e.to_string()),
            },
        }
    }

    fn submit(
        &self,
        request: InferenceRequest,
        completion: Arc<Completion>,
    ) -> Result<(), TritonError> {
        let allocator = ResponseAllocator::new()?;

        let (raw, buffers) = request.into_raw_parts();
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
            pending_outputs: Vec::new(),
            error: None,
        });
        let context_ptr = Box::into_raw(context) as *mut c_void;

        if let Err(e) = triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetResponseCallback(
                raw,
                (*context_ptr.cast::<InferenceContext>()).allocator.as_ptr(),
                ptr::null_mut(),
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
/// `userp` is the boxed `Vec<Vec<u8>>` of input buffers referenced by the
/// request, they are freed only after the request itself is deleted.
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
        unsafe { drop(Box::from_raw(userp as *mut Vec<Vec<u8>>)) };
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

#[cfg(test)]
mod tests {
    use super::{Completion, InferenceFuture, InferenceResult};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    struct CountingWaker(AtomicUsize);

    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn poll(future: &InferenceFuture, waker: &Waker) -> Poll<Result<usize, String>> {
        future
            .poll_result(&mut Context::from_waker(waker))
            .map(|r| r.map(|result| result.outputs.len()))
    }

    fn succeeded() -> InferenceResult {
        InferenceResult {
            outputs: vec![],
            error: None,
        }
    }

    #[test]
    fn handles_can_move_to_tasks() {
        fn assert_send<T: Send + 'static>() {}
        fn assert_shared<T: Send + Sync + Clone + 'static>() {}
        assert_send::<InferenceFuture>();
        assert_send::<crate::InferenceRequest>();
        assert_shared::<super::Server>();
    }

    #[test]
    fn completion_wakes_pending_future() {
        let counter = Arc::new(CountingWaker(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&counter));
        let completion = Completion::new();
        let future = InferenceFuture {
            completion: Arc::clone(&completion),
        };

        assert_eq!(poll(&future, &waker), Poll::Pending);
        completion.complete(succeeded());

        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        assert_eq!(poll(&future, &waker), Poll::Ready(Ok(0)));
    }

    #[test]
    fn completion_from_another_thread() {
        let completion = Completion::new();
        let future = InferenceFuture {
            completion: Arc::clone(&completion),
        };

        std::thread::spawn(move || completion.complete(succeeded()))
            .join()
            .unwrap();

        assert_eq!(poll(&future, Waker::noop()), Poll::Ready(Ok(0)));
    }

    #[test]
    fn failed_completion_resolves_to_error() {
        let future = InferenceFuture {
            completion: Completion::failed("submit failed"),
        };

        assert_eq!(
            poll(&future, Waker::noop()),
            Poll::Ready(Err("submit failed".to_string()))
        );
    }

    #[test]
    fn only_first_result_is_kept() {
        let completion = Completion::new();
        let future = InferenceFuture {
            completion: Arc::clone(&completion),
        };

        completion.complete(succeeded());
        completion.complete(InferenceResult::failed("late"));

        assert_eq!(poll(&future, Waker::noop()), Poll::Ready(Ok(0)));
    }

    #[test]
    fn polling_after_completion_is_an_error() {
        let future = InferenceFuture {
            completion: Completion::failed("first"),
        };

        assert!(poll(&future, Waker::noop()).is_ready());
        assert_eq!(
            poll(&future, Waker::noop()),
            Poll::Ready(Err("inference result was already taken".to_string()))
        );
    }
}
