use crate::failure::Failure;
use crate::inference_request::InputBuffers;
use crate::inference_response::InferenceResponse;
use crate::inference_stream::StreamResults;
use crate::inference_trace::InferenceTrace;
use crate::message::Message;
use crate::model_control::{check_load_arguments, check_model_control_allowed};
use crate::request_cancel::{CancelOnDrop, RequestCancel};
use crate::request_release::RequestRelease;
use crate::response_allocator::{OutputPlan, ResponseAllocator};
use crate::triton_parameter::TritonParameter;
use crate::utils::cstring_from_str;
use crate::{ErrorCode, InferenceRequest, InferenceResult, InferenceStream, TritonError};
use std::ffi::c_void;
use std::future::Future;
use std::ptr;
use std::sync::Arc;
use tokio::sync::oneshot;
use triton_ng_macros::triton_call;

/// What the response callback of one BLS request needs, handed to Triton as
/// `response_userp` and freed with the final response.
pub(crate) struct InferenceContext {
    receiver: Receiver,
    allocator: Arc<ResponseAllocator>,
    outputs: OutputPlan,
    /// The client request whose trace is the parent of the trace of this
    /// request, see `RequestRelease`: responses report into the trace until
    /// they are deleted, which may happen after the release callback.
    _parent: Option<Arc<RequestRelease>>,
}

/// What the release callback of one BLS request frees, handed to Triton as
/// `release_userp`.
struct ReleaseContext {
    /// Triton may read the input buffers until the request is released.
    _buffers: InputBuffers,
    /// Told before the request is deleted, see `RequestCancel`.
    cancel: Arc<RequestCancel>,
    /// Released after the request is deleted, see `InferenceContext`.
    _parent: Option<Arc<RequestRelease>>,
}

/// Where the response callback of a BLS request puts its results.
enum Receiver {
    /// [`Server::infer_async`]: exactly one response, delivered with the
    /// final flag.
    Single {
        sender: oneshot::Sender<Result<InferenceResult, Failure>>,
        result: Option<Result<InferenceResult, Failure>>,
    },
    /// [`Server::infer_stream`]: every response as it arrives. Dropping the
    /// sender with the final flag ends the stream.
    Stream(StreamResults),
}

impl Receiver {
    fn single(sender: oneshot::Sender<Result<InferenceResult, Failure>>) -> Self {
        Self::Single {
            sender,
            result: None,
        }
    }

    fn response(&mut self, response: Result<InferenceResult, Failure>) {
        match self {
            Self::Single { result, .. } => {
                *result = Some(match result.take() {
                    None => response,
                    Some(_) => Err(Failure {
                        code: ErrorCode::Internal,
                        message: "the model sent more than one response, read the responses \
                                  of a decoupled model with Server::infer_stream"
                            .to_string(),
                    }),
                });
            }
            // A dropped stream discards its results.
            Self::Stream(results) => {
                let _ = results.send(response);
            }
        }
    }

    fn finish(self) {
        if let Self::Single { sender, result } = self {
            // A dropped future discards the result.
            let _ = sender.send(result.unwrap_or_else(|| {
                Err(Failure {
                    code: ErrorCode::Internal,
                    message: "the model sent no response".to_string(),
                })
            }));
        }
    }
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

    /// Major and minor version of the Triton server C API this process has.
    pub fn api_version(&self) -> Result<(u32, u32), TritonError> {
        let mut major: u32 = 0;
        let mut minor: u32 = 0;

        triton_call!(triton_ng_sys::TRITONSERVER_ApiVersion(
            &mut major, &mut minor
        ))?;

        Ok((major, minor))
    }

    /// Whether the server is ready to serve inference: every model that must be loaded is.
    pub fn is_ready(&self) -> Result<bool, TritonError> {
        let mut ready = false;

        triton_call!(triton_ng_sys::TRITONSERVER_ServerIsReady(
            self.ptr, &mut ready
        ))?;

        Ok(ready)
    }

    /// Whether `version` of `model_name` is loaded and ready; `-1` is the latest version.
    ///
    /// Fails for an unknown model.
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

    /// The metadata of `version` of `model_name` (`-1` for the latest) as the JSON of the inference protocol: name, versions, platform, inputs and outputs.
    ///
    /// Fails for an unknown model.
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

    /// Loads the model `model_name`, or reloads it when it is loaded, and
    /// returns once it is ready, like `pb_utils.load_model` of python_backend.
    ///
    /// Without `config` and `files` the model comes from the model
    /// repository. `config` replaces its `config.pbtxt` with the JSON of a
    /// model config. `files` give the model directory instead of the
    /// repository: every name is `file:<path>` with `<path>` relative to the
    /// model directory (`file:1/model.onnx`), and they need `config`. Triton
    /// writes the files to a temporary directory during the call, nothing of
    /// them is kept.
    ///
    /// Blocks the calling thread until the model is loaded, which takes
    /// seconds for a big model. In `model_instance_execute` that is the
    /// thread of the instance, which takes no other batch meanwhile.
    ///
    /// Works only with `--model-control-mode=explicit`, and only after the
    /// server has started: models listed with `--load-model` cannot load
    /// others from their hooks (`Unavailable`). Fails with `Unsupported` in
    /// the finalize hooks and `auto_complete_config`, which run inside a load
    /// or unload of Triton; with `InvalidArg` when `config` is not a JSON
    /// object or a file name is malformed, given twice or leaves the model
    /// directory; and with the error of Triton when the model fails to load.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// server.load_model("punctuation", None, &[])?;
    ///
    /// let config = r#"{"backend": "onnxruntime", "max_batch_size": 8}"#;
    /// let onnx = std::fs::read("/weights/punctuation_v2.onnx")?;
    /// server.load_model("punctuation_v2", Some(config), &[("file:1/model.onnx", &onnx)])?;
    /// ```
    pub fn load_model(
        &self,
        model_name: &str,
        config: Option<&str>,
        files: &[(&str, &[u8])],
    ) -> Result<(), TritonError> {
        check_model_control_allowed("loaded")
            .map_err(|e| TritonError::with_code(ErrorCode::Unsupported, e))?;
        check_load_arguments(model_name, config, files)
            .map_err(|e| TritonError::with_code(ErrorCode::InvalidArg, e))?;

        let model_name = cstring_from_str(model_name)?;
        let mut parameters = Vec::with_capacity(files.len() + 1);
        if let Some(config) = config {
            parameters.push(TritonParameter::string("config", config)?);
        }
        for (name, data) in files {
            parameters.push(TritonParameter::bytes(name, data)?);
        }
        let parameter_ptrs: Vec<_> = parameters.iter().map(TritonParameter::as_ptr).collect();

        // The core reads the parameters, and the bytes of the files they point
        // at, only during the call (ModelRepositoryManager::LoadUnloadModel
        // writes the files out before it returns); `parameters` and `files`
        // outlive it.
        triton_call!(triton_ng_sys::TRITONSERVER_ServerLoadModelWithParameters(
            self.ptr,
            model_name.as_ptr(),
            parameter_ptrs.as_ptr().cast_mut(),
            parameter_ptrs.len() as u64,
        ))
    }

    /// Starts to unload the model `model_name`, and the models loaded along
    /// with it (the composing models of an ensemble) when
    /// `unload_dependents`, like `pb_utils.unload_model` of python_backend.
    ///
    /// Does not wait for the model to unload: Triton unloads it once its
    /// requests in flight are answered, so a model may unload itself from
    /// `model_instance_execute`. Unloading a model that is not loaded
    /// succeeds.
    ///
    /// Works only with `--model-control-mode=explicit` and after the server
    /// has started (`Unavailable` otherwise). Fails with `Unsupported` in the
    /// finalize hooks and `auto_complete_config`, see
    /// [`Server::load_model`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// ctx.server().unload_model("punctuation_v1", false)?;
    /// ctx.server().unload_model("asr_ensemble", true)?;
    /// ```
    pub fn unload_model(
        &self,
        model_name: &str,
        unload_dependents: bool,
    ) -> Result<(), TritonError> {
        check_model_control_allowed("unloaded")
            .map_err(|e| TritonError::with_code(ErrorCode::Unsupported, e))?;
        if model_name.is_empty() {
            return Err(TritonError::with_code(
                ErrorCode::InvalidArg,
                "the model name is empty",
            ));
        }

        let model_name = cstring_from_str(model_name)?;
        if unload_dependents {
            triton_call!(triton_ng_sys::TRITONSERVER_ServerUnloadModelAndDependents(
                self.ptr,
                model_name.as_ptr(),
            ))
        } else {
            triton_call!(triton_ng_sys::TRITONSERVER_ServerUnloadModel(
                self.ptr,
                model_name.as_ptr(),
            ))
        }
    }

    /// Submits inference and returns a future of its result. Submission errors
    /// are reported by the future, with the code Triton gave them (`NotFound`
    /// for an unknown model).
    ///
    /// The request is submitted by this call, not when the future is first
    /// polled: requests made one after another run in Triton at the same
    /// time, whatever order their futures are awaited in.
    ///
    /// Dropping the future cancels the request, unless the model has already
    /// answered; a failure to cancel is only logged.
    ///
    /// The future fails when the model sends other than exactly one
    /// response: the results of a decoupled model are read with
    /// [`Server::infer_stream`]. It holds no Triton pointers, it is
    /// `Send + 'static` and is woken from Triton's threads.
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
    pub fn infer_async(
        &self,
        request: InferenceRequest,
    ) -> impl Future<Output = Result<InferenceResult, TritonError>> + Send + 'static + use<> {
        let (sender, receiver) = oneshot::channel();
        let submitted = self
            .submit(request, Receiver::single(sender))
            .map(CancelOnDrop::new);
        async move {
            let _cancel = submitted?;
            match receiver.await {
                Ok(result) => result.map_err(TritonError::from),
                Err(_) => Err(TritonError::with_code(
                    ErrorCode::Internal,
                    "the response callback ended without a result",
                )),
            }
        }
    }

    /// Submits inference to a decoupled model and returns the stream of its
    /// results, one per response, see [`InferenceStream`]. Submission errors
    /// are the only item of the stream. Request ownership is as for
    /// [`Server::infer_async`].
    ///
    /// Outputs requested into a buffer
    /// ([`InferenceRequest::add_requested_output_into`]) are not supported:
    /// every response needs memory of its own, so the stream fails with
    /// INVALID_ARG and the request is not submitted.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut inference = InferenceRequest::new(ctx.server(), "tokenizer_stream", -1)?;
    /// inference.add_input_with_data("TEXT", DataType::Bytes, &[1], text)?;
    /// inference.add_requested_output("TOKEN")?;
    ///
    /// let mut stream = ctx.server().infer_stream(inference);
    /// while let Some(result) = stream.next().await {
    ///     let token = result?.output("TOKEN")?.as_i64_vec()?;
    ///     // ...
    /// }
    /// ```
    pub fn infer_stream(&self, request: InferenceRequest) -> InferenceStream {
        if request.has_output_buffers() {
            return InferenceStream::failed(Failure {
                code: ErrorCode::InvalidArg,
                message: "infer_stream does not support outputs requested into a buffer, \
                          every response needs memory of its own"
                    .to_string(),
            });
        }
        let (results, receiver) = InferenceStream::channel();
        match self.submit(request, Receiver::Stream(results)) {
            Ok(cancel) => InferenceStream::submitted(receiver, cancel),
            Err(e) => InferenceStream::failed(e.into()),
        }
    }

    fn submit(
        &self,
        request: InferenceRequest,
        receiver: Receiver,
    ) -> Result<Arc<RequestCancel>, TritonError> {
        let allocator = ResponseAllocator::shared()?;

        let (raw, buffers, outputs, trace) = request.into_raw_parts();
        let parent = trace.as_ref().map(InferenceTrace::parent);
        let cancel = RequestCancel::new(raw);
        let buffers_ptr = Box::into_raw(Box::new(ReleaseContext {
            _buffers: buffers,
            cancel: Arc::clone(&cancel),
            _parent: parent.clone(),
        })) as *mut c_void;

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
            receiver,
            allocator,
            outputs: OutputPlan::new(outputs),
            _parent: parent,
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

        // The trace is handed over last: until here a failure deletes it
        // with `trace`, from here on Triton may have taken it, see
        // `InferenceTrace::into_raw`. `_parent` keeps the parent trace alive
        // while Triton may still release the child on a failure.
        let (trace_ptr, _parent) = match trace.map(InferenceTrace::into_raw) {
            Some((trace_ptr, parent)) => (trace_ptr, Some(parent)),
            None => (ptr::null_mut(), None),
        };

        // Transfer ownership to Triton. The release callback now owns deletion.
        if let Err(e) = triton_call!(triton_ng_sys::TRITONSERVER_ServerInferAsync(
            self.ptr, raw, trace_ptr,
        )) {
            // Triton guarantees it will NOT call neither the release nor the
            // response callback on failure, so we must clean up manually.
            unsafe {
                drop(Box::from_raw(context_ptr as *mut InferenceContext));
                inference_request_release(raw, RELEASE_ALL, buffers_ptr);
            }
            if !trace_ptr.is_null() {
                crate::log::warn(&format!(
                    "BLS request with a child trace failed to submit, the trace may have leaked: {e}"
                ));
            }
            return Err(e);
        }

        Ok(cancel)
    }
}

/// Called by Triton after the backend has fully released the inference request.
/// At this point it is safe to delete the request object.
///
/// `userp` is the boxed `ReleaseContext` of the request: the input buffers it
/// references and the guard of its parent request, freed only after the
/// request itself is deleted.
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

    let context =
        (!userp.is_null()).then(|| unsafe { Box::from_raw(userp as *mut ReleaseContext) });
    if let Some(context) = &context {
        context.cancel.released();
    }

    if !request.is_null() {
        unsafe { triton_ng_sys::TRITONSERVER_InferenceRequestDelete(request) };
    }

    drop(context);
}

const RELEASE_ALL: u32 =
    triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL;

/// Called by Triton for each response, the last call with FINAL set, which
/// a decoupled model may send without a response.
///
/// For one-to-one models this is called exactly once with FINAL set.
/// Every response goes to the receiver of the request, which is dropped
/// with the context when FINAL arrives.
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
        let read =
            InferenceResponse::from_ptr(response_ptr).and_then(|response| match response.error() {
                Some(error) => Err(error),
                None => Ok(InferenceResult {
                    outputs: response.outputs()?,
                    parameters: response.parameters()?,
                }),
            });
        context.receiver.response(read.map_err(Failure::from));
    }

    if is_final {
        let context = unsafe { Box::from_raw(userp as *mut InferenceContext) };
        // The rest of the context, the parent guard among it, goes after the result.
        let InferenceContext { receiver, .. } = *context;
        receiver.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::Receiver;
    use crate::failure::Failure;
    use crate::{ErrorCode, InferenceResult, InferenceStream};
    use std::task::Poll;
    use tokio::sync::oneshot;

    fn result() -> Result<InferenceResult, Failure> {
        Ok(InferenceResult {
            outputs: vec![],
            parameters: vec![],
        })
    }

    fn single_result(responses: usize) -> Poll<Result<usize, String>> {
        let (sender, mut received) = oneshot::channel();
        let mut receiver = Receiver::single(sender);
        (0..responses).for_each(|_| receiver.response(result()));
        receiver.finish();
        let result = received.try_recv().expect("finish sends the result");
        Poll::Ready(result.map(|r| r.outputs.len()).map_err(|f| f.message))
    }

    #[test]
    fn single_response_is_the_result() {
        assert_eq!(single_result(1), Poll::Ready(Ok(0)));
    }

    #[test]
    fn no_response_is_an_error() {
        assert_eq!(
            single_result(0),
            Poll::Ready(Err("the model sent no response".to_string()))
        );
    }

    #[test]
    fn several_responses_are_an_error() {
        let Poll::Ready(Err(message)) = single_result(3) else {
            panic!("several responses are not an error");
        };
        assert!(message.contains("more than one response"), "{message}");
    }

    #[test]
    fn stream_receiver_forwards_every_response() {
        let (results, mut stream) = InferenceStream::channel();
        let mut receiver = Receiver::Stream(results);
        receiver.response(result());
        receiver.response(Err(Failure {
            code: ErrorCode::Cancelled,
            message: "cancelled".to_string(),
        }));
        receiver.finish();

        assert!(matches!(stream.try_recv(), Ok(Ok(_))));
        assert!(matches!(stream.try_recv(), Ok(Err(f)) if f.code == ErrorCode::Cancelled));
        assert!(stream.try_recv().is_err());
        assert!(stream.is_closed());
    }
}
