use crate::Input;
use crate::ResponseSender;
use crate::batch_statistics::BatchStatistics;
use crate::error::{Error, ErrorCode, TritonError};
use crate::executor::RequestPolicy;
use crate::final_response::{FinalResponse, Pending};
use crate::request_release::RequestRelease;
use crate::types::{CorrelationId, Parameter, RequestFlags};
use crate::utils::{cstr_to_string, cstring_from_str};
use libc::{c_char, c_void};
use std::ptr;
use std::sync::Arc;
use std::time::Duration;
use triton_ng_macros::triton_call;

/// Inference request passed to [`crate::CustomBackend::model_instance_execute`].
///
/// The backend owns the request and can answer it from any thread. The
/// request gets exactly one final response, sent by [`crate::Response::send`],
/// [`Request::respond_error`] or [`Request::respond_cancelled`], all of which
/// consume it. A request of a decoupled model can instead get a stream of
/// responses through [`Request::into_sender`]. A request dropped without a
/// final response gets an error response.
/// Either way the request is released to Triton when it is dropped, or, when
/// a BLS request traced as its child is still in flight (see
/// [`crate::InferenceRequest::set_parent_trace`]), once that BLS request is
/// done. While a [`crate::SharedInput`] of the request is alive, the final
/// response is held back and sent when the last one is dropped, see
/// [`crate::Input::share`].
///
/// Owns the `TRITONBACKEND_Request` and releases it on drop. `Send`, not
/// `Sync`: Triton allows a request to be used from one thread at a time.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///     for request in requests {
///         let text = match request.get_input("TEXT").and_then(|input| input.as_string()) {
///             Ok(text) => text,
///             Err(e) => {
///                 request.respond_error(&e.to_string())?;
///                 continue;
///             }
///         };
///
///         let mut response = Response::new(request)?;
///         response
///             .create_output("TEXT", DataType::Bytes, &[1])?
///             .write_string(&text.to_uppercase())?;
///         response.send()?;
///     }
///     Ok(())
/// }
/// ```
pub struct Request {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
    /// What the model lets its requests do, see [`Request::into_sender`]
    /// and [`Request::reschedule`].
    policy: RequestPolicy,
    responded: bool,
    /// Sends the final response, see `FinalResponse`.
    final_response: Arc<FinalResponse>,
    /// Dropped after `Drop` sends the error response, see `RequestRelease`.
    release: Arc<RequestRelease>,
}

// SAFETY: once ModelInstanceExecute succeeds, Triton hands the request over to
// the backend, which may respond to and release it from any thread, as long as
// one request is not used from several threads at once (tritonbackend.h,
// TRITONBACKEND_Response). `Request` is not `Sync`, so it never is.
unsafe impl Send for Request {}

impl Request {
    pub(crate) fn from_ptr(
        ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
        policy: RequestPolicy,
        statistics: Arc<BatchStatistics>,
    ) -> Self {
        let release = Arc::new(RequestRelease::new(ptr, statistics));
        Self {
            ptr,
            policy,
            responded: false,
            final_response: Arc::new(FinalResponse::new(ptr, Arc::clone(&release))),
            release,
        }
    }

    /// Holds the final response of the request back while the returned value
    /// is alive, see `FinalResponse`.
    pub(crate) fn final_response(&self) -> Arc<FinalResponse> {
        Arc::clone(&self.final_response)
    }

    /// Keeps the request from being released while the returned value is
    /// alive, even after the `Request` is dropped.
    pub(crate) fn release_guard(&self) -> Arc<RequestRelease> {
        Arc::clone(&self.release)
    }

    /// The trace of the request, null when it is not traced. Triton owns it,
    /// it stays valid until the request is released.
    pub(crate) fn trace(
        &self,
    ) -> Result<*mut triton_ng_sys::TRITONSERVER_InferenceTrace, TritonError> {
        let mut trace = ptr::null_mut();
        match triton_call!(triton_ng_sys::TRITONBACKEND_RequestTrace(
            self.ptr, &mut trace
        )) {
            Ok(()) => Ok(trace),
            // Triton built without tracing: no request is traced.
            Err(e) if e.code() == ErrorCode::Unsupported => Ok(ptr::null_mut()),
            Err(e) => Err(e),
        }
    }

    /// The ID the client gave the request (`id` of the gRPC
    /// `ModelInferRequest`), `None` when it has none. Responses carry it
    /// back, and Triton logs it, so it is the ID to log and to pass on to BLS
    /// calls made for the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let id = request.id()?.unwrap_or_default();
    /// log::info(&format!("request {id}: {} samples", samples.len()));
    /// inference.set_id(&id)?;
    /// ```
    pub fn id(&self) -> Result<Option<String>, TritonError> {
        let mut id: *const c_char = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestId(self.ptr, &mut id))?;
        // SAFETY: the string is owned by the request, which outlives the call.
        let id = unsafe { cstr_to_string(id) };
        Ok((!id.is_empty()).then_some(id))
    }

    /// The correlation ID of the request, `None` when it is not part of a
    /// sequence, see [`CorrelationId`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let Some(session) = request.correlation_id()? else {
    ///     request.respond_error_with_code(ErrorCode::InvalidArg, "sequence_id is required")?;
    ///     continue;
    /// };
    /// ```
    pub fn correlation_id(&self) -> Result<Option<CorrelationId>, TritonError> {
        // Triton tells the type of the ID only by failing the getter of the
        // other type with INVALID_ARG.
        let mut number = 0u64;
        match triton_call!(triton_ng_sys::TRITONBACKEND_RequestCorrelationId(
            self.ptr,
            &mut number
        )) {
            Ok(()) => return Ok((number != 0).then_some(CorrelationId::Int(number))),
            Err(e) if e.code() == ErrorCode::InvalidArg => {}
            Err(e) => return Err(e),
        }

        let mut string: *const c_char = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestCorrelationIdString(
            self.ptr,
            &mut string
        ))?;
        // SAFETY: the string is owned by the request, which outlives the call.
        let string = unsafe { cstr_to_string(string) };
        Ok((!string.is_empty()).then_some(CorrelationId::String(string)))
    }

    /// Whether the request starts or ends its sequence, see [`RequestFlags`].
    ///
    /// Fails when Triton sets flags unknown to this version of triton-ng.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if request.flags()?.sequence_end {
    ///     sessions.close(&request.correlation_id()?);
    /// }
    /// ```
    pub fn flags(&self) -> Result<RequestFlags, TritonError> {
        let mut flags = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestFlags(
            self.ptr, &mut flags
        ))?;
        RequestFlags::from_sys(flags).map_err(TritonError::from_message)
    }

    /// How long the client lets the request wait in the queue of the model,
    /// `None` when it set no timeout. Clients set it as the `timeout`
    /// parameter, in microseconds.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(timeout) = request.timeout()? {
    ///     inference.set_timeout(timeout)?;
    /// }
    /// ```
    pub fn timeout(&self) -> Result<Option<Duration>, TritonError> {
        let mut micros = 0u64;
        triton_call!(
            triton_ng_sys::TRITONBACKEND_InferenceRequestTimeoutMicroseconds(self.ptr, &mut micros)
        )?;
        Ok((micros != 0).then(|| Duration::from_micros(micros)))
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONBACKEND_Request {
        self.ptr
    }

    /// The batch size of the request, reported to Triton with the
    /// statistics of its batch: the first dimension of its first input for a
    /// model with `max_batch_size`, 1 for a model without batching.
    ///
    /// Fails when the request has no input or its shape cannot be read.
    pub(crate) fn batch_size(&self) -> Result<u64, Error> {
        if !self.policy.batching {
            return Ok(1);
        }
        let name = self
            .input_names()?
            .into_iter()
            .next()
            .ok_or("the request has no inputs")?;
        let shape = self.get_input(&name)?.properties()?.shape;
        let first = shape
            .first()
            .copied()
            .ok_or("the first input of the request has no dimensions")?;
        Ok(u64::try_from(first).map_err(|_| format!("the batch dimension is {first}"))?)
    }

    /// Gives the final response of the request, sent now or once no shared
    /// input is left. The request counts as answered even when sending
    /// fails: the client then gets an error response, see `FinalResponse`.
    pub(crate) fn send_final(&mut self, pending: Pending) -> Result<(), TritonError> {
        if self.responded {
            return Err(TritonError::from_message(
                "the request already has a final response",
            ));
        }
        self.responded = true;
        self.final_response.send(pending)
    }

    /// The names of the inputs of the request, in the order the client sent them.
    pub fn input_names(&self) -> Result<Vec<String>, TritonError> {
        let mut count = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestInputCount(
            self.ptr, &mut count
        ))?;
        (0..count)
            .map(|i| {
                let mut name: *const c_char = std::ptr::null();
                triton_call!(triton_ng_sys::TRITONBACKEND_RequestInputName(
                    self.ptr, i, &mut name,
                ))?;
                Ok(unsafe { cstr_to_string(name) })
            })
            .collect()
    }

    /// The names of the outputs the client asked for; empty when it asked for all of them.
    pub fn output_names(&self) -> Result<Vec<String>, TritonError> {
        let mut count = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestOutputCount(
            self.ptr, &mut count
        ))?;
        (0..count)
            .map(|i| {
                let mut name: *const c_char = std::ptr::null();
                triton_call!(triton_ng_sys::TRITONBACKEND_RequestOutputName(
                    self.ptr, i, &mut name,
                ))?;
                Ok(unsafe { cstr_to_string(name) })
            })
            .collect()
    }

    /// The input `name` of the request, borrowed from it.
    ///
    /// Fails when the request has no such input.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let audio = request.get_input("AUDIO")?.as_fp32_vec()?;
    /// ```
    pub fn get_input(&self, name: &str) -> Result<Input<'_>, TritonError> {
        let name = cstring_from_str(name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_RequestInput(
            self.ptr,
            name.as_ptr(),
            &mut _
        ))?;
        Ok(Input::from_ptr(ptr, self))
    }

    /// The value of the request parameter `key`, `None` when the request has no
    /// such parameter. Clients set request parameters next to the inputs
    /// (`parameters` of the gRPC `ModelInferRequest`).
    ///
    /// Fails when Triton cannot return the parameters or `key` has the BYTES
    /// type, which is not supported.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let verbose = request.parameter("verbose")? == Some(Parameter::Bool(true));
    /// ```
    pub fn parameter(&self, key: &str) -> Result<Option<Parameter>, Error> {
        let mut count = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestParameterCount(
            self.ptr, &mut count
        ))?;

        for index in 0..count {
            let mut name: *const c_char = ptr::null();
            let mut kind: triton_ng_sys::TRITONSERVER_ParameterType = 0;
            let mut value: *const c_void = ptr::null();
            triton_call!(triton_ng_sys::TRITONBACKEND_RequestParameter(
                self.ptr, index, &mut name, &mut kind, &mut value
            ))?;

            if unsafe { cstr_to_string(name) } == key {
                // SAFETY: Triton returns `value` matching `kind`, owned by the request.
                return unsafe { Parameter::from_sys(kind, value) }
                    .map(Some)
                    .map_err(|e| format!("request parameter '{key}': {e}").into());
            }
        }

        Ok(None)
    }

    /// Turns the request into a [`ResponseSender`], which answers it with any
    /// number of responses and then the final flag.
    ///
    /// Fails when the model is not decoupled
    /// (`model_transaction_policy { decoupled: true }`): clients of other
    /// models wait for exactly one response. The request is answered with
    /// that error then.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut sender = request.into_sender()?;
    /// for digit in digits {
    ///     let mut response = sender.response()?;
    ///     response.create_output("DIGIT", DataType::Int32, &[1])?.write_i32_vec(&[digit])?;
    ///     response.send()?;
    /// }
    /// sender.finish()?;
    /// ```
    pub fn into_sender(self) -> Result<ResponseSender, TritonError> {
        if !self.policy.decoupled {
            let message = "only requests of a decoupled model can get several responses";
            self.respond_error(message)?;
            return Err(TritonError::from_message(message));
        }
        Ok(ResponseSender::new(self))
    }

    /// Hands the request back to Triton without a final response, to be
    /// executed again in a later batch: one step of a generation that Triton
    /// batches with the steps of other sequences
    /// (`sequence_batching { iterative_sequence: true }`).
    ///
    /// The request comes back as a new [`Request`] with the same inputs and
    /// correlation ID, and with neither the start nor the end flag of a
    /// sequence; keep what a step produced in the state of the model, by
    /// correlation ID. Clients of a model that is not decoupled get only the
    /// final response; a decoupled model sends the steps through
    /// [`ResponseSender::reschedule`].
    ///
    /// Triton `r25.09` reschedules a request only once: it hooks the
    /// rescheduling into the request only when it starts a sequence, and the
    /// rescheduled request does not (fixed in `r26.04`, core #456). The
    /// second time the request would leave Triton without a response, so it
    /// is refused: the request comes back once and must then get its final
    /// response.
    ///
    /// Fails, and answers the request with the error, when the model has no
    /// `iterative_sequence` or the request does not start a sequence: Triton
    /// would otherwise drop the request without a response.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// for request in requests {
    ///     let id = request.correlation_id()?.ok_or("no correlation ID")?;
    ///     if state.step(&id)? < STEPS {
    ///         request.reschedule()?;
    ///     } else {
    ///         answer(request, state.finish(&id)?)?;
    ///     }
    /// }
    /// ```
    pub fn reschedule(mut self) -> Result<(), TritonError> {
        if let Some(message) = reschedule_refusal(self.policy, self.flags()?) {
            self.respond_error(message)?;
            return Err(TritonError::from_message(message));
        }
        self.release.reschedule();
        self.responded = true;
        Ok(())
    }

    /// Returns `true` if the client cancelled the request.
    ///
    /// A cancelled request should stop being processed and be answered with
    /// [`Request::respond_cancelled`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// for request in requests {
    ///     if request.is_cancelled()? {
    ///         request.respond_cancelled()?;
    ///         continue;
    ///     }
    ///     // ... process the request ...
    /// }
    /// ```
    pub fn is_cancelled(&self) -> Result<bool, TritonError> {
        let mut cancelled = false;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestIsCancelled(
            self.ptr,
            &mut cancelled
        ))?;
        Ok(cancelled)
    }

    /// Sends a final response carrying `message` as an internal error, see
    /// [`Request::respond_error_with_code`].
    ///
    /// Use it to fail a single request of a batch without failing the others.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Err(e) = validate(&request) {
    ///     request.respond_error(&e.to_string())?;
    /// }
    /// ```
    pub fn respond_error(self, message: &str) -> Result<(), TritonError> {
        self.respond_error_with_code(ErrorCode::Internal, message)
    }

    /// Sends a final response carrying `message` as an error with `code`,
    /// which the client gets as the status of its call, see [`ErrorCode`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let Ok(text) = request.get_input("TEXT").and_then(|input| input.as_string()) else {
    ///     request.respond_error_with_code(ErrorCode::InvalidArg, "TEXT must be a string")?;
    ///     continue;
    /// };
    /// match ctx.server().infer_async(inference).await {
    ///     Ok(result) => answer(request, result)?,
    ///     Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
    /// }
    /// ```
    pub fn respond_error_with_code(
        mut self,
        code: ErrorCode,
        message: &str,
    ) -> Result<(), TritonError> {
        self.respond_with(code, message)
    }

    /// Sends a final response with the CANCELLED status, see
    /// [`Request::is_cancelled`].
    pub fn respond_cancelled(self) -> Result<(), TritonError> {
        self.respond_error_with_code(ErrorCode::Cancelled, "request was cancelled")
    }

    fn respond_with(&mut self, code: ErrorCode, message: &str) -> Result<(), TritonError> {
        self.send_final(Pending::Error(code, message.to_string()))
    }
}

/// Why a request with `flags` of a model with `policy` cannot be
/// rescheduled, see [`Request::reschedule`].
fn reschedule_refusal(policy: RequestPolicy, flags: RequestFlags) -> Option<&'static str> {
    if !policy.iterative_sequence {
        return Some("only requests of a model with iterative_sequence can be rescheduled");
    }
    if !flags.sequence_start {
        return Some("Triton r25.09 reschedules a request only once, when it starts a sequence");
    }
    None
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.responded
            && let Err(e) = self.respond_with(
                ErrorCode::Internal,
                "backend did not produce a final response for the request",
            )
        {
            crate::log::error(&format!("failed to send error response: {e}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reschedule_refusal;
    use crate::executor::RequestPolicy;
    use crate::types::RequestFlags;

    fn flags(sequence_start: bool) -> RequestFlags {
        RequestFlags {
            sequence_start,
            sequence_end: false,
        }
    }

    #[test]
    fn only_a_request_starting_an_iterative_sequence_is_rescheduled() {
        let iterative = RequestPolicy {
            decoupled: true,
            iterative_sequence: true,
            batching: false,
        };
        assert_eq!(reschedule_refusal(iterative, flags(true)), None);
        assert!(reschedule_refusal(iterative, flags(false)).is_some());
        assert!(reschedule_refusal(RequestPolicy::default(), flags(true)).is_some());
    }
}
