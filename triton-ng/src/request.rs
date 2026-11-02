use crate::Input;
use crate::error::{Error, ErrorCode, TritonError};
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
/// The backend owns the request and can answer it from any thread of the
/// instance runtime. The
/// request gets exactly one final response, sent by [`crate::Response::send`],
/// [`Request::respond_error`] or [`Request::respond_cancelled`], all of which
/// consume it. A request dropped without a response gets an error response.
/// Either way the request is released to Triton when it is dropped, or, when
/// a BLS request traced as its child is still in flight (see
/// [`crate::InferenceRequest::set_parent_trace`]), once that BLS request is
/// done.
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
    responded: bool,
    /// Dropped after `Drop` sends the error response, see `RequestRelease`.
    release: Arc<RequestRelease>,
}

// SAFETY: once ModelInstanceExecute succeeds, Triton hands the request over to
// the backend, which may respond to and release it from any thread, as long as
// one request is not used from several threads at once (tritonbackend.h,
// TRITONBACKEND_Response). `Request` is not `Sync`, so it never is.
unsafe impl Send for Request {}

impl Request {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Request) -> Self {
        Self {
            ptr,
            responded: false,
            release: Arc::new(RequestRelease::new(ptr)),
        }
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

    pub(crate) fn mark_responded(&mut self) {
        self.responded = true;
    }

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

    pub fn get_input(&self, name: &str) -> Result<Input<'_>, TritonError> {
        let name = cstring_from_str(name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_RequestInput(
            self.ptr,
            name.as_ptr(),
            &mut _
        ))?;
        Ok(Input::from_ptr(ptr))
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
        if self.responded {
            return Err(TritonError::from_message(
                "the request already has a final response",
            ));
        }

        // The caller keeps ownership of the error object passed to ResponseSend,
        // it is deleted on drop.
        let error = TritonError::in_triton(code, message);
        if error.as_ptr().is_null() {
            return Err(TritonError::from_message("failed to create error object"));
        }

        let response = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(&mut _, self.ptr))?;

        // ResponseSend takes ownership of the response whatever it returns.
        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            error.as_ptr(),
        ))?;

        self.responded = true;
        Ok(())
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.responded
            && let Err(e) = self.respond_with(
                ErrorCode::Internal,
                "backend did not produce a response for the request",
            )
        {
            crate::log::error(&format!("failed to send error response: {e}"));
        }
    }
}
