use crate::Input;
use crate::error::{Error, TritonError};
use crate::types::Parameter;
use crate::utils::{cstr_to_string, cstring_from_str};
use libc::{c_char, c_void};
use std::ptr;
use triton_ng_macros::triton_call;

/// Inference request passed to [`crate::Backend::model_instance_execute`].
///
/// The backend owns the request and can answer it from any thread of the
/// instance runtime. The
/// request gets exactly one final response, sent by [`crate::Response::send`],
/// [`Request::respond_error`] or [`Request::respond_cancelled`], all of which
/// consume it. A request dropped without a response gets an error response.
/// Either way the request is released to Triton when it is dropped.
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
        }
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

    pub fn is_cancelled(&self) -> Result<bool, TritonError> {
        let mut cancelled = false;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestIsCancelled(
            self.ptr,
            &mut cancelled
        ))?;
        Ok(cancelled)
    }

    /// Sends a final response carrying `message` as an internal error.
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
    pub fn respond_error(mut self, message: &str) -> Result<(), TritonError> {
        self.respond_with(TritonError::internal(message))
    }

    /// Sends a final response with the CANCELLED status, see
    /// [`Request::is_cancelled`].
    pub fn respond_cancelled(mut self) -> Result<(), TritonError> {
        self.respond_with(TritonError::cancelled("request was cancelled"))
    }

    fn respond_with(&mut self, error: TritonError) -> Result<(), TritonError> {
        if self.responded {
            return Err(TritonError::from_message(
                "the request already has a final response",
            ));
        }

        // The caller keeps ownership of the error object passed to ResponseSend,
        // it is deleted on drop.
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
            && let Err(e) = self.respond_with(TritonError::internal(
                "backend did not produce a response for the request",
            ))
        {
            crate::log::error(&format!("failed to send error response: {e}"));
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
