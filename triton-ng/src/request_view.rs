use crate::input::Input;
use crate::types::{CorrelationId, Parameter, RequestFlags};
use crate::utils::{cstr_to_string, cstring_from_str};
use crate::{Error, ErrorCode, TritonError};
use libc::{c_char, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::time::Duration;
use triton_ng_macros::triton_call;

/// A request of a batch Triton is forming, given to
/// [`BatchStrategy::include`](crate::BatchStrategy::include).
///
/// Triton owns the request and hands it to `model_instance_execute` once the
/// batch is formed, so the view only reads: it has no final response and no
/// [`Response`](crate::Response). It is a pure borrow, neither `Send` nor
/// `Sync`, unlike the `Request` it borrows, which is `Send`.
///
/// # Examples
///
/// ```ignore
/// fn include(&self, batch: &mut Batch, request: &RequestView<'_>) -> Result<bool, Error> {
///     let key = request.parameter("batch_key")?.and_then(|p| match p {
///         Parameter::String(key) => Some(key),
///         _ => None,
///     });
///     Ok(key == Some(batch.key.clone()))
/// }
/// ```
pub struct RequestView<'a> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
    _marker: PhantomData<&'a ()>,
}

impl<'a> RequestView<'a> {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Request) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// The ID the client gave the request, `None` when it has none, see
    /// [`Request::id`](crate::Request::id).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let id = request.id()?.unwrap_or_default();
    /// batch.ids.push(id);
    /// ```
    pub fn id(&self) -> Result<Option<String>, TritonError> {
        let mut id: *const c_char = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestId(self.ptr, &mut id))?;
        // SAFETY: the string is owned by the request, which outlives the call.
        let id = unsafe { cstr_to_string(id) };
        Ok((!id.is_empty()).then_some(id))
    }

    /// The correlation ID of the request, `None` when it is not part of a
    /// sequence, see [`Request::correlation_id`](crate::Request::correlation_id).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let same_sequence = request.correlation_id()? == batch.correlation_id;
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

    /// Whether the request starts or ends its sequence, see
    /// [`Request::flags`](crate::Request::flags).
    ///
    /// Fails when Triton sets flags unknown to this version of triton-ng.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let starts = request.flags()?.sequence_start;
    /// ```
    pub fn flags(&self) -> Result<RequestFlags, TritonError> {
        let mut flags = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestFlags(
            self.ptr, &mut flags
        ))?;
        RequestFlags::from_sys(flags).map_err(TritonError::from_message)
    }

    /// How long the client lets the request wait in the queue, `None` when it
    /// set no timeout, see [`Request::timeout`](crate::Request::timeout).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if request.timeout()?.is_some() {
    ///     batch.has_deadline = true;
    /// }
    /// ```
    pub fn timeout(&self) -> Result<Option<Duration>, TritonError> {
        let mut micros = 0u64;
        triton_call!(
            triton_ng_sys::TRITONBACKEND_InferenceRequestTimeoutMicroseconds(self.ptr, &mut micros)
        )?;
        Ok((micros != 0).then(|| Duration::from_micros(micros)))
    }

    /// The names of the inputs of the request, in the order the client sent
    /// them, see [`Request::input_names`](crate::Request::input_names).
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

    /// The names of the outputs the client asked for, see
    /// [`Request::output_names`](crate::Request::output_names).
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

    /// The input `name` of the request, borrowed from it, see
    /// [`Request::get_input`](crate::Request::get_input).
    ///
    /// The returned [`Input`] cannot be shared into a BLS request: the
    /// request has no final response yet, so nothing would hold it back.
    ///
    /// Fails when the request has no such input.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let shape = request.get_input("AUDIO")?.properties()?.shape;
    /// let elements = shape.iter().map(|d| d.max(&0)).product::<i64>();
    /// ```
    pub fn get_input(&self, name: &str) -> Result<Input<'_, Self>, TritonError> {
        let ptr = self.input_ptr(name)?;
        Ok(Input::from_ptr(ptr, self))
    }

    /// The `TRITONBACKEND_Input` of the request for `name`.
    pub(crate) fn input_ptr(
        &self,
        name: &str,
    ) -> Result<*mut triton_ng_sys::TRITONBACKEND_Input, TritonError> {
        let name = cstring_from_str(name)?;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestInput(
            self.ptr,
            name.as_ptr(),
            &mut _
        ))
    }

    /// The value of the request parameter `key`, `None` when the request has
    /// no such parameter, see [`Request::parameter`](crate::Request::parameter).
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

    /// Returns `true` if the client cancelled the request, see
    /// [`Request::is_cancelled`](crate::Request::is_cancelled).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if request.is_cancelled()? {
    ///     return Ok(false);
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
}
