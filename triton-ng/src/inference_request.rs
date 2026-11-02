use crate::error::ErrorCode;
use crate::inference_trace::InferenceTrace;
use crate::server::Server;
use crate::types::{CorrelationId, DataType, Parameter, RequestFlags};
use crate::utils::cstring_from_str;
use crate::{InputBuffer, OutputBuffer, Request, TritonError};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use triton_ng_macros::triton_call;

/// Buffers referenced by a submitted request, see `Server::infer_async`.
pub(crate) type InputBuffers = Vec<Box<dyn Send>>;

/// Buffers that requested outputs are written into, by output name. `None`
/// once the buffer is handed to Triton.
pub(crate) type OutputBuffers = HashMap<String, Option<OutputBuffer>>;

/// A BLS request to another model, submitted with [`crate::Server::infer_async`].
///
/// Owns the `TRITONSERVER_InferenceRequest` and its input buffers. Dropped
/// before it is submitted, it deletes the request; once submitted, Triton
/// owns it and the release callback deletes it. `Send`, not `Sync`.
///
/// # Examples
///
/// ```ignore
/// let mut inference = InferenceRequest::new(ctx.server(), "encoder", -1)?;
/// inference.add_input_with_data("AUDIO", DataType::Fp32, &[1, samples], bytes)?;
/// inference.add_requested_output("EMBEDDING")?;
/// let result = ctx.server().infer_async(inference).await?;
/// ```
pub struct InferenceRequest {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
    /// Input buffers referenced by the request. Triton does not copy input
    /// data, so the buffers must outlive the request: they are handed over to
    /// the release callback together with the request (see `Server::infer_async`).
    buffers: InputBuffers,
    outputs: OutputBuffers,
    trace: Option<InferenceTrace>,
    /// Names of the parameters set: Triton keeps duplicates, and the called
    /// model would see only one of them.
    parameters: HashSet<String>,
}

// SAFETY: Triton lets a request be built and submitted from any thread, it is
// only not safe to use one request from several threads at once, and
// `InferenceRequest` is not `Sync`. The buffers and the trace are `Send`.
unsafe impl Send for InferenceRequest {}

impl InferenceRequest {
    pub fn new(server: &Server, model_name: &str, model_version: i64) -> Result<Self, TritonError> {
        let model_name_cstr = cstring_from_str(model_name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestNew(
            &mut _,
            server.as_ptr(),
            model_name_cstr.as_ptr(),
            model_version,
        ))?;
        Ok(Self {
            ptr,
            buffers: Vec::new(),
            outputs: HashMap::new(),
            trace: None,
            parameters: HashSet::new(),
        })
    }

    /// Splits the request into the raw pointer, the input buffers it
    /// references, the buffers of its outputs and its trace, without running
    /// `Drop`.
    pub(crate) fn into_raw_parts(
        self,
    ) -> (
        *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
        InputBuffers,
        OutputBuffers,
        Option<InferenceTrace>,
    ) {
        let mut this = std::mem::ManuallyDrop::new(self);
        (
            this.ptr,
            std::mem::take(&mut this.buffers),
            std::mem::take(&mut this.outputs),
            this.trace.take(),
        )
    }

    /// Sets the ID of the request, which the called model gets from
    /// [`Request::id`] and Triton writes into its logs and traces. Pass the
    /// ID of the client request on to follow it through a chain of models.
    ///
    /// Fails when `id` contains a NUL byte.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(id) = request.id()? {
    ///     inference.set_id(&id)?;
    /// }
    /// ```
    pub fn set_id(&mut self, id: &str) -> Result<(), TritonError> {
        let id = cstring_from_str(id)?;
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestSetId(
            self.ptr,
            id.as_ptr()
        ))
    }

    /// Sets the correlation ID of the request, which stateful models use to
    /// tell sequences apart, see [`CorrelationId`].
    ///
    /// Fails for a zero or empty ID, which Triton reads as no ID, and for a
    /// string longer than 128 bytes or containing a NUL byte.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(id) = request.correlation_id()? {
    ///     inference.set_correlation_id(id)?;
    ///     inference.set_flags(request.flags()?)?;
    /// }
    /// ```
    pub fn set_correlation_id(&mut self, id: impl Into<CorrelationId>) -> Result<(), TritonError> {
        match id.into() {
            CorrelationId::Int(0) => Err(TritonError::with_code(
                ErrorCode::InvalidArg,
                "correlation ID 0 means no correlation ID",
            )),
            CorrelationId::Int(id) => {
                triton_call!(
                    triton_ng_sys::TRITONSERVER_InferenceRequestSetCorrelationId(self.ptr, id)
                )
            }
            CorrelationId::String(id) => {
                validate_string_correlation_id(&id)
                    .map_err(|e| TritonError::with_code(ErrorCode::InvalidArg, e))?;
                let id = cstring_from_str(&id)?;
                triton_call!(
                    triton_ng_sys::TRITONSERVER_InferenceRequestSetCorrelationIdString(
                        self.ptr,
                        id.as_ptr()
                    )
                )
            }
        }
    }

    /// Sets whether the request starts or ends its sequence, see
    /// [`RequestFlags`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.set_correlation_id(session)?;
    /// inference.set_flags(RequestFlags { sequence_start: true, sequence_end: false })?;
    /// ```
    pub fn set_flags(&mut self, flags: RequestFlags) -> Result<(), TritonError> {
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestSetFlags(
            self.ptr,
            flags.to_sys()
        ))
    }

    /// Sets how long the request may wait in the queue of the called model
    /// before Triton fails it with UNAVAILABLE. Applies only to models whose
    /// scheduler supports timeouts (dynamic batching).
    ///
    /// Fails for a zero timeout, which Triton reads as no timeout, and for
    /// one that does not fit `u64` microseconds.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(timeout) = request.timeout()? {
    ///     inference.set_timeout(timeout)?;
    /// }
    /// ```
    pub fn set_timeout(&mut self, timeout: Duration) -> Result<(), TritonError> {
        let micros = timeout_micros(timeout)
            .map_err(|e| TritonError::with_code(ErrorCode::InvalidArg, e))?;
        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(self.ptr, micros)
        )
    }

    /// Sets the priority of the request in the queue of the called model:
    /// 1 is the highest, 0 the default of the model. Applies only to models
    /// with `priority_levels` in their dynamic batching config; Triton moves
    /// a priority outside the levels to the default one.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.set_priority(1)?;
    /// ```
    pub fn set_priority(&mut self, priority: u64) -> Result<(), TritonError> {
        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetPriorityUInt64(self.ptr, priority)
        )
    }

    /// Adds the parameter `name` to the request. The called model reads it
    /// with [`Request::parameter`].
    ///
    /// Fails when `name` or a string value contains a NUL byte, or when the
    /// request already has the parameter `name`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.set_parameter("language", "ru")?;
    /// inference.set_parameter("beam_size", 4)?;
    /// ```
    pub fn set_parameter(
        &mut self,
        name: &str,
        value: impl Into<Parameter>,
    ) -> Result<(), TritonError> {
        if self.parameters.contains(name) {
            return Err(TritonError::with_code(
                ErrorCode::AlreadyExists,
                format!("parameter '{name}' is already set"),
            ));
        }
        let name_cstr = cstring_from_str(name)?;
        match value.into() {
            Parameter::Bool(value) => {
                triton_call!(
                    triton_ng_sys::TRITONSERVER_InferenceRequestSetBoolParameter(
                        self.ptr,
                        name_cstr.as_ptr(),
                        value
                    )
                )
            }
            Parameter::Int(value) => {
                triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestSetIntParameter(
                    self.ptr,
                    name_cstr.as_ptr(),
                    value
                ))
            }
            Parameter::Double(value) => {
                triton_call!(
                    triton_ng_sys::TRITONSERVER_InferenceRequestSetDoubleParameter(
                        self.ptr,
                        name_cstr.as_ptr(),
                        value
                    )
                )
            }
            Parameter::String(value) => {
                let value_cstr = cstring_from_str(&value)?;
                triton_call!(
                    triton_ng_sys::TRITONSERVER_InferenceRequestSetStringParameter(
                        self.ptr,
                        name_cstr.as_ptr(),
                        value_cstr.as_ptr()
                    )
                )
            }
        }?;
        self.parameters.insert(name.to_string());
        Ok(())
    }

    /// Makes the request a child of the trace of `request`, so that Triton
    /// traces show it inside the client request it is made for. Does nothing
    /// when `request` is not traced. A batched BLS request is made for many
    /// requests and can be the child of one of them.
    ///
    /// `request` is then released to Triton only after this BLS request and
    /// its response are done, even if `request` is answered and dropped
    /// before: its trace must outlive the child.
    ///
    /// Fails when the request already has a parent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut inference = InferenceRequest::new(ctx.server(), "encoder", -1)?;
    /// inference.set_parent_trace(&requests[0])?;
    /// ```
    pub fn set_parent_trace(&mut self, request: &Request) -> Result<(), TritonError> {
        if self.trace.is_some() {
            return Err(TritonError::from_message(
                "the request already has a parent trace",
            ));
        }
        self.trace = InferenceTrace::spawn(request)?;
        Ok(())
    }

    pub fn add_input(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
    ) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;

        triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestAddInput(
            self.ptr,
            name_cstr.as_ptr(),
            datatype.to_sys(),
            shape.as_ptr(),
            shape.len() as u64,
        ))
    }

    /// Appends a copy of `data` to the input `name`.
    pub fn append_input_data(&mut self, name: &str, data: &[u8]) -> Result<(), TritonError> {
        self.append_input_data_owned(name, data.to_vec())
    }

    /// Appends `data` to the input `name`, the request takes ownership of the buffer.
    pub fn append_input_data_owned(
        &mut self,
        name: &str,
        data: Vec<u8>,
    ) -> Result<(), TritonError> {
        self.append_input_buffer(name, data)
    }

    /// Appends `buffer` to the input `name` without copying it; Triton reads
    /// it from the memory it is in, see [`InputBuffer`]. The request owns the
    /// buffer until Triton releases the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.add_input("hi", DataType::Fp32, &[1, batch, 320])?;
    /// inference.append_input_buffer("hi", Arc::clone(&state))?;
    /// ```
    pub fn append_input_buffer(
        &mut self,
        name: &str,
        buffer: impl InputBuffer,
    ) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;
        let raw = buffer.raw();

        // The memory of every input buffer is on the heap or in Triton's
        // pools, so the pointer stays valid after the buffer is moved into
        // `self.buffers`.
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestAppendInputData(
            self.ptr,
            name_cstr.as_ptr(),
            raw.ptr.cast_const(),
            raw.len,
            raw.memory_type,
            raw.memory_type_id,
        ))?;

        self.buffers.push(Box::new(buffer));

        Ok(())
    }

    /// Adds an input tensor together with its data in one call.
    pub fn add_input_with_data(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
        data: Vec<u8>,
    ) -> Result<(), TritonError> {
        self.add_input_with_buffer(name, datatype, shape, data)
    }

    /// Adds an input tensor whose data is `buffer`, see
    /// [`InferenceRequest::append_input_buffer`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tokens = ctx.memory().pinned(batch * 8)?;
    /// tokens.copy_from_slice(&token_bytes);
    /// inference.add_input_with_buffer("x", DataType::Int64, &[batch, 1], tokens)?;
    /// ```
    pub fn add_input_with_buffer(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
        buffer: impl InputBuffer,
    ) -> Result<(), TritonError> {
        self.add_input(name, datatype, shape)?;
        self.append_input_buffer(name, buffer)
    }

    /// Requests the output `name`, written into CPU memory that the result
    /// takes over without copying it.
    pub fn add_requested_output(&mut self, name: &str) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;

        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestAddRequestedOutput(
                self.ptr,
                name_cstr.as_ptr(),
            )
        )
    }

    /// Requests the output `name` written into `buffer`. The result hands the
    /// buffer back with [`crate::InferenceResult::take_gpu`] (or `take_pinned`,
    /// `take_ram`), so the memory can be cached and reused.
    ///
    /// The inference fails when the output size differs from the length of
    /// the buffer, and when the model returns the output more than once
    /// (decoupled models): the buffer receives one output. Fails right away
    /// when `name` is requested twice.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let state = ctx.memory().gpu(ctx.device_id(), batch * 320 * 4)?;
    /// inference.add_requested_output_into("ho", state)?;
    /// let mut result = ctx.server().infer_async(inference).await?;
    /// let state: GpuBuffer = result.take_gpu("ho")?;
    /// ```
    pub fn add_requested_output_into(
        &mut self,
        name: &str,
        buffer: impl Into<OutputBuffer>,
    ) -> Result<(), TritonError> {
        if self.outputs.contains_key(name) {
            return Err(TritonError::from_message(format!(
                "output '{name}' already has a buffer"
            )));
        }
        self.add_requested_output(name)?;
        self.outputs.insert(name.to_string(), Some(buffer.into()));
        Ok(())
    }
}

/// Triton refuses string correlation IDs longer than 128 bytes; an empty one
/// is no ID.
fn validate_string_correlation_id(id: &str) -> Result<(), String> {
    if id.is_empty() {
        return Err("an empty correlation ID means no correlation ID".to_string());
    }
    if id.len() > 128 {
        return Err(format!(
            "correlation ID of {} bytes is longer than 128 bytes",
            id.len()
        ));
    }
    Ok(())
}

fn timeout_micros(timeout: Duration) -> Result<u64, String> {
    match u64::try_from(timeout.as_micros()) {
        Ok(0) => Err(format!(
            "timeout {timeout:?} is zero microseconds, which means no timeout"
        )),
        Ok(micros) => Ok(micros),
        Err(_) => Err(format!("timeout {timeout:?} does not fit u64 microseconds")),
    }
}

impl Drop for InferenceRequest {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONSERVER_InferenceRequestDelete(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{timeout_micros, validate_string_correlation_id};
    use std::time::Duration;

    #[test]
    fn string_correlation_ids_up_to_128_bytes_are_accepted() {
        assert!(validate_string_correlation_id("call-42").is_ok());
        assert!(validate_string_correlation_id(&"x".repeat(128)).is_ok());
        assert!(validate_string_correlation_id(&"x".repeat(129)).is_err());
        assert!(validate_string_correlation_id("").is_err());
    }

    #[test]
    fn timeouts_are_whole_microseconds() {
        assert_eq!(timeout_micros(Duration::from_millis(5)), Ok(5_000));
        assert!(timeout_micros(Duration::ZERO).is_err());
        assert!(timeout_micros(Duration::from_nanos(999)).is_err());
        assert!(timeout_micros(Duration::MAX).is_err());
    }
}
