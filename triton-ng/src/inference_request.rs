use crate::TritonError;
use crate::memory::{InputBuffer, OutputBuffer};
use crate::server::Server;
use crate::types::DataType;
use crate::utils::cstring_from_str;
use std::collections::HashMap;
use triton_ng_macros::triton_call;

/// Buffers referenced by a submitted request, see `Server::infer_async`.
pub(crate) type InputBuffers = Vec<Box<dyn Send>>;

/// Buffers that requested outputs are written into, by output name. `None`
/// once the buffer is handed to Triton.
pub(crate) type OutputBuffers = HashMap<String, Option<OutputBuffer>>;

pub struct InferenceRequest {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
    /// Input buffers referenced by the request. Triton does not copy input
    /// data, so the buffers must outlive the request: they are handed over to
    /// the release callback together with the request (see `Server::infer_async`).
    buffers: InputBuffers,
    outputs: OutputBuffers,
}

// SAFETY: Triton lets a request be built and submitted from any thread, it is
// only not safe to use one request from several threads at once, and
// `InferenceRequest` is not `Sync`. The buffers are `Send`.
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
        })
    }

    /// Splits the request into the raw pointer, the input buffers it
    /// references and the buffers of its outputs, without running `Drop`.
    pub(crate) fn into_raw_parts(
        self,
    ) -> (
        *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
        InputBuffers,
        OutputBuffers,
    ) {
        let mut this = std::mem::ManuallyDrop::new(self);
        (
            this.ptr,
            std::mem::take(&mut this.buffers),
            std::mem::take(&mut this.outputs),
        )
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

    /// Requests the output `name`, copied to CPU memory that the result owns.
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

impl Drop for InferenceRequest {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONSERVER_InferenceRequestDelete(self.ptr);
            }
        }
    }
}
