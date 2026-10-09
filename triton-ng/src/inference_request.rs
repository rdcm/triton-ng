use crate::TritonError;
use crate::server::Server;
use crate::types::DataType;
use crate::utils::cstring_from_str;
use std::ffi::c_void;
use triton_ng_macros::triton_call;

pub struct InferenceRequest {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
    /// Input buffers referenced by the request. Triton does not copy input
    /// data, so the buffers must outlive the request: they are handed over to
    /// the release callback together with the request (see `Server::infer_async`).
    buffers: Vec<Vec<u8>>,
}

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
        })
    }

    /// Splits the request into the raw pointer and the input buffers it
    /// references, without running `Drop`.
    pub(crate) fn into_raw_parts(
        self,
    ) -> (
        *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
        Vec<Vec<u8>>,
    ) {
        let mut this = std::mem::ManuallyDrop::new(self);
        (this.ptr, std::mem::take(&mut this.buffers))
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
        let name_cstr = cstring_from_str(name)?;

        // Moving a Vec does not move its heap allocation, so the pointer stays
        // valid after the buffer is pushed into `self.buffers`.
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestAppendInputData(
            self.ptr,
            name_cstr.as_ptr(),
            data.as_ptr() as *const c_void,
            data.len(),
            triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
            0, // device_id
        ))?;

        self.buffers.push(data);

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
        self.add_input(name, datatype, shape)?;
        self.append_input_data_owned(name, data)
    }

    pub fn add_requested_output(&mut self, name: &str) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;

        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestAddRequestedOutput(
                self.ptr,
                name_cstr.as_ptr(),
            )
        )
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
