use crate::TritonError;
use crate::server::Server;
use crate::types::DataType;
use crate::utils::cstring_from_str;
use std::ffi::c_void;
use std::ptr;

pub struct InferenceRequest {
    ptr: *mut triton_sys::TRITONSERVER_InferenceRequest,
}

impl InferenceRequest {
    pub fn new(server: &Server, model_name: &str, model_version: i64) -> Result<Self, TritonError> {
        let mut request_ptr: *mut triton_sys::TRITONSERVER_InferenceRequest = ptr::null_mut();
        let model_name_cstr = cstring_from_str(model_name)?;

        ffi_call!(triton_sys::TRITONSERVER_InferenceRequestNew(
            &mut request_ptr,
            server.as_ptr(),
            model_name_cstr.as_ptr(),
            model_version,
        ))?;

        ensure_ptr!(request_ptr)?;

        Ok(Self { ptr: request_ptr })
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONSERVER_InferenceRequest {
        self.ptr
    }

    pub fn add_input(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
    ) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;

        ffi_call!(triton_sys::TRITONSERVER_InferenceRequestAddInput(
            self.ptr,
            name_cstr.as_ptr(),
            datatype.to_sys(),
            shape.as_ptr(),
            shape.len() as u64,
        ))
    }

    pub fn append_input_data(&mut self, name: &str, data: &[u8]) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;

        ffi_call!(triton_sys::TRITONSERVER_InferenceRequestAppendInputData(
            self.ptr,
            name_cstr.as_ptr(),
            data.as_ptr() as *const c_void,
            data.len(),
            triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
            0, // device_id
        ))
    }

    pub fn add_requested_output(&mut self, name: &str) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;

        ffi_call!(triton_sys::TRITONSERVER_InferenceRequestAddRequestedOutput(
            self.ptr,
            name_cstr.as_ptr(),
        ))
    }
}

impl Drop for InferenceRequest {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_sys::TRITONSERVER_InferenceRequestDelete(self.ptr);
            }
        }
    }
}
