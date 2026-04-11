use crate::TritonError;
use crate::server::OutputTensor;
use crate::types::DataType;
use crate::utils::cstr_to_string;
use std::ffi::{c_char, c_void};
use std::ptr;
use std::slice;
use triton_ng_macros::triton_call;

pub struct InferenceResponse {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceResponse,
}

impl InferenceResponse {
    pub(crate) fn from_ptr(
        ptr: *mut triton_ng_sys::TRITONSERVER_InferenceResponse,
    ) -> Result<Self, TritonError> {
        if ptr.is_null() {
            return Err(TritonError::from_message("null inference response"));
        }
        Ok(Self { ptr })
    }

    pub fn error(&self) -> Option<TritonError> {
        let error_ptr = unsafe { triton_ng_sys::TRITONSERVER_InferenceResponseError(self.ptr) };

        if error_ptr.is_null() {
            None
        } else {
            Some(unsafe { TritonError::new(error_ptr) })
        }
    }

    pub fn outputs(&self) -> Result<Vec<OutputTensor>, TritonError> {
        let mut output_count: u32 = 0;
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceResponseOutputCount(
            self.ptr,
            &mut output_count,
        ))?;

        (0..output_count).map(|i| self.get_output(i)).collect()
    }

    fn get_output(&self, index: u32) -> Result<OutputTensor, TritonError> {
        let mut name_ptr: *const c_char = ptr::null();
        let mut datatype: triton_ng_sys::TRITONSERVER_DataType = 0;
        let mut shape_ptr: *const i64 = ptr::null();
        let mut dim_count: u64 = 0;
        let mut base: *const c_void = ptr::null();
        let mut byte_size: usize = 0;
        let mut memory_type: triton_ng_sys::TRITONSERVER_MemoryType = 0;
        let mut memory_type_id: i64 = 0;
        let mut userp: *mut c_void = ptr::null_mut();

        triton_call!(triton_ng_sys::TRITONSERVER_InferenceResponseOutput(
            self.ptr,
            index,
            &mut name_ptr,
            &mut datatype,
            &mut shape_ptr,
            &mut dim_count,
            &mut base,
            &mut byte_size,
            &mut memory_type,
            &mut memory_type_id,
            &mut userp,
        ))?;

        let name = if name_ptr.is_null() {
            format!("output_{index}")
        } else {
            unsafe { cstr_to_string(name_ptr) }
        };

        let shape = unsafe { slice::from_raw_parts(shape_ptr, dim_count as usize).to_vec() };
        let data = unsafe { slice::from_raw_parts(base as *const u8, byte_size).to_vec() };

        Ok(OutputTensor {
            name,
            data,
            shape,
            datatype: DataType::from_sys(datatype),
        })
    }
}

impl Drop for InferenceResponse {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONSERVER_InferenceResponseDelete(self.ptr);
            }
        }
    }
}
