use crate::TritonError;
use crate::server::OutputTensor;
use crate::types::DataType;
use crate::utils::cstr_to_string;
use std::ffi::{c_char, c_void};
use std::ptr;
use std::slice;

pub struct InferenceResponse {
    ptr: *mut triton_sys::TRITONSERVER_InferenceResponse,
}

impl InferenceResponse {
    pub fn from_ptr(
        ptr: *mut triton_sys::TRITONSERVER_InferenceResponse,
    ) -> Result<Self, TritonError> {
        ensure_ptr!(ptr)?;
        Ok(Self { ptr })
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONSERVER_InferenceResponse {
        self.ptr
    }

    pub fn error(&self) -> Option<TritonError> {
        let error_ptr = unsafe { triton_sys::TRITONSERVER_InferenceResponseError(self.ptr) };

        if error_ptr.is_null() {
            None
        } else {
            Some(unsafe { TritonError::new(error_ptr) })
        }
    }

    pub fn outputs(&self) -> Result<Vec<OutputTensor>, TritonError> {
        unsafe {
            let mut output_count: u32 = 0;
            let err =
                triton_sys::TRITONSERVER_InferenceResponseOutputCount(self.ptr, &mut output_count);

            if !err.is_null() {
                return Err(TritonError::new(err));
            }

            let mut outputs = Vec::new();

            for i in 0..output_count {
                if let Ok(output) = self.get_output(i) {
                    outputs.push(output);
                }
            }

            Ok(outputs)
        }
    }

    fn get_output(&self, index: u32) -> Result<OutputTensor, TritonError> {
        unsafe {
            let mut name_ptr: *const c_char = ptr::null();
            let mut datatype: triton_sys::TRITONSERVER_DataType = 0;
            let mut shape_ptr: *const i64 = ptr::null();
            let mut dim_count: u64 = 0;
            let mut base: *const c_void = ptr::null();
            let mut byte_size: usize = 0;
            let mut memory_type: triton_sys::TRITONSERVER_MemoryType = 0;
            let mut memory_type_id: i64 = 0;
            let mut userp: *mut c_void = ptr::null_mut();

            let err = triton_sys::TRITONSERVER_InferenceResponseOutput(
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
            );

            if !err.is_null() {
                return Err(TritonError::new(err));
            }

            let name = if name_ptr.is_null() {
                format!("output_{}", index)
            } else {
                cstr_to_string(name_ptr)
            };

            let shape: Vec<i64> = slice::from_raw_parts(shape_ptr, dim_count as usize).to_vec();
            let data = slice::from_raw_parts(base as *const u8, byte_size).to_vec();

            Ok(OutputTensor {
                name,
                data,
                shape,
                datatype: DataType::from_sys(datatype),
            })
        }
    }
}

impl Drop for InferenceResponse {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_sys::TRITONSERVER_InferenceResponseDelete(self.ptr);
            }
        }
    }
}
