use crate::Error;
use crate::error::TritonError;
use crate::request::Request;
use crate::types::DataType;
use crate::utils::{cstring_from_str, encode_string};
use std::cell::Cell;
use std::ptr;
use std::rc::Rc;
use std::slice;
use triton_ng_macros::triton_call;

pub struct Response {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Response,
    /// Set to true after a successful `send()` so Drop doesn't double-delete.
    /// `TRITONBACKEND_ResponseSend` transfers ownership to Triton.
    sent: bool,
    /// Shared with the originating [`Request`], marks it as answered.
    responded: Rc<Cell<bool>>,
}

impl Response {
    pub fn new(request: &Request) -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(
            &mut _,
            request.as_ptr()
        ))?;
        Ok(Self {
            ptr,
            sent: false,
            responded: request.responded_flag(),
        })
    }

    pub fn create_output(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
    ) -> Result<Output, TritonError> {
        let name_cstr = cstring_from_str(name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseOutput(
            self.ptr,
            &mut _,
            name_cstr.as_ptr(),
            datatype.to_sys(),
            shape.as_ptr(),
            shape.len() as u32,
        ))?;
        Ok(Output::from_ptr(ptr))
    }

    pub fn send(mut self) -> Result<(), TritonError> {
        let send_flags =
            triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL;

        let result = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            self.ptr,
            send_flags,
            ptr::null_mut(),
        ));

        if result.is_ok() {
            // Triton now owns the response object.
            self.sent = true;
            self.responded.set(true);
        }

        result
    }
}

impl Drop for Response {
    fn drop(&mut self) {
        if !self.ptr.is_null() && !self.sent {
            unsafe {
                triton_ng_sys::TRITONBACKEND_ResponseDelete(self.ptr);
            }
        }
    }
}

pub struct Output {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Output,
}

impl Output {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Output) -> Self {
        Self { ptr }
    }

    pub fn write_string(&mut self, value: &str) -> Result<(), Error> {
        let encoded = encode_string(value)?;
        self.write_bytes(&encoded)?;
        Ok(())
    }

    /// Writes every element of a BYTES tensor.
    pub fn write_strings<S: AsRef<str>>(&mut self, values: &[S]) -> Result<(), Error> {
        let mut encoded = Vec::new();
        for value in values {
            encoded.extend(encode_string(value.as_ref())?);
        }
        self.write_bytes(&encoded)?;
        Ok(())
    }

    pub fn write_bytes(&mut self, data: &[u8]) -> Result<(), TritonError> {
        let buffer_byte_size = data.len() as u64;
        let mut memory_type: triton_ng_sys::TRITONSERVER_MemoryType =
            triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
        let mut memory_type_id = 0;

        if data.is_empty() {
            return Ok(());
        }

        let buffer = triton_call!(triton_ng_sys::TRITONBACKEND_OutputBuffer(
            self.ptr,
            &mut _,
            buffer_byte_size,
            &mut memory_type,
            &mut memory_type_id,
        ))?;

        if memory_type == triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU {
            return Err(TritonError::from_message(
                "output buffer was allocated in GPU memory, only host memory is supported",
            ));
        }

        let mem: &mut [u8] =
            unsafe { slice::from_raw_parts_mut(buffer as *mut u8, buffer_byte_size as usize) };

        mem.copy_from_slice(data);

        Ok(())
    }

    pub fn write_fp32_vec(&mut self, data: &[f32]) -> Result<(), TritonError> {
        let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        self.write_bytes(&bytes)
    }

    pub fn write_fp64_vec(&mut self, data: &[f64]) -> Result<(), TritonError> {
        let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        self.write_bytes(&bytes)
    }

    pub fn write_i32_vec(&mut self, data: &[i32]) -> Result<(), TritonError> {
        let bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();

        self.write_bytes(&bytes)
    }

    pub fn write_i64_vec(&mut self, data: &[i64]) -> Result<(), TritonError> {
        let bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();

        self.write_bytes(&bytes)
    }

    pub fn write_bool_vec(&mut self, data: &[bool]) -> Result<(), TritonError> {
        let bytes: Vec<u8> = data.iter().map(|&v| v as u8).collect();

        self.write_bytes(&bytes)
    }

    pub fn write_u64(&mut self, value: u64) -> Result<(), TritonError> {
        let bytes = value.to_le_bytes();
        self.write_bytes(&bytes)
    }
}
