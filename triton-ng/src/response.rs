use crate::Error;
use crate::error::TritonError;
use crate::request::Request;
use crate::utils::{cstring_from_str, encode_string};
use std::ffi::c_void;
use std::ptr;
use std::slice;

pub struct Response {
    ptr: *mut triton_sys::TRITONBACKEND_Response,
    /// Set to true after a successful `send()` so Drop doesn't double-delete.
    /// `TRITONBACKEND_ResponseSend` transfers ownership to Triton.
    sent: bool,
}

impl Response {
    pub fn new(request: &Request) -> Result<Self, TritonError> {
        let mut response: *mut triton_sys::TRITONBACKEND_Response = ptr::null_mut();

        ffi_call!(triton_sys::TRITONBACKEND_ResponseNew(
            &mut response,
            request.as_ptr()
        ))?;

        ensure_ptr!(response)?;

        Ok(Self {
            ptr: response,
            sent: false,
        })
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONBACKEND_Response {
        self.ptr
    }

    pub fn create_output(
        &mut self,
        name: &str,
        datatype: triton_sys::TRITONSERVER_DataType,
        shape: &[i64],
    ) -> Result<Output, TritonError> {
        let mut output: *mut triton_sys::TRITONBACKEND_Output = ptr::null_mut();
        let name_cstr = cstring_from_str(name)?;

        ffi_call!(triton_sys::TRITONBACKEND_ResponseOutput(
            self.ptr,
            &mut output,
            name_cstr.as_ptr(),
            datatype,
            shape.as_ptr(),
            shape.len() as u32,
        ))?;

        ensure_ptr!(output)?;

        Ok(Output::from_ptr(output))
    }

    pub fn send(mut self) -> Result<(), TritonError> {
        let send_flags =
            triton_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL;

        let result = ffi_call!(triton_sys::TRITONBACKEND_ResponseSend(
            self.ptr,
            send_flags,
            ptr::null_mut()
        ));

        if result.is_ok() {
            // Triton now owns the response object.
            self.sent = true;
        }

        result
    }
}

impl Drop for Response {
    fn drop(&mut self) {
        if !self.ptr.is_null() && !self.sent {
            unsafe {
                triton_sys::TRITONBACKEND_ResponseDelete(self.ptr);
            }
        }
    }
}

pub struct Output {
    ptr: *mut triton_sys::TRITONBACKEND_Output,
}

impl Output {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Output) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONBACKEND_Output {
        self.ptr
    }

    pub fn write_string(&mut self, value: &str) -> Result<(), Error> {
        let encoded = encode_string(value)?;
        self.write_bytes(&encoded)?;
        Ok(())
    }

    pub fn write_bytes(&mut self, data: &[u8]) -> Result<(), TritonError> {
        let mut buffer: *mut c_void = ptr::null_mut();
        let buffer_byte_size = data.len() as u64;
        let mut memory_type: triton_sys::TRITONSERVER_MemoryType = 0;
        let mut memory_type_id = 0;

        ffi_call!(triton_sys::TRITONBACKEND_OutputBuffer(
            self.ptr,
            &mut buffer,
            buffer_byte_size,
            &mut memory_type,
            &mut memory_type_id,
        ))?;

        ensure_ptr!(buffer as *mut u8)?;

        let mem: &mut [u8] =
            unsafe { slice::from_raw_parts_mut(buffer as *mut u8, buffer_byte_size as usize) };

        mem.copy_from_slice(data);

        Ok(())
    }

    pub fn write_fp32_vec(&mut self, data: &[f32]) -> Result<(), TritonError> {
        let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        self.write_bytes(&bytes)
    }

    pub fn write_u64(&mut self, value: u64) -> Result<(), TritonError> {
        let bytes = value.to_le_bytes();
        self.write_bytes(&bytes)
    }
}
