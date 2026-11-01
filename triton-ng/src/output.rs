use crate::utils::{count_strings, encode_string};
use crate::{DataType, Error, TritonError};
use std::ffi::c_void;
use std::ptr;
use std::slice;
use triton_ng_macros::triton_call;

/// Output tensor of a [`crate::Response`].
///
/// Borrows the `TRITONBACKEND_Output` from the response: Triton frees it
/// together with the response, so there is nothing to drop, and the lifetime
/// keeps it from being used after the response is sent. Neither `Send` nor
/// `Sync`.
pub struct Output<'a> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Output,
    name: String,
    datatype: DataType,
    elements: usize,
    written: bool,
    pending_outputs: &'a mut usize,
}

impl<'a> Output<'a> {
    pub(crate) fn new(
        ptr: *mut triton_ng_sys::TRITONBACKEND_Output,
        name: String,
        datatype: DataType,
        elements: usize,
        pending_outputs: &'a mut usize,
    ) -> Self {
        Self {
            ptr,
            name,
            datatype,
            elements,
            written: false,
            pending_outputs,
        }
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

    /// Writes the raw tensor content: little-endian elements, or serialized
    /// strings for a BYTES tensor. The size must match the shape.
    pub fn write_bytes(&mut self, data: &[u8]) -> Result<(), TritonError> {
        if self.written {
            return Err(self.error("is already written"));
        }

        self.check_size(data)?;

        let buffer_byte_size = data.len() as u64;
        let mut memory_type: triton_ng_sys::TRITONSERVER_MemoryType =
            triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
        let mut memory_type_id = 0;
        let mut buffer: *mut c_void = ptr::null_mut();

        // An empty tensor still needs a buffer: the HTTP frontend crashes on an
        // output that never got one.
        triton_call!(triton_ng_sys::TRITONBACKEND_OutputBuffer(
            self.ptr,
            &mut buffer,
            buffer_byte_size,
            &mut memory_type,
            &mut memory_type_id,
        ))?;

        if !data.is_empty() {
            if buffer.is_null() {
                return Err(self.error("got a null buffer"));
            }

            if memory_type != triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU
                && memory_type
                    != triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED
            {
                return Err(
                    self.error("buffer is not in host memory, only host memory is supported")
                );
            }

            let mem: &mut [u8] =
                unsafe { slice::from_raw_parts_mut(buffer as *mut u8, data.len()) };
            mem.copy_from_slice(data);
        }

        self.written = true;
        *self.pending_outputs = self.pending_outputs.saturating_sub(1);

        Ok(())
    }

    pub fn write_fp32_vec(&mut self, data: &[f32]) -> Result<(), TritonError> {
        self.write_le(DataType::Fp32, data, |v| v.to_le_bytes())
    }

    pub fn write_fp64_vec(&mut self, data: &[f64]) -> Result<(), TritonError> {
        self.write_le(DataType::Fp64, data, |v| v.to_le_bytes())
    }

    pub fn write_i32_vec(&mut self, data: &[i32]) -> Result<(), TritonError> {
        self.write_le(DataType::Int32, data, |v| v.to_le_bytes())
    }

    pub fn write_i64_vec(&mut self, data: &[i64]) -> Result<(), TritonError> {
        self.write_le(DataType::Int64, data, |v| v.to_le_bytes())
    }

    pub fn write_bool_vec(&mut self, data: &[bool]) -> Result<(), TritonError> {
        self.write_le(DataType::Bool, data, |v| [u8::from(v)])
    }

    pub fn write_u64(&mut self, value: u64) -> Result<(), TritonError> {
        self.write_le(DataType::Uint64, &[value], |v| v.to_le_bytes())
    }

    fn write_le<T: Copy, const N: usize>(
        &mut self,
        datatype: DataType,
        data: &[T],
        to_le: fn(T) -> [u8; N],
    ) -> Result<(), TritonError> {
        if self.datatype != datatype {
            return Err(self.error(&format!(
                "has datatype {:?}, cannot write {datatype:?}",
                self.datatype
            )));
        }

        let bytes: Vec<u8> = data.iter().flat_map(|&v| to_le(v)).collect();
        self.write_bytes(&bytes)
    }

    fn check_size(&self, data: &[u8]) -> Result<(), TritonError> {
        match self.datatype.element_size() {
            Some(size) => {
                let expected = self.elements.checked_mul(size);
                if expected != Some(data.len()) {
                    return Err(self.error(&format!(
                        "expects {} elements of {size} bytes, got {} bytes",
                        self.elements,
                        data.len()
                    )));
                }
            }
            None => {
                let strings = count_strings(data)
                    .map_err(|e| self.error(&format!("has malformed BYTES content: {e}")))?;
                if strings != self.elements {
                    return Err(
                        self.error(&format!("expects {} strings, got {strings}", self.elements))
                    );
                }
            }
        }
        Ok(())
    }

    fn error(&self, message: &str) -> TritonError {
        TritonError::from_message(format!("output '{}' {message}", self.name))
    }
}
