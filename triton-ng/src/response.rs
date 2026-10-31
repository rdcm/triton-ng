use crate::Error;
use crate::error::TritonError;
use crate::request::Request;
use crate::types::DataType;
use crate::utils::{count_strings, cstring_from_str, encode_string};
use std::ffi::c_void;
use std::ptr;
use std::slice;
use triton_ng_macros::triton_call;

/// Final response of a [`Request`].
///
/// The response takes the request over and keeps it until it is sent, so the
/// request gets no other response. A response dropped without being sent, or
/// whose sending fails, is deleted and its request gets an error response.
///
/// # Examples
///
/// ```ignore
/// let scores = request.get_input("SCORES")?.as_fp32_vec()?;
///
/// let mut response = Response::new(request)?;
/// response
///     .create_output("SCORES", DataType::Fp32, &[1, scores.len() as i64])?
///     .write_fp32_vec(&scores)?;
/// response.send()?;
/// ```
pub struct Response {
    /// Null once the response was handed over to `TRITONBACKEND_ResponseSend`,
    /// which takes ownership whatever it returns.
    ptr: *mut triton_ng_sys::TRITONBACKEND_Response,
    /// Dropped after the response, see `Drop`: an unsent response must be
    /// deleted before the request gets its error response.
    request: Request,
    /// Outputs created but not written yet. Triton frontends read every
    /// output of a sent response, so one without a buffer crashes the server.
    pending_outputs: usize,
}

// SAFETY: see `Request`. Outputs are created through `&mut self`, so they are
// never added from several threads at once, as tritonbackend.h requires.
unsafe impl Send for Response {}

impl Response {
    /// Creates the response of `request`. On error the request is dropped and
    /// gets an error response.
    pub fn new(request: Request) -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(
            &mut _,
            request.as_ptr()
        ))?;
        Ok(Self {
            ptr,
            request,
            pending_outputs: 0,
        })
    }

    /// The request this response answers, e.g. to read its inputs.
    pub fn request(&self) -> &Request {
        &self.request
    }

    /// Creates an output tensor. It must be written exactly once before
    /// [`Response::send`].
    pub fn create_output(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
    ) -> Result<Output<'_>, TritonError> {
        if datatype == DataType::Invalid {
            return Err(TritonError::from_message(format!(
                "output '{name}' has invalid datatype"
            )));
        }
        let elements = element_count(shape)
            .map_err(|e| TritonError::from_message(format!("output '{name}': {e}")))?;

        let name_cstr = cstring_from_str(name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseOutput(
            self.ptr,
            &mut _,
            name_cstr.as_ptr(),
            datatype.to_sys(),
            shape.as_ptr(),
            shape.len() as u32,
        ))?;

        self.pending_outputs += 1;

        Ok(Output {
            ptr,
            name: name.to_string(),
            datatype,
            elements,
            written: false,
            pending_outputs: &mut self.pending_outputs,
        })
    }

    /// Sends the response as the final one for its request.
    pub fn send(mut self) -> Result<(), TritonError> {
        if self.pending_outputs > 0 {
            return Err(TritonError::from_message(format!(
                "{} output(s) of the response were created but not written",
                self.pending_outputs
            )));
        }

        let response = std::mem::replace(&mut self.ptr, ptr::null_mut());

        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            ptr::null_mut(),
        ))?;

        self.request.mark_responded();

        Ok(())
    }
}

impl Drop for Response {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONBACKEND_ResponseDelete(self.ptr);
            }
        }
    }
}

/// Number of elements of a tensor with `shape`.
pub(crate) fn element_count(shape: &[i64]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |count, &dim| {
        let dim = usize::try_from(dim)
            .map_err(|_| format!("invalid dimension {dim} in shape {shape:?}"))?;
        count
            .checked_mul(dim)
            .ok_or_else(|| format!("shape {shape:?} is too large"))
    })
}

/// Output tensor of a [`Response`], borrowed from it so that it cannot be used
/// after the response is sent.
pub struct Output<'a> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Output,
    name: String,
    datatype: DataType,
    elements: usize,
    written: bool,
    pending_outputs: &'a mut usize,
}

impl Output<'_> {
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

#[cfg(test)]
mod tests {
    use super::element_count;

    #[test]
    fn element_count_multiplies_dimensions() {
        assert_eq!(element_count(&[2, 3, 4]), Ok(24));
        assert_eq!(element_count(&[1, 0, 2]), Ok(0));
        assert_eq!(element_count(&[]), Ok(1));
    }

    #[test]
    fn element_count_rejects_negative_dimensions() {
        assert!(element_count(&[1, -1]).is_err());
    }

    #[test]
    fn element_count_rejects_overflow() {
        assert!(element_count(&[i64::MAX, i64::MAX]).is_err());
    }
}
