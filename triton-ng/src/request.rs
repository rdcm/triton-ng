use crate::error::{Error, TritonError};
use crate::types::DataType;
use crate::utils::{cstr_to_string, cstring_from_str, decode_string};
use libc::{c_char, c_void};
use std::cell::Cell;
use std::ptr;
use std::rc::Rc;
use std::slice;
use triton_ng_macros::triton_call;

pub struct Request {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
    /// Set once a final response has been sent for this request, so that
    /// `declare_backend!` knows which requests still need an error response.
    responded: Rc<Cell<bool>>,
}

impl Request {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Request) -> Self {
        Self {
            ptr,
            responded: Rc::new(Cell::new(false)),
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONBACKEND_Request {
        self.ptr
    }

    pub(crate) fn responded_flag(&self) -> Rc<Cell<bool>> {
        Rc::clone(&self.responded)
    }

    /// Returns `true` if a final response (successful or not) was already sent.
    pub fn is_responded(&self) -> bool {
        self.responded.get()
    }

    pub fn input_names(&self) -> Result<Vec<String>, TritonError> {
        let mut count = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestInputCount(
            self.ptr, &mut count
        ))?;
        (0..count)
            .map(|i| {
                let mut name: *const c_char = std::ptr::null();
                triton_call!(triton_ng_sys::TRITONBACKEND_RequestInputName(
                    self.ptr, i, &mut name,
                ))?;
                Ok(unsafe { cstr_to_string(name) })
            })
            .collect()
    }

    pub fn output_names(&self) -> Result<Vec<String>, TritonError> {
        let mut count = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestOutputCount(
            self.ptr, &mut count
        ))?;
        (0..count)
            .map(|i| {
                let mut name: *const c_char = std::ptr::null();
                triton_call!(triton_ng_sys::TRITONBACKEND_RequestOutputName(
                    self.ptr, i, &mut name,
                ))?;
                Ok(unsafe { cstr_to_string(name) })
            })
            .collect()
    }

    pub fn get_input(&self, name: &str) -> Result<Input, TritonError> {
        let name = cstring_from_str(name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_RequestInput(
            self.ptr,
            name.as_ptr(),
            &mut _
        ))?;
        Ok(Input::from_ptr(ptr))
    }

    /// Sends a final response carrying `message` as an internal error.
    ///
    /// Use it to fail a single request of a batch without failing the others.
    pub fn respond_error(&self, message: &str) -> Result<(), TritonError> {
        let response = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(&mut _, self.ptr))?;

        let message = cstring_from_str(&message.replace('\0', "?"))?;
        let error = unsafe {
            triton_ng_sys::TRITONSERVER_ErrorNew(
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                message.as_ptr(),
            )
        };

        let result = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            error,
        ));

        // The caller keeps ownership of the error object passed to ResponseSend.
        unsafe { triton_ng_sys::TRITONSERVER_ErrorDelete(error) };

        match result {
            Ok(()) => {
                self.responded.set(true);
                Ok(())
            }
            Err(e) => {
                // Ownership of the response was not transferred to Triton.
                unsafe { triton_ng_sys::TRITONBACKEND_ResponseDelete(response) };
                Err(e)
            }
        }
    }

    /// Returns ownership of the request to Triton. Must be called exactly once
    /// per request after a successful `ModelInstanceExecute`.
    pub(crate) fn release(&self) -> Result<(), TritonError> {
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestRelease(
            self.ptr,
            triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL,
        ))
    }
}

pub struct Input {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Input,
}

impl Input {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Input) -> Self {
        Self { ptr }
    }

    /// Copies the tensor into a contiguous host buffer.
    ///
    /// Triton may split a tensor into several buffers (`buffer_count`),
    /// all of them are concatenated in order.
    fn buffer(&self) -> Result<Vec<u8>, Error> {
        let properties = self.properties()?;
        let mut data = Vec::with_capacity(properties.byte_size as usize);

        for index in 0..properties.buffer_count {
            let mut buffer: *const c_void = ptr::null();
            let mut memory_type: triton_ng_sys::TRITONSERVER_MemoryType =
                triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
            let mut memory_type_id = 0;
            let mut buffer_byte_size = 0;
            triton_call!(triton_ng_sys::TRITONBACKEND_InputBuffer(
                self.ptr,
                index,
                &mut buffer,
                &mut buffer_byte_size,
                &mut memory_type,
                &mut memory_type_id,
            ))?;

            if memory_type == triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU {
                return Err(format!(
                    "input '{}' buffer {index} is in GPU memory, only host memory is supported",
                    properties.name
                )
                .into());
            }

            if buffer_byte_size > 0 {
                let mem: &[u8] = unsafe {
                    slice::from_raw_parts(buffer as *const u8, buffer_byte_size as usize)
                };
                data.extend_from_slice(mem);
            }
        }

        Ok(data)
    }

    /// Raw little-endian tensor bytes.
    pub fn as_bytes(&self) -> Result<Vec<u8>, Error> {
        self.buffer()
    }

    /// First element of a BYTES tensor decoded as UTF-8 (lossy).
    pub fn as_string(&self) -> Result<String, Error> {
        let buffer = self.buffer()?;
        let strings = decode_string(&buffer)?;
        strings
            .into_iter()
            .next()
            .ok_or_else(|| "empty BYTES tensor".into())
    }

    /// All elements of a BYTES tensor decoded as UTF-8 (lossy).
    pub fn as_strings(&self) -> Result<Vec<String>, Error> {
        Ok(decode_string(&self.buffer()?)?)
    }

    pub fn as_u64(&self) -> Result<u64, Error> {
        let buffer = self.buffer()?;

        let bytes: [u8; 8] = buffer
            .as_slice()
            .try_into()
            .map_err(|_| format!("expected 8 bytes for UINT64 scalar, got {}", buffer.len()))?;

        Ok(u64::from_le_bytes(bytes))
    }

    pub fn as_fp32_vec(&self) -> Result<Vec<f32>, Error> {
        decode_le(&self.buffer()?, f32::from_le_bytes)
    }

    pub fn as_i32_vec(&self) -> Result<Vec<i32>, Error> {
        decode_le(&self.buffer()?, i32::from_le_bytes)
    }

    pub fn as_i64_vec(&self) -> Result<Vec<i64>, Error> {
        decode_le(&self.buffer()?, i64::from_le_bytes)
    }

    pub fn properties(&self) -> Result<InputProperties, Error> {
        let mut name = std::ptr::null();
        let mut datatype = 0u32;
        let mut shape_ptr: *const i64 = std::ptr::null();
        let mut dims_count = 0u32;
        let mut byte_size = 0u64;
        let mut buffer_count = 0u32;

        triton_call!(triton_ng_sys::TRITONBACKEND_InputProperties(
            self.ptr,
            &mut name,
            &mut datatype,
            &mut shape_ptr,
            &mut dims_count,
            &mut byte_size,
            &mut buffer_count,
        ))?;

        let name = unsafe { cstr_to_string(name) };

        let shape = if shape_ptr.is_null() || dims_count == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(shape_ptr, dims_count as usize).to_vec() }
        };

        Ok(InputProperties {
            name,
            datatype: DataType::from_sys(datatype),
            shape,
            dims_count,
            byte_size,
            buffer_count,
        })
    }
}

/// Decodes a buffer of little-endian fixed-size elements.
pub(crate) fn decode_le<T, const N: usize>(
    buffer: &[u8],
    from_le: fn([u8; N]) -> T,
) -> Result<Vec<T>, Error> {
    if !buffer.len().is_multiple_of(N) {
        return Err(format!(
            "tensor byte size {} is not a multiple of element size {N}",
            buffer.len()
        )
        .into());
    }

    Ok(buffer
        .chunks_exact(N)
        .filter_map(|chunk| chunk.try_into().ok().map(from_le))
        .collect())
}

#[derive(Debug)]
pub struct InputProperties {
    pub name: String,
    pub datatype: DataType,
    pub shape: Vec<i64>,
    pub dims_count: u32,
    pub byte_size: u64,
    pub buffer_count: u32,
}
