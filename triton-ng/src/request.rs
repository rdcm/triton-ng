use crate::error::{Error, TritonError};
use crate::types::DataType;
use crate::utils::{cstr_to_string, cstring_from_str, decode_string};
use libc::{c_char, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::slice;
use triton_ng_macros::triton_call;

/// Inference request passed to [`crate::Backend::model_instance_execute`].
///
/// The backend owns the request and can answer it from any thread of the
/// instance runtime. The
/// request gets exactly one final response, sent by [`crate::Response::send`],
/// [`Request::respond_error`] or [`Request::respond_cancelled`], all of which
/// consume it. A request dropped without a response gets an error response.
/// Either way the request is released to Triton when it is dropped.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context<State>, requests: Vec<Request>) -> Result<(), Error> {
///     for request in requests {
///         let text = match request.get_input("TEXT").and_then(|input| input.as_string()) {
///             Ok(text) => text,
///             Err(e) => {
///                 request.respond_error(&e.to_string())?;
///                 continue;
///             }
///         };
///
///         let mut response = Response::new(request)?;
///         response
///             .create_output("TEXT", DataType::Bytes, &[1])?
///             .write_string(&text.to_uppercase())?;
///         response.send()?;
///     }
///     Ok(())
/// }
/// ```
pub struct Request {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
    responded: bool,
}

// SAFETY: once ModelInstanceExecute succeeds, Triton hands the request over to
// the backend, which may respond to and release it from any thread, as long as
// one request is not used from several threads at once (tritonbackend.h,
// TRITONBACKEND_Response). `Request` is not `Sync`, so it never is.
unsafe impl Send for Request {}

impl Request {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Request) -> Self {
        Self {
            ptr,
            responded: false,
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONBACKEND_Request {
        self.ptr
    }

    pub(crate) fn mark_responded(&mut self) {
        self.responded = true;
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

    pub fn get_input(&self, name: &str) -> Result<Input<'_>, TritonError> {
        let name = cstring_from_str(name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_RequestInput(
            self.ptr,
            name.as_ptr(),
            &mut _
        ))?;
        Ok(Input::from_ptr(ptr))
    }

    /// Returns `true` if the client cancelled the request.
    ///
    /// A cancelled request should stop being processed and be answered with
    /// [`Request::respond_cancelled`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// for request in requests {
    ///     if request.is_cancelled()? {
    ///         request.respond_cancelled()?;
    ///         continue;
    ///     }
    ///     // ... process the request ...
    /// }
    /// ```
    pub fn is_cancelled(&self) -> Result<bool, TritonError> {
        let mut cancelled = false;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestIsCancelled(
            self.ptr,
            &mut cancelled
        ))?;
        Ok(cancelled)
    }

    /// Sends a final response carrying `message` as an internal error.
    ///
    /// Use it to fail a single request of a batch without failing the others.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Err(e) = validate(&request) {
    ///     request.respond_error(&e.to_string())?;
    /// }
    /// ```
    pub fn respond_error(mut self, message: &str) -> Result<(), TritonError> {
        self.respond_with(TritonError::internal(message))
    }

    /// Sends a final response with the CANCELLED status, see
    /// [`Request::is_cancelled`].
    pub fn respond_cancelled(mut self) -> Result<(), TritonError> {
        self.respond_with(TritonError::cancelled("request was cancelled"))
    }

    fn respond_with(&mut self, error: TritonError) -> Result<(), TritonError> {
        if self.responded {
            return Err(TritonError::from_message(
                "the request already has a final response",
            ));
        }

        // The caller keeps ownership of the error object passed to ResponseSend,
        // it is deleted on drop.
        if error.as_ptr().is_null() {
            return Err(TritonError::from_message("failed to create error object"));
        }

        let response = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(&mut _, self.ptr))?;

        // ResponseSend takes ownership of the response whatever it returns.
        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            error.as_ptr(),
        ))?;

        self.responded = true;
        Ok(())
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.responded
            && let Err(e) = self.respond_with(TritonError::internal(
                "backend did not produce a response for the request",
            ))
        {
            crate::log::error(&format!("failed to send error response: {e}"));
        }

        // On error the request stays with the backend, nothing else can be done with it.
        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_RequestRelease(
            self.ptr,
            triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL,
        )) {
            crate::log::error(&format!("failed to release request: {e}"));
        }
    }
}

/// Input tensor of a [`Request`], borrowed from it: Triton frees the tensor
/// together with the request.
pub struct Input<'a> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Input,
    request: PhantomData<&'a Request>,
}

impl Input<'_> {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Input) -> Self {
        Self {
            ptr,
            request: PhantomData,
        }
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

            if buffer_byte_size == 0 {
                continue;
            }

            if buffer.is_null() {
                return Err(format!("input '{}' buffer {index} is null", properties.name).into());
            }

            if memory_type != triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU
                && memory_type
                    != triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED
            {
                return Err(format!(
                    "input '{}' buffer {index} is not in host memory, only host memory is supported",
                    properties.name
                )
                .into());
            }

            let mem: &[u8] =
                unsafe { slice::from_raw_parts(buffer as *const u8, buffer_byte_size as usize) };
            data.extend_from_slice(mem);
        }

        if data.len() as u64 != properties.byte_size {
            return Err(format!(
                "input '{}' has {} bytes in its buffers, expected {}",
                properties.name,
                data.len(),
                properties.byte_size
            )
            .into());
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
