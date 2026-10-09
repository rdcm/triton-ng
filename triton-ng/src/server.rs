use crate::inference_request::InferenceRequest;
use crate::inference_response::InferenceResponse;
use crate::request::decode_le;
use crate::response_allocator::ResponseAllocator;
use crate::types::DataType;
use crate::utils::cstring_from_str;
use crate::{Error, TritonError};
use crossbeam::channel::{Sender, bounded};
use std::ffi::c_void;
use std::ptr;
use triton_ng_macros::triton_call;

pub struct OutputTensor {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub datatype: crate::types::DataType,
}

impl OutputTensor {
    pub fn as_fp32_vec(&self) -> Result<Vec<f32>, Error> {
        self.expect_datatype(DataType::Fp32)?;
        decode_le(&self.data, f32::from_le_bytes)
    }

    pub fn as_i32_vec(&self) -> Result<Vec<i32>, Error> {
        self.expect_datatype(DataType::Int32)?;
        decode_le(&self.data, i32::from_le_bytes)
    }

    pub fn as_i64_vec(&self) -> Result<Vec<i64>, Error> {
        self.expect_datatype(DataType::Int64)?;
        decode_le(&self.data, i64::from_le_bytes)
    }

    /// Integer tensor widened to `i64`, accepts both INT32 and INT64.
    pub fn as_index_vec(&self) -> Result<Vec<i64>, Error> {
        match self.datatype {
            DataType::Int32 => Ok(decode_le(&self.data, i32::from_le_bytes)?
                .into_iter()
                .map(i64::from)
                .collect()),
            _ => self.as_i64_vec(),
        }
    }

    fn expect_datatype(&self, expected: DataType) -> Result<(), Error> {
        if self.datatype != expected {
            return Err(format!(
                "output '{}' has datatype {:?}, expected {:?}",
                self.name, self.datatype, expected
            )
            .into());
        }
        Ok(())
    }
}

pub struct InferenceResult {
    pub outputs: Vec<OutputTensor>,
    pub error: Option<String>,
}

impl InferenceResult {
    /// Returns the output tensor named `name`.
    pub fn output(&self, name: &str) -> Result<&OutputTensor, Error> {
        self.outputs
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| format!("missing output '{name}'").into())
    }
}

pub struct InferenceContext {
    tx: Sender<InferenceResult>,
    allocator: ResponseAllocator,
    pending_outputs: Vec<OutputTensor>,
    error: Option<String>,
}

pub struct Server {
    ptr: *mut triton_ng_sys::TRITONSERVER_Server,
}

impl Server {
    pub(crate) fn from_ptr(
        ptr: *mut triton_ng_sys::TRITONSERVER_Server,
    ) -> Result<Self, TritonError> {
        if ptr.is_null() {
            return Err(TritonError::from_message(
                "TRITONBACKEND_ModelServer returned null",
            ));
        }
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONSERVER_Server {
        self.ptr
    }

    pub fn api_version(&self) -> Result<(u32, u32), TritonError> {
        let mut major: u32 = 0;
        let mut minor: u32 = 0;

        triton_call!(triton_ng_sys::TRITONSERVER_ApiVersion(
            &mut major, &mut minor
        ))?;

        Ok((major, minor))
    }

    pub fn is_ready(&self) -> Result<bool, TritonError> {
        let mut ready = false;

        triton_call!(triton_ng_sys::TRITONSERVER_ServerIsReady(
            self.ptr, &mut ready
        ))?;

        Ok(ready)
    }

    pub fn is_model_ready(&self, model_name: &str, version: i64) -> Result<bool, TritonError> {
        let model_name_cstr = cstring_from_str(model_name)?;
        let mut ready = false;

        triton_call!(triton_ng_sys::TRITONSERVER_ServerModelIsReady(
            self.ptr,
            model_name_cstr.as_ptr(),
            version,
            &mut ready
        ))?;

        Ok(ready)
    }

    pub fn model_metadata(&self, model_name: &str, version: i64) -> Result<String, TritonError> {
        let model_name_cstr = cstring_from_str(model_name)?;
        let metadata_ptr = triton_call!(triton_ng_sys::TRITONSERVER_ServerModelMetadata(
            self.ptr,
            model_name_cstr.as_ptr(),
            version,
            &mut _,
        ))?;

        let mut base: *const i8 = std::ptr::null();
        let mut byte_size: usize = 0;

        triton_call!(triton_ng_sys::TRITONSERVER_MessageSerializeToJson(
            metadata_ptr,
            &mut base,
            &mut byte_size
        ))?;

        let json_bytes = unsafe { std::slice::from_raw_parts(base as *const u8, byte_size) };
        let result = String::from_utf8_lossy(json_bytes).to_string();

        unsafe {
            triton_ng_sys::TRITONSERVER_MessageDelete(metadata_ptr);
        }

        Ok(result)
    }

    /// Run inference synchronously.
    ///
    /// Takes ownership of `request`. The request is freed by the
    /// `inference_request_release` callback, which Triton calls only *after* the
    /// backend has fully released all references to it. This prevents the race
    /// where `InferenceRequest::Drop` frees the object while Triton's backend
    /// thread still holds a pointer (which causes a SIGSEGV in
    /// `TRITONBACKEND_RequestRelease`).
    ///
    /// The input buffers owned by the request are freed by the same callback:
    /// Triton may read them until the request is released, which can happen
    /// after the final response has already been delivered.
    pub fn infer_async(&self, request: InferenceRequest) -> Result<InferenceResult, TritonError> {
        let (tx, rx) = bounded(1);
        let allocator = ResponseAllocator::new()?;

        let (raw, buffers) = request.into_raw_parts();
        let buffers_ptr = Box::into_raw(Box::new(buffers)) as *mut c_void;

        // Register the release callback before handing the request to Triton.
        if let Err(e) = triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetReleaseCallback(
                raw,
                Some(inference_request_release),
                buffers_ptr,
            )
        ) {
            // Triton never saw this request, release everything ourselves.
            unsafe { inference_request_release(raw, RELEASE_ALL, buffers_ptr) };
            return Err(e);
        }

        let context = Box::new(InferenceContext {
            tx,
            allocator,
            pending_outputs: Vec::new(),
            error: None,
        });
        let context_ptr = Box::into_raw(context) as *mut c_void;

        if let Err(e) = triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetResponseCallback(
                raw,
                (*context_ptr.cast::<InferenceContext>()).allocator.as_ptr(),
                ptr::null_mut(),
                Some(inference_response_complete),
                context_ptr,
            )
        ) {
            unsafe {
                drop(Box::from_raw(context_ptr as *mut InferenceContext));
                inference_request_release(raw, RELEASE_ALL, buffers_ptr);
            }
            return Err(e);
        }

        // Transfer ownership to Triton. The release callback now owns deletion.
        if let Err(e) = triton_call!(triton_ng_sys::TRITONSERVER_ServerInferAsync(
            self.ptr,
            raw,
            ptr::null_mut(),
        )) {
            // Triton guarantees it will NOT call neither the release nor the
            // response callback on failure, so we must clean up manually.
            unsafe {
                drop(Box::from_raw(context_ptr as *mut InferenceContext));
                inference_request_release(raw, RELEASE_ALL, buffers_ptr);
            }
            return Err(e);
        }

        let result = rx
            .recv()
            .map_err(|_| TritonError::from_message("inference channel closed"))?;

        if let Some(error) = result.error {
            return Err(TritonError::from_message(&error));
        }

        Ok(result)
    }
}

/// Called by Triton after the backend has fully released the inference request.
/// At this point it is safe to delete the request object.
///
/// `userp` is the boxed `Vec<Vec<u8>>` of input buffers referenced by the
/// request, they are freed only after the request itself is deleted.
///
/// The header says RELEASE_ALL should always be set today, but recommends
/// checking explicitly in case future versions add new flags.
unsafe extern "C" fn inference_request_release(
    request: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
    flags: u32,
    userp: *mut std::os::raw::c_void,
) {
    if flags & RELEASE_ALL == 0 {
        return;
    }

    if !request.is_null() {
        unsafe { triton_ng_sys::TRITONSERVER_InferenceRequestDelete(request) };
    }

    if !userp.is_null() {
        unsafe { drop(Box::from_raw(userp as *mut Vec<Vec<u8>>)) };
    }
}

const RELEASE_ALL: u32 =
    triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL;

/// Called by Triton for each response (and once more with a null response when
/// RESPONSE_COMPLETE_FINAL is set, to signal end-of-stream).
///
/// For one-to-one models this is called exactly once with FINAL set.
/// For decoupled/streaming models it may be called multiple times; we
/// accumulate outputs across all non-final callbacks and only send the
/// collected result (and drop the context) when FINAL arrives.
unsafe extern "C" fn inference_response_complete(
    response_ptr: *mut triton_ng_sys::TRITONSERVER_InferenceResponse,
    flags: u32,
    userp: *mut std::os::raw::c_void,
) {
    use triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL as FINAL;
    let is_final = flags & FINAL != 0;

    // Borrow context without taking ownership — we only own it on the final call.
    let context = unsafe { &mut *(userp as *mut InferenceContext) };

    if !response_ptr.is_null() {
        match InferenceResponse::from_ptr(response_ptr) {
            Err(e) => {
                context.error.get_or_insert_with(|| e.to_string());
            }
            Ok(response) => {
                if let Some(err) = response.error() {
                    context.error.get_or_insert_with(|| err.to_string());
                } else {
                    match response.outputs() {
                        Ok(outputs) => context.pending_outputs.extend(outputs),
                        Err(e) => {
                            context.error.get_or_insert_with(|| e.to_string());
                        }
                    }
                }
            }
        }
    }

    if is_final {
        let context = unsafe { Box::from_raw(userp as *mut InferenceContext) };
        let result = if let Some(error) = context.error {
            InferenceResult {
                outputs: vec![],
                error: Some(error),
            }
        } else {
            InferenceResult {
                outputs: context.pending_outputs,
                error: None,
            }
        };
        let _ = context.tx.send(result);
    }
}
