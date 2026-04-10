use crate::TritonError;
use crate::inference_request::InferenceRequest;
use crate::inference_response::InferenceResponse;
use crate::response_allocator::ResponseAllocator;
use crate::utils::cstring_from_str;
use crossbeam::channel::{Sender, bounded};
use std::ffi::c_void;
use std::ptr;

pub struct OutputTensor {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub datatype: String,
}

pub struct InferenceResult {
    pub outputs: Vec<OutputTensor>,
    pub error: Option<String>,
}

pub struct InferenceContext {
    tx: Sender<InferenceResult>,
    allocator: ResponseAllocator,
}

pub struct Server {
    ptr: *mut triton_sys::TRITONSERVER_Server,
}

impl Server {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONSERVER_Server) -> Result<Self, TritonError> {
        ensure_ptr!(ptr)?;
        Ok(Self { ptr })
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONSERVER_Server {
        self.ptr
    }

    pub fn api_version(&self) -> Result<(u32, u32), TritonError> {
        let mut major: u32 = 0;
        let mut minor: u32 = 0;

        ffi_call!(triton_sys::TRITONSERVER_ApiVersion(&mut major, &mut minor))?;

        Ok((major, minor))
    }

    pub fn is_ready(&self) -> Result<bool, TritonError> {
        let mut ready = false;

        ffi_call!(triton_sys::TRITONSERVER_ServerIsReady(self.ptr, &mut ready))?;

        Ok(ready)
    }

    pub fn is_model_ready(&self, model_name: &str, version: i64) -> Result<bool, TritonError> {
        let model_name_cstr = cstring_from_str(model_name);
        let mut ready = false;

        ffi_call!(triton_sys::TRITONSERVER_ServerModelIsReady(
            self.ptr,
            model_name_cstr.as_ptr(),
            version,
            &mut ready
        ))?;

        Ok(ready)
    }

    pub fn model_metadata(&self, model_name: &str, version: i64) -> Result<String, TritonError> {
        let model_name_cstr = cstring_from_str(model_name);
        let mut metadata_ptr: *mut triton_sys::TRITONSERVER_Message = std::ptr::null_mut();

        ffi_call!(triton_sys::TRITONSERVER_ServerModelMetadata(
            self.ptr,
            model_name_cstr.as_ptr(),
            version,
            &mut metadata_ptr
        ))?;

        ensure_ptr!(metadata_ptr)?;

        let mut base: *const i8 = std::ptr::null();
        let mut byte_size: usize = 0;

        ffi_call!(triton_sys::TRITONSERVER_MessageSerializeToJson(
            metadata_ptr,
            &mut base,
            &mut byte_size
        ))?;

        let json_bytes = unsafe { std::slice::from_raw_parts(base as *const u8, byte_size) };
        let result = String::from_utf8_lossy(json_bytes).to_string();

        unsafe {
            triton_sys::TRITONSERVER_MessageDelete(metadata_ptr);
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
    pub fn infer_async(&self, request: InferenceRequest) -> Result<InferenceResult, TritonError> {
        let (tx, rx) = bounded(1);
        let allocator = ResponseAllocator::new()?;

        // Register the release callback before handing the request to Triton.
        // If this call fails, `request` is still owned by us and Drop will
        // call `TRITONSERVER_InferenceRequestDelete` normally.
        ffi_call!(triton_sys::TRITONSERVER_InferenceRequestSetReleaseCallback(
            request.as_ptr(),
            Some(inference_request_release),
            ptr::null_mut(),
        ))?;

        let context = Box::new(InferenceContext { tx, allocator });
        let context_ptr = Box::into_raw(context) as *mut c_void;

        if let Err(e) = ffi_call!(
            triton_sys::TRITONSERVER_InferenceRequestSetResponseCallback(
                request.as_ptr(),
                (*context_ptr.cast::<InferenceContext>()).allocator.as_ptr(),
                ptr::null_mut(),
                Some(inference_response_complete),
                context_ptr,
            )
        ) {
            unsafe { drop(Box::from_raw(context_ptr as *mut InferenceContext)) };
            return Err(e);
            // `request` drops here — Drop calls Delete (correct: Triton never saw
            // this request so the release callback will not fire).
        }

        // Transfer ownership to Triton. The release callback now owns deletion;
        // prevent Drop from running a second `InferenceRequestDelete`.
        let raw = request.as_ptr();
        std::mem::forget(request);

        if let Err(e) = ffi_call!(triton_sys::TRITONSERVER_ServerInferAsync(
            self.ptr,
            raw,
            ptr::null_mut(),
        )) {
            // Triton guarantees it will NOT call the release callback on failure,
            // so we must delete manually.
            unsafe { triton_sys::TRITONSERVER_InferenceRequestDelete(raw) };
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
unsafe extern "C" fn inference_request_release(
    request: *mut triton_sys::TRITONSERVER_InferenceRequest,
    _flags: u32,
    _userp: *mut std::os::raw::c_void,
) {
    if !request.is_null() {
        unsafe { triton_sys::TRITONSERVER_InferenceRequestDelete(request) };
    }
}

unsafe extern "C" fn inference_response_complete(
    response_ptr: *mut triton_sys::TRITONSERVER_InferenceResponse,
    _flags: u32,
    userp: *mut std::os::raw::c_void,
) {
    let context = unsafe { Box::from_raw(userp as *mut InferenceContext) };

    let response = match InferenceResponse::from_ptr(response_ptr) {
        Ok(r) => r,
        Err(e) => {
            let _ = context.tx.send(InferenceResult {
                outputs: vec![],
                error: Some(e.to_string()),
            });
            return;
        }
    };

    let result = if let Some(error) = response.error() {
        InferenceResult {
            outputs: vec![],
            error: Some(error.to_string()),
        }
    } else {
        match response.outputs() {
            Ok(outputs) => InferenceResult {
                outputs,
                error: None,
            },
            Err(e) => InferenceResult {
                outputs: vec![],
                error: Some(e.to_string()),
            },
        }
    };

    let _ = context.tx.send(result);
}
