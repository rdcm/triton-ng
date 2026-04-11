#[path = "backend.rs"]
pub mod backend;
#[path = "backend_handle.rs"]
pub mod backend_handle;
#[path = "error.rs"]
pub mod error;
#[path = "inference_request.rs"]
pub mod inference_request;
#[path = "inference_response.rs"]
pub mod inference_response;
#[path = "model.rs"]
pub mod model;
#[path = "model_instance.rs"]
pub mod model_instance;
#[path = "request.rs"]
pub mod request;
#[path = "response.rs"]
pub mod response;
#[path = "response_allocator.rs"]
pub mod response_allocator;
#[path = "server.rs"]
pub mod server;
#[path = "types.rs"]
pub mod types;
#[path = "utils.rs"]
pub(crate) mod utils;

pub use backend::*;
pub use backend_handle::*;
pub use error::*;
pub use inference_request::*;
pub use inference_response::*;
pub use model::*;
pub use model_instance::*;
pub use request::*;
pub use response::*;
pub use response_allocator::*;
pub use server::*;
pub use types::*;

/// Support module for `declare_backend!` macro expansion in external crates.
/// Not part of the public API.
#[doc(hidden)]
pub mod __macro_support {
    use std::ffi::c_void;

    /// Opaque error pointer returned by Triton backend entry points.
    /// Equivalent to `*const TRITONSERVER_Error` at the ABI level.
    pub type ErrorPtr = *const c_void;

    pub const NULL_ERROR: ErrorPtr = std::ptr::null::<c_void>();

    pub fn make_internal_error(msg: &str) -> ErrorPtr {
        // Replace null bytes: CString cannot contain them.
        let sanitized = msg.replace('\0', "?");
        // SAFETY: replace('\0') guarantees no null bytes remain.
        let c_str = unsafe { std::ffi::CString::from_vec_unchecked(sanitized.into_bytes()) };
        unsafe {
            triton_sys::TRITONSERVER_ErrorNew(
                triton_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                c_str.as_ptr(),
            ) as ErrorPtr
        }
    }

    /// Wraps a raw backend handle pointer. Called from `declare_backend!`.
    pub unsafe fn backend_handle(ptr: *mut c_void) -> crate::backend_handle::BackendHandle {
        crate::backend_handle::BackendHandle::from_ptr(
            ptr as *mut triton_sys::TRITONBACKEND_Backend,
        )
    }

    /// Wraps a raw model instance pointer. Called from `declare_backend!`.
    pub unsafe fn model_instance(ptr: *mut c_void) -> crate::model_instance::ModelInstance {
        crate::model_instance::ModelInstance::from_ptr(
            ptr as *mut triton_sys::TRITONBACKEND_ModelInstance,
        )
    }

    /// Extracts a `Model` and `Vec<Request>` from a `ModelInstanceExecute` call.
    /// Returns `Err(ErrorPtr)` if `TRITONBACKEND_ModelInstanceModel` fails.
    pub unsafe fn model_and_requests(
        instance: *mut c_void,
        requests_ptr: *const *mut c_void,
        request_count: u32,
    ) -> Result<(crate::model::Model, Vec<crate::request::Request>), ErrorPtr> {
        let mut model_ptr: *mut triton_sys::TRITONBACKEND_Model = std::ptr::null_mut();
        let err = unsafe {
            triton_sys::TRITONBACKEND_ModelInstanceModel(
                instance as *mut triton_sys::TRITONBACKEND_ModelInstance,
                &mut model_ptr,
            )
        };
        if !err.is_null() {
            return Err(err as ErrorPtr);
        }

        let model = crate::model::Model::from_ptr(model_ptr);

        let raw = unsafe { std::slice::from_raw_parts(requests_ptr, request_count as usize) };
        let requests = raw
            .iter()
            .map(|&r| {
                crate::request::Request::from_ptr(r as *mut triton_sys::TRITONBACKEND_Request)
            })
            .collect();

        Ok((model, requests))
    }
}
