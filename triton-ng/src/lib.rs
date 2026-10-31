#[path = "backend.rs"]
pub mod backend;
#[path = "backend_handle.rs"]
pub mod backend_handle;
#[path = "error.rs"]
pub mod error;
#[path = "executor.rs"]
pub mod executor;
#[path = "inference_request.rs"]
pub mod inference_request;
#[path = "inference_response.rs"]
pub mod inference_response;
#[path = "log.rs"]
pub mod log;
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
pub use executor::Context;
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

    /// Runs backend code, turning a panic into an error: unwinding across the
    /// `extern "C"` boundary would abort the whole Triton server.
    pub fn catch<T>(
        f: impl FnOnce() -> Result<T, crate::error::Error>,
    ) -> Result<T, crate::error::Error> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|payload| {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            Err(format!("backend panicked: {message}").into())
        })
    }

    pub fn make_internal_error(msg: &str) -> ErrorPtr {
        // Replace null bytes: CString cannot contain them.
        let sanitized = msg.replace('\0', "?");
        // SAFETY: replace('\0') guarantees no null bytes remain.
        let c_str = unsafe { std::ffi::CString::from_vec_unchecked(sanitized.into_bytes()) };
        unsafe {
            triton_ng_sys::TRITONSERVER_ErrorNew(
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                c_str.as_ptr(),
            ) as ErrorPtr
        }
    }

    /// Wraps a raw backend handle pointer. Called from `declare_backend!`.
    pub unsafe fn backend_handle(ptr: *mut c_void) -> crate::backend_handle::BackendHandle {
        crate::backend_handle::BackendHandle::from_ptr(
            ptr as *mut triton_ng_sys::TRITONBACKEND_Backend,
        )
    }

    /// Wraps a raw model pointer. Called from `declare_backend!`.
    pub unsafe fn model(ptr: *mut c_void) -> crate::model::Model {
        crate::model::Model::from_ptr(ptr as *mut triton_ng_sys::TRITONBACKEND_Model)
    }

    /// Drops the model state on finalize. Called from `declare_backend!`.
    pub fn clear_model_state(model: &crate::model::Model) {
        if let Err(e) = model.clear_state() {
            crate::log::error(&format!("failed to clear model state: {e}"));
        }
    }

    /// Starts the runtime of a model instance. Called from `declare_backend!`.
    pub fn start_executor(
        instance: &crate::model_instance::ModelInstance,
    ) -> Result<(), crate::error::Error> {
        catch(|| {
            let model = instance.model()?;
            let thread_name = format!("{}-{}", model.name()?, instance.name()?);
            let executor = crate::executor::InstanceExecutor::new(&model, thread_name)?;
            instance.set_executor(executor)?;
            Ok(())
        })
    }

    /// Waits for the batches of a model instance and stops its runtime.
    /// Called from `declare_backend!`.
    pub fn stop_executor(instance: &crate::model_instance::ModelInstance) {
        match catch(|| Ok(instance.take_executor()?)) {
            Ok(Some(executor)) => {
                if let Err(e) = catch(|| {
                    executor.shutdown();
                    Ok(())
                }) {
                    crate::log::error(&format!("failed to stop model instance runtime: {e}"));
                }
            }
            Ok(None) => {}
            Err(e) => crate::log::error(&format!("failed to take model instance runtime: {e}")),
        }
    }

    /// Hands a batch over to the runtime of its instance. Called from `declare_backend!`.
    ///
    /// Always reports success to Triton once the requests are wrapped: they are
    /// owned by the backend from then on, returning an error would make Triton
    /// respond to (and release) them a second time.
    pub unsafe fn execute<B: crate::backend::Backend>(
        instance: *mut c_void,
        requests_ptr: *const *mut c_void,
        request_count: u32,
    ) -> ErrorPtr {
        let (model, requests) =
            match unsafe { model_and_requests(instance, requests_ptr, request_count) } {
                Ok(parts) => parts,
                Err(err) => return err,
            };
        let instance = unsafe { model_instance(instance) };

        let prepared = catch(|| {
            let executor = instance.executor()?;
            let ctx = crate::executor::context(&model, instance.device_id()?)?;
            Ok((executor, ctx))
        });

        match prepared {
            Ok((executor, ctx)) => {
                let result = catch(|| {
                    executor.spawn(B::model_instance_execute(ctx, requests));
                    Ok(())
                });
                if let Err(e) = result {
                    crate::log::error(&format!("failed to start the batch: {e}"));
                }
            }
            Err(e) => {
                let message = e.to_string();
                crate::log::error(&format!("model instance execute failed: {message}"));
                for request in requests {
                    if let Err(e) = request.respond_error(&message) {
                        crate::log::error(&format!("failed to send error response: {e}"));
                    }
                }
            }
        }

        NULL_ERROR
    }

    /// Wraps a raw model instance pointer. Called from `declare_backend!`.
    pub unsafe fn model_instance(ptr: *mut c_void) -> crate::model_instance::ModelInstance {
        crate::model_instance::ModelInstance::from_ptr(
            ptr as *mut triton_ng_sys::TRITONBACKEND_ModelInstance,
        )
    }

    /// Extracts a `Model` and `Vec<Request>` from a `ModelInstanceExecute` call.
    /// Returns `Err(ErrorPtr)` if `TRITONBACKEND_ModelInstanceModel` fails.
    pub unsafe fn model_and_requests(
        instance: *mut c_void,
        requests_ptr: *const *mut c_void,
        request_count: u32,
    ) -> Result<(crate::model::Model, Vec<crate::request::Request>), ErrorPtr> {
        if instance.is_null() {
            return Err(make_internal_error("model instance is null"));
        }
        if request_count > 0 && requests_ptr.is_null() {
            return Err(make_internal_error("requests array is null"));
        }

        let mut model_ptr: *mut triton_ng_sys::TRITONBACKEND_Model = std::ptr::null_mut();
        let err = unsafe {
            triton_ng_sys::TRITONBACKEND_ModelInstanceModel(
                instance as *mut triton_ng_sys::TRITONBACKEND_ModelInstance,
                &mut model_ptr,
            )
        };
        if !err.is_null() {
            return Err(err as ErrorPtr);
        }

        if model_ptr.is_null() {
            return Err(make_internal_error("model of the instance is null"));
        }

        let model = crate::model::Model::from_ptr(model_ptr);

        if request_count == 0 {
            return Ok((model, Vec::new()));
        }

        let raw = unsafe { std::slice::from_raw_parts(requests_ptr, request_count as usize) };
        // Nothing is touched yet, so returning an error leaves the requests to Triton.
        if raw.iter().any(|r| r.is_null()) {
            return Err(make_internal_error("one of the requests is null"));
        }
        let requests = raw
            .iter()
            .map(|&r| {
                crate::request::Request::from_ptr(r as *mut triton_ng_sys::TRITONBACKEND_Request)
            })
            .collect();

        Ok((model, requests))
    }
}
