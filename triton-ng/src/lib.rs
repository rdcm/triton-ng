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

pub use triton_ng_macros::backend;

/// Entry points called by the code that [`backend`] generates. Not part of the
/// public API.
#[doc(hidden)]
pub mod __macro_support {
    use crate::backend::Backend;
    use crate::backend_handle::BackendHandle;
    use crate::error::Error;
    use crate::executor::InstanceExecutor;
    use crate::model::Model;
    use crate::model_instance::ModelInstance;
    use crate::request::Request;
    use std::ffi::c_void;

    /// Opaque error pointer returned by Triton backend entry points.
    /// Equivalent to `*const TRITONSERVER_Error` at the ABI level.
    pub type ErrorPtr = *const c_void;

    const NULL_ERROR: ErrorPtr = std::ptr::null::<c_void>();

    /// TRITONBACKEND_Initialize.
    pub unsafe fn initialize<B: Backend>(backend: *mut c_void) -> ErrorPtr {
        let backend = unsafe { backend_handle(backend) };
        into_error_ptr(catch(|| B::initialize(&backend)))
    }

    /// TRITONBACKEND_Finalize.
    pub unsafe fn finalize<B: Backend>(backend: *mut c_void) -> ErrorPtr {
        let backend = unsafe { backend_handle(backend) };
        into_error_ptr(catch(|| B::finalize(&backend)))
    }

    /// TRITONBACKEND_ModelInitialize.
    pub unsafe fn model_initialize<B: Backend>(model: *mut c_void) -> ErrorPtr {
        let model = unsafe { self::model(model) };
        into_error_ptr(catch(|| B::model_initialize(&model)))
    }

    /// TRITONBACKEND_ModelFinalize. Drops the model state after the hook.
    pub unsafe fn model_finalize<B: Backend>(model: *mut c_void) -> ErrorPtr {
        let model = unsafe { self::model(model) };
        let result = catch(|| B::model_finalize(&model));
        if let Err(e) = model.clear_state() {
            crate::log::error(&format!("failed to clear model state: {e}"));
        }
        into_error_ptr(result)
    }

    /// TRITONBACKEND_ModelInstanceInitialize. Starts the runtime of the
    /// instance after the hook.
    pub unsafe fn model_instance_initialize<B: Backend>(instance: *mut c_void) -> ErrorPtr {
        let instance = unsafe { model_instance(instance) };
        into_error_ptr(catch(|| {
            B::model_instance_initialize(&instance)?;
            start_executor(&instance)
        }))
    }

    /// TRITONBACKEND_ModelInstanceFinalize. Waits for the batches of the
    /// instance and stops its runtime before the hook.
    pub unsafe fn model_instance_finalize<B: Backend>(instance: *mut c_void) -> ErrorPtr {
        let instance = unsafe { model_instance(instance) };
        stop_executor(&instance);
        into_error_ptr(catch(|| B::model_instance_finalize(&instance)))
    }

    /// TRITONBACKEND_ModelInstanceExecute. Hands the batch over to the runtime
    /// of its instance.
    ///
    /// Always reports success to Triton once the requests are wrapped: they are
    /// owned by the backend from then on, returning an error would make Triton
    /// respond to (and release) them a second time.
    pub unsafe fn model_instance_execute<B: Backend>(
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

    /// Runs backend code, turning a panic into an error: unwinding across the
    /// `extern "C"` boundary would abort the whole Triton server.
    fn catch<T>(f: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|payload| {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            Err(format!("backend panicked: {message}").into())
        })
    }

    fn into_error_ptr(result: Result<(), Error>) -> ErrorPtr {
        match result {
            Ok(()) => NULL_ERROR,
            Err(e) => make_internal_error(&e.to_string()),
        }
    }

    fn make_internal_error(msg: &str) -> ErrorPtr {
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

    unsafe fn backend_handle(ptr: *mut c_void) -> BackendHandle {
        BackendHandle::from_ptr(ptr as *mut triton_ng_sys::TRITONBACKEND_Backend)
    }

    unsafe fn model(ptr: *mut c_void) -> Model {
        Model::from_ptr(ptr as *mut triton_ng_sys::TRITONBACKEND_Model)
    }

    unsafe fn model_instance(ptr: *mut c_void) -> ModelInstance {
        ModelInstance::from_ptr(ptr as *mut triton_ng_sys::TRITONBACKEND_ModelInstance)
    }

    fn start_executor(instance: &ModelInstance) -> Result<(), Error> {
        let model = instance.model()?;
        let thread_name = format!("{}-{}", model.name()?, instance.name()?);
        let executor = InstanceExecutor::new(&model, thread_name)?;
        instance.set_executor(executor)?;
        Ok(())
    }

    fn stop_executor(instance: &ModelInstance) {
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

    /// Extracts a `Model` and `Vec<Request>` from a `ModelInstanceExecute` call.
    /// Returns `Err(ErrorPtr)` if `TRITONBACKEND_ModelInstanceModel` fails.
    unsafe fn model_and_requests(
        instance: *mut c_void,
        requests_ptr: *const *mut c_void,
        request_count: u32,
    ) -> Result<(Model, Vec<Request>), ErrorPtr> {
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

        let model = Model::from_ptr(model_ptr);

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
            .map(|&r| Request::from_ptr(r as *mut triton_ng_sys::TRITONBACKEND_Request))
            .collect();

        Ok((model, requests))
    }
}
