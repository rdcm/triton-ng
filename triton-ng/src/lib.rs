#[path = "allocation.rs"]
pub(crate) mod allocation;
#[path = "backend.rs"]
pub mod backend;
#[path = "backend_config.rs"]
pub mod backend_config;
#[path = "buffers.rs"]
pub mod buffers;
#[path = "custom_backend.rs"]
pub mod custom_backend;
#[path = "error.rs"]
pub mod error;
#[path = "executor.rs"]
pub mod executor;
#[path = "inference_future.rs"]
pub mod inference_future;
#[path = "inference_request.rs"]
pub mod inference_request;
#[path = "inference_response.rs"]
pub(crate) mod inference_response;
#[path = "inference_result.rs"]
pub mod inference_result;
#[path = "inference_stream.rs"]
pub mod inference_stream;
#[path = "inference_trace.rs"]
pub(crate) mod inference_trace;
#[path = "input.rs"]
pub mod input;
#[path = "input_buffer.rs"]
pub mod input_buffer;
#[path = "log.rs"]
pub mod log;
#[path = "memory_manager.rs"]
pub mod memory_manager;
#[path = "message.rs"]
pub(crate) mod message;
#[path = "metric.rs"]
pub(crate) mod metric;
#[path = "metric_args.rs"]
pub(crate) mod metric_args;
#[path = "metric_family.rs"]
pub(crate) mod metric_family;
#[path = "metric_parameter.rs"]
pub(crate) mod metric_parameter;
#[path = "metrics.rs"]
pub mod metrics;
#[path = "model.rs"]
pub mod model;
#[path = "model_instance.rs"]
pub mod model_instance;
#[path = "output.rs"]
pub mod output;
#[path = "output_buffer.rs"]
pub mod output_buffer;
#[path = "raw_buffer.rs"]
mod raw_buffer;
#[path = "request.rs"]
pub mod request;
#[path = "request_cancel.rs"]
mod request_cancel;
#[path = "request_release.rs"]
pub(crate) mod request_release;
#[path = "response.rs"]
pub mod response;
#[path = "response_allocator.rs"]
pub(crate) mod response_allocator;
#[path = "response_factory.rs"]
mod response_factory;
#[path = "response_sender.rs"]
pub mod response_sender;
#[path = "server.rs"]
pub mod server;
#[path = "stream_response.rs"]
pub mod stream_response;
#[path = "types.rs"]
pub mod types;
#[path = "utils.rs"]
pub(crate) mod utils;

pub use backend::*;
pub use backend_config::*;
pub use buffers::*;
pub use custom_backend::*;
pub use error::*;
pub use executor::Context;
pub use inference_future::*;
pub use inference_request::*;
pub use inference_result::*;
pub use inference_stream::*;
pub use input::*;
pub use input_buffer::*;
pub use memory_manager::*;
pub use metrics::*;
pub use model::*;
pub use model_instance::*;
pub use output::*;
pub use output_buffer::*;
pub(crate) use raw_buffer::{RawBuffer, Sealed};
pub use request::*;
pub use response::*;
pub use response_sender::*;
pub use server::*;
pub use stream_response::*;
pub use types::*;

pub use triton_ng_macros::backend;

/// Entry points called by the code that [`macro@backend`] generates. Not part of the
/// public API.
#[doc(hidden)]
pub mod __macro_support {
    use crate::backend::Backend;
    use crate::custom_backend::CustomBackend;
    use crate::error::{Error, ErrorCode, TritonError};
    use crate::executor::InstanceExecutor;
    use crate::model::Model;
    use crate::model_instance::ModelInstance;
    use crate::request::Request;
    use crate::response_allocator::ResponseAllocator;
    use std::ffi::c_void;

    /// Opaque error pointer returned by Triton backend entry points.
    /// Equivalent to `*const TRITONSERVER_Error` at the ABI level.
    pub type ErrorPtr = *const c_void;

    const NULL_ERROR: ErrorPtr = std::ptr::null::<c_void>();

    /// TRITONBACKEND_Initialize.
    pub unsafe fn initialize<B: CustomBackend>(backend: *mut c_void) -> ErrorPtr {
        let backend = unsafe { self::backend(backend) };
        into_error_ptr(catch(|| B::initialize(&backend)))
    }

    /// TRITONBACKEND_Finalize. Lets the shared BLS response allocator go
    /// after the hook.
    pub unsafe fn finalize<B: CustomBackend>(backend: *mut c_void) -> ErrorPtr {
        let backend = unsafe { self::backend(backend) };
        let result = catch(|| B::finalize(&backend));
        ResponseAllocator::release_shared();
        into_error_ptr(result)
    }

    /// TRITONBACKEND_ModelInitialize.
    pub unsafe fn model_initialize<B: CustomBackend>(model: *mut c_void) -> ErrorPtr {
        let model = unsafe { self::model(model) };
        into_error_ptr(catch(|| B::model_initialize(&model)))
    }

    /// TRITONBACKEND_ModelFinalize. Drops the model state after the hook.
    pub unsafe fn model_finalize<B: CustomBackend>(model: *mut c_void) -> ErrorPtr {
        let model = unsafe { self::model(model) };
        let result = catch(|| B::model_finalize(&model));
        if let Err(e) = model.clear_state() {
            crate::log::error(&format!("failed to clear model state: {e}"));
        }
        into_error_ptr(result)
    }

    /// TRITONBACKEND_ModelInstanceInitialize. Starts the runtime of the
    /// instance before the hook, which may store the instance state in it,
    /// and stops it again when the hook fails.
    pub unsafe fn model_instance_initialize<B: CustomBackend>(instance: *mut c_void) -> ErrorPtr {
        let instance = unsafe { model_instance(instance) };
        let result = catch(|| {
            start_executor(&instance)?;
            B::model_instance_initialize(&instance)
        });
        if result.is_err() {
            stop_executor(&instance);
        }
        into_error_ptr(result)
    }

    /// TRITONBACKEND_ModelInstanceFinalize. Waits for the batches of the
    /// instance and stops its runtime before the hook.
    pub unsafe fn model_instance_finalize<B: CustomBackend>(instance: *mut c_void) -> ErrorPtr {
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
    pub unsafe fn model_instance_execute<B: CustomBackend>(
        instance: *mut c_void,
        requests_ptr: *const *mut c_void,
        request_count: u32,
    ) -> ErrorPtr {
        let (model, raw_requests) =
            match unsafe { model_and_requests(instance, requests_ptr, request_count) } {
                Ok(parts) => parts,
                Err(err) => return err,
            };
        let instance = unsafe { model_instance(instance) };

        let prepared = catch(|| {
            let executor = instance.executor()?;
            let ctx = crate::executor::context(&model, executor, instance.device_id()?)?;
            Ok((executor, ctx))
        });

        // Without the executor the requests only get error responses, for
        // which it does not matter whether the model is decoupled.
        let decoupled = prepared
            .as_ref()
            .is_ok_and(|(executor, _)| executor.decoupled());
        let requests: Vec<Request> = raw_requests
            .into_iter()
            .map(|r| Request::from_ptr(r, decoupled))
            .collect();

        match prepared {
            Ok((executor, ctx)) => {
                let batch = B::model_instance_execute(ctx, requests);
                let result = catch(|| {
                    executor.spawn(async move {
                        if let Err(e) = batch.await {
                            crate::log::error(&format!("model instance execute failed: {e}"));
                        }
                    })?;
                    Ok(())
                });
                if let Err(e) = result {
                    crate::log::error(&format!("failed to start the batch: {e}"));
                }
            }
            Err(e) => {
                let code = TritonError::code_of(&e);
                let message = e.to_string();
                crate::log::error(&format!("model instance execute failed: {message}"));
                for request in requests {
                    if let Err(e) = request.respond_error_with_code(code, &message) {
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
            Err(e) => make_error(TritonError::code_of(&e), &e.to_string()),
        }
    }

    fn make_internal_error(msg: &str) -> ErrorPtr {
        make_error(ErrorCode::Internal, msg)
    }

    fn make_error(code: ErrorCode, msg: &str) -> ErrorPtr {
        // Replace null bytes: CString cannot contain them.
        let sanitized = msg.replace('\0', "?");
        // SAFETY: replace('\0') guarantees no null bytes remain.
        let c_str = unsafe { std::ffi::CString::from_vec_unchecked(sanitized.into_bytes()) };
        unsafe { triton_ng_sys::TRITONSERVER_ErrorNew(code.to_sys(), c_str.as_ptr()) as ErrorPtr }
    }

    unsafe fn backend(ptr: *mut c_void) -> Backend {
        Backend::from_ptr(ptr as *mut triton_ng_sys::TRITONBACKEND_Backend)
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

    /// Extracts the `Model` and the requests of a `ModelInstanceExecute` call.
    /// Returns `Err(ErrorPtr)` if `TRITONBACKEND_ModelInstanceModel` fails.
    unsafe fn model_and_requests(
        instance: *mut c_void,
        requests_ptr: *const *mut c_void,
        request_count: u32,
    ) -> Result<(Model, Vec<*mut triton_ng_sys::TRITONBACKEND_Request>), ErrorPtr> {
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
            .map(|&r| r as *mut triton_ng_sys::TRITONBACKEND_Request)
            .collect();

        Ok((model, requests))
    }
}

#[cfg(test)]
mod tests {
    use crate::allocation::Allocation;
    use crate::inference_response::InferenceResponse;
    use crate::inference_trace::InferenceTrace;
    use crate::message::Message;
    use crate::metric::Metric;
    use crate::metric_args::MetricArgs;
    use crate::metric_family::MetricFamily;
    use crate::metric_parameter::MetricParameter;
    use crate::request_cancel::RequestCancel;
    use crate::request_release::RequestRelease;
    use crate::response_allocator::ResponseAllocator;
    use crate::response_factory::ResponseFactory;
    use crate::*;

    fn send<T: ?Sized + Send>() {}
    fn sync<T: ?Sized + Sync>() {}

    trait NotSend<A> {
        fn check() {}
    }
    impl<T: ?Sized> NotSend<()> for T {}
    impl<T: ?Sized + Send> NotSend<u8> for T {}

    trait NotSync<A> {
        fn check() {}
    }
    impl<T: ?Sized> NotSync<()> for T {}
    impl<T: ?Sized + Sync> NotSync<u8> for T {}

    macro_rules! send_and_sync {
        ($($t:ty),*) => {$(send::<$t>(); sync::<$t>();)*};
    }
    macro_rules! send_not_sync {
        ($($t:ty),*) => {$(send::<$t>(); <$t as NotSync<_>>::check();)*};
    }
    macro_rules! neither {
        ($($t:ty),*) => {$(<$t as NotSend<_>>::check(); <$t as NotSync<_>>::check();)*};
    }

    #[test]
    fn thread_safety_of_every_handle_is_as_documented() {
        send_and_sync!(
            TritonError,
            Server,
            Context,
            MemoryManager,
            RamBuffer,
            PinnedRamBuffer,
            GpuBuffer,
            UninitGpuBuffer,
            OutputBuffer,
            InferenceFuture,
            InferenceResult,
            InferenceStream,
            RequestCancel,
            Message,
            ResponseAllocator,
            Allocation,
            RequestRelease,
            CounterFamily,
            Counter,
            GaugeFamily,
            Gauge,
            HistogramFamily,
            Histogram,
            MetricFamily,
            Metric
        );
        send_not_sync!(
            Request,
            Response,
            ResponseBody,
            ResponseSender,
            StreamResponse<'static>,
            InferenceRequest,
            InferenceResponse,
            InferenceTrace
        );
        neither!(
            MetricParameter,
            MetricArgs,
            Model,
            ModelInstance,
            Backend,
            ResponseFactory,
            Input<'static>,
            Output<'static>,
            RawBuffer
        );
    }
}
