#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

#[path = "allocation.rs"]
pub(crate) mod allocation;
#[path = "auto_complete_config.rs"]
mod auto_complete_config;
#[path = "backend.rs"]
mod backend;
#[path = "backend_attribute.rs"]
pub(crate) mod backend_attribute;
#[path = "backend_attributes.rs"]
mod backend_attributes;
#[path = "backend_config.rs"]
mod backend_config;
#[path = "batch_statistics.rs"]
pub(crate) mod batch_statistics;
#[path = "buffer_attributes.rs"]
mod buffer_attributes;
#[path = "buffers.rs"]
mod buffers;
#[path = "custom_backend.rs"]
mod custom_backend;
#[path = "error.rs"]
mod error;
#[path = "executor.rs"]
mod executor;
#[path = "failure.rs"]
pub(crate) mod failure;
#[path = "final_response.rs"]
pub(crate) mod final_response;
#[path = "inference_request.rs"]
mod inference_request;
#[path = "inference_response.rs"]
pub(crate) mod inference_response;
#[path = "inference_result.rs"]
mod inference_result;
#[path = "inference_stream.rs"]
mod inference_stream;
#[path = "inference_trace.rs"]
pub(crate) mod inference_trace;
#[path = "input.rs"]
mod input;
#[path = "input_buffer.rs"]
mod input_buffer;
#[path = "log.rs"]
pub mod log;
#[path = "memory_manager.rs"]
mod memory_manager;
#[path = "message.rs"]
pub(crate) mod message;
#[path = "metric.rs"]
pub(crate) mod metric;
#[path = "metric_args.rs"]
pub(crate) mod metric_args;
#[path = "metric_family.rs"]
pub(crate) mod metric_family;
#[path = "metrics.rs"]
mod metrics;
#[path = "model.rs"]
mod model;
#[path = "model_control.rs"]
pub(crate) mod model_control;
#[path = "model_instance.rs"]
mod model_instance;
#[path = "output.rs"]
mod output;
#[path = "output_buffer.rs"]
mod output_buffer;
#[path = "raw_buffer.rs"]
mod raw_buffer;
#[path = "request.rs"]
mod request;
#[path = "request_cancel.rs"]
mod request_cancel;
#[path = "request_release.rs"]
pub(crate) mod request_release;
#[path = "response.rs"]
mod response;
#[path = "response_allocator.rs"]
pub(crate) mod response_allocator;
#[path = "response_factory.rs"]
mod response_factory;
#[path = "response_sender.rs"]
mod response_sender;
#[path = "server.rs"]
mod server;
#[path = "shared_input.rs"]
mod shared_input;
#[path = "stream_response.rs"]
mod stream_response;
#[path = "triton_parameter.rs"]
pub(crate) mod triton_parameter;
#[path = "types.rs"]
mod types;
#[path = "utils.rs"]
pub(crate) mod utils;

pub use auto_complete_config::*;
pub use backend::*;
pub use backend_attributes::*;
pub use backend_config::*;
pub use buffer_attributes::*;
pub use buffers::*;
pub use custom_backend::*;
pub use error::*;
pub use executor::Context;
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
pub use shared_input::*;
pub use stream_response::*;
pub use types::*;

pub use triton_ng_macros::backend;

/// The guides of the repository's `docs/` directory, one module per guide, as on GitHub.
///
#[doc = include_str!("../docs/README.md")]
pub mod docs {
    #[doc = "Batching and concurrency: how Triton forms batches, why a batch runs on the Triton thread of its instance, tuning with `instance_group`, measurements."]
    #[doc = ""]
    #[doc = include_str!("../docs/batching_and_concurrency.md")]
    pub mod batching_and_concurrency {}

    #[doc = "Execution model and thread safety: instances and their threads, state, request ownership, error codes, BLS futures, cancellation, shutdown."]
    #[doc = ""]
    #[doc = include_str!("../docs/execution_model_and_thread_safety.md")]
    pub mod execution_model_and_thread_safety {}

    #[doc = "Streaming: decoupled models, streams of responses from a backend, BLS streams, cancellation of BLS calls."]
    #[doc = ""]
    #[doc = include_str!("../docs/streaming.md")]
    pub mod streaming {}

    #[doc = "Model parameters: the `parameters` map of a model config and how a backend reads it."]
    #[doc = ""]
    #[doc = include_str!("../docs/model_parameters.md")]
    pub mod model_parameters {}

    #[doc = "Request metadata and tracing: ID, correlation ID, sequence flags and timeout of a request, BLS calls as children of its trace."]
    #[doc = ""]
    #[doc = include_str!("../docs/request_metadata_and_tracing.md")]
    pub mod request_metadata_and_tracing {}

    #[doc = "Metrics: counters, gauges and histograms of the backend on Triton's Prometheus endpoint, and the inference statistics triton-ng reports."]
    #[doc = ""]
    #[doc = include_str!("../docs/metrics.md")]
    pub mod metrics {}

    #[doc = "Model management and configuration: backend config and attributes, completing model configs, loading and unloading models, rescheduling requests."]
    #[doc = ""]
    #[doc = include_str!("../docs/model_management_and_configuration.md")]
    pub mod model_management_and_configuration {}

    #[doc = "Memory buffers: RAM, pinned and GPU buffers from Triton's memory manager for BLS inputs and outputs, passing inputs of a request on without a copy."]
    #[doc = ""]
    #[doc = include_str!("../docs/memory_buffers.md")]
    pub mod memory_buffers {}

    #[doc = "C API coverage: what of `tritonbackend.h` and of the server functions a backend calls is wrapped, and what is not."]
    #[doc = ""]
    #[doc = include_str!("../docs/c_api_coverage.md")]
    pub mod c_api_coverage {}

    #[doc = "Benchmarks: python_backend and triton-ng on the same BLS workloads with `perf_analyzer`, the method, the models, the environment and the results."]
    #[doc = ""]
    #[doc = include_str!("../docs/benchmarks.md")]
    pub mod benchmarks {}
}

/// Entry points called by the code that [`macro@backend`] generates. Not part of the
/// public API.
#[doc(hidden)]
pub mod __macro_support {
    use crate::auto_complete_config::AutoCompleteConfig;
    use crate::backend::Backend;
    use crate::backend_attribute::BackendAttribute;
    use crate::custom_backend::CustomBackend;
    use crate::error::{Error, ErrorCode, TritonError};
    use crate::executor::InstanceExecutor;
    use crate::model::Model;
    use crate::model_control::forbid_model_control;
    use crate::model_instance::ModelInstance;
    use crate::request::Request;
    use crate::response_allocator::ResponseAllocator;
    use std::ffi::c_void;
    use triton_ng_macros::triton_call;

    /// Opaque error pointer returned by Triton backend entry points.
    /// Equivalent to `*const TRITONSERVER_Error` at the ABI level.
    pub type ErrorPtr = *const c_void;

    const NULL_ERROR: ErrorPtr = std::ptr::null::<c_void>();

    /// TRITONBACKEND_Initialize. Rejects a server whose backend API version
    /// this build cannot run on, before the hook.
    pub unsafe fn initialize<B: CustomBackend>(backend: *mut c_void) -> ErrorPtr {
        let backend = unsafe { self::backend(backend) };
        into_error_ptr(catch(|| {
            check_api_version()?;
            B::initialize(&backend)
        }))
    }

    /// Fails unless the server's backend API version is compatible with the
    /// headers this crate was built against, and logs both versions.
    fn check_api_version() -> Result<(), Error> {
        let mut major = 0;
        let mut minor = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_ApiVersion(
            &mut major, &mut minor
        ))?;
        validate_api_version(major, minor)
            .map_err(|message| TritonError::with_code(ErrorCode::Unsupported, message))?;
        crate::log::info(&format!(
            "TRITONBACKEND API version: server {major}.{minor}, compiled {}.{}",
            triton_ng_sys::TRITONBACKEND_API_VERSION_MAJOR,
            triton_ng_sys::TRITONBACKEND_API_VERSION_MINOR
        ));
        Ok(())
    }

    /// The check tritonbackend.h asks for: same major version, server minor not
    /// lower than the compiled one, so that every symbol this build uses exists.
    /// Returns the message of the mismatch; the caller gives it the
    /// `UNSUPPORTED` code the header asks for.
    fn validate_api_version(server_major: u32, server_minor: u32) -> Result<(), String> {
        let compiled_major = triton_ng_sys::TRITONBACKEND_API_VERSION_MAJOR;
        let compiled_minor = triton_ng_sys::TRITONBACKEND_API_VERSION_MINOR;
        if server_major != compiled_major || server_minor < compiled_minor {
            return Err(format!(
                "the server's TRITONBACKEND API version {server_major}.{server_minor} is older than \
                 the {compiled_major}.{compiled_minor} this backend was built against",
            ));
        }
        Ok(())
    }

    /// TRITONBACKEND_Finalize. Drops the backend state and lets the shared BLS
    /// response allocator go after the hook.
    pub unsafe fn finalize<B: CustomBackend>(backend: *mut c_void) -> ErrorPtr {
        let backend = unsafe { self::backend(backend) };
        let result = forbid_model_control("finalize", || catch(|| B::finalize(&backend)));
        if let Err(e) = catch(|| {
            backend.clear_state()?;
            Ok(())
        }) {
            crate::log::error(&format!("failed to clear backend state: {e}"));
        }
        ResponseAllocator::release_shared();
        into_error_ptr(result)
    }

    /// TRITONBACKEND_GetBackendAttribute. Logs a failure: the core drops the
    /// returned error without a word.
    pub unsafe fn backend_attributes<B: CustomBackend>(
        backend: *mut c_void,
        attributes: *mut c_void,
    ) -> ErrorPtr {
        let backend = unsafe { self::backend(backend) };
        let result = catch(|| {
            let attribute = BackendAttribute::from_ptr(
                attributes as *mut triton_ng_sys::TRITONBACKEND_BackendAttribute,
            )?;
            attribute.set(&B::attributes(&backend)?)?;
            Ok(())
        });
        if let Err(e) = &result {
            crate::log::error(&format!("failed to set the backend attributes: {e}"));
        }
        into_error_ptr(result)
    }

    /// TRITONBACKEND_ModelInitialize.
    pub unsafe fn model_initialize<B: CustomBackend>(model: *mut c_void) -> ErrorPtr {
        let model = unsafe { self::model(model) };
        into_error_ptr(catch(|| {
            auto_complete_config::<B>(&model)?;
            B::model_initialize(&model)
        }))
    }

    /// Runs `CustomBackend::auto_complete_config` and gives Triton the
    /// completed config, when Triton asks for it.
    fn auto_complete_config<B: CustomBackend>(model: &Model) -> Result<(), Error> {
        if !model.auto_complete_config_enabled()? {
            return Ok(());
        }
        let original = AutoCompleteConfig::from_json(&model.config_json()?)?;
        let mut config = original.clone();
        forbid_model_control("auto_complete_config", || {
            B::auto_complete_config(model, &mut config)
        })?;
        if config != original {
            model.set_config(&config.to_json())?;
        }
        Ok(())
    }

    /// TRITONBACKEND_ModelFinalize. Drops the model state after the hook.
    pub unsafe fn model_finalize<B: CustomBackend>(model: *mut c_void) -> ErrorPtr {
        let model = unsafe { self::model(model) };
        let result = forbid_model_control("model_finalize", || catch(|| B::model_finalize(&model)));
        if let Err(e) = model.clear_state() {
            crate::log::error(&format!("failed to clear model state: {e}"));
        }
        into_error_ptr(result)
    }

    /// TRITONBACKEND_ModelInstanceInitialize. Creates the executor of the
    /// instance before the hook, which may store the instance state in it,
    /// and drops it again when the hook fails.
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

    /// TRITONBACKEND_ModelInstanceFinalize. Drops the executor of the
    /// instance, with the instance state, before the hook. No batch is
    /// running: Triton finalizes an instance only between executes.
    pub unsafe fn model_instance_finalize<B: CustomBackend>(instance: *mut c_void) -> ErrorPtr {
        let instance = unsafe { model_instance(instance) };
        stop_executor(&instance);
        into_error_ptr(forbid_model_control("model_instance_finalize", || {
            catch(|| B::model_instance_finalize(&instance))
        }))
    }

    /// TRITONBACKEND_ModelInstanceExecute. Runs the batch to its end on the
    /// calling Triton thread, so the instance takes its next batch only
    /// after it.
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
        let statistics = std::sync::Arc::new(crate::batch_statistics::BatchStatistics::new(
            instance as *mut triton_ng_sys::TRITONBACKEND_ModelInstance,
        ));
        let instance = unsafe { model_instance(instance) };

        let prepared = catch(|| {
            let executor = instance.executor()?;
            let ctx = crate::executor::context(&model, executor, instance.device_id()?)?;
            Ok((executor, ctx))
        });

        // Without the executor the requests only get error responses, for
        // which the policy of the model does not matter.
        let policy = prepared
            .as_ref()
            .map(|(executor, _)| executor.policy())
            .unwrap_or_default();
        let requests: Vec<Request> = raw_requests
            .into_iter()
            .map(|r| Request::from_ptr(r, policy, std::sync::Arc::clone(&statistics)))
            .collect();
        // A request whose batch size cannot be read counts as one element,
        // which is what Triton assumes for a request without inputs.
        let batch_size: u64 = requests
            .iter()
            .map(|request| request.batch_size().unwrap_or(1))
            .sum();

        match prepared {
            Ok((_, ctx)) => {
                let result =
                    catch(|| crate::executor::block_on(B::model_instance_execute(ctx, requests)));
                if let Err(e) = result {
                    crate::log::error(&format!("model instance execute failed: {e}"));
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
        if let Err(e) = statistics.report_batch(batch_size) {
            crate::log::error(&format!("failed to report batch statistics: {e}"));
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
        let executor = InstanceExecutor::new(&instance.model()?)?;
        instance.set_executor(executor)?;
        Ok(())
    }

    fn stop_executor(instance: &ModelInstance) {
        match catch(|| Ok(instance.take_executor()?)) {
            Ok(Some(executor)) => {
                if let Err(e) = catch(|| {
                    drop(executor);
                    Ok(())
                }) {
                    crate::log::error(&format!("failed to drop model instance state: {e}"));
                }
            }
            Ok(None) => {}
            Err(e) => crate::log::error(&format!("failed to take model instance executor: {e}")),
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

    #[cfg(test)]
    mod tests {
        use super::validate_api_version;
        use triton_ng_sys::{TRITONBACKEND_API_VERSION_MAJOR, TRITONBACKEND_API_VERSION_MINOR};

        #[test]
        fn the_same_version_and_a_newer_server_are_accepted() {
            for minor in [
                TRITONBACKEND_API_VERSION_MINOR,
                TRITONBACKEND_API_VERSION_MINOR + 1,
            ] {
                validate_api_version(TRITONBACKEND_API_VERSION_MAJOR, minor).unwrap();
            }
        }

        #[test]
        fn another_major_version_is_rejected() {
            for major in [
                TRITONBACKEND_API_VERSION_MAJOR - 1,
                TRITONBACKEND_API_VERSION_MAJOR + 1,
            ] {
                let message =
                    validate_api_version(major, TRITONBACKEND_API_VERSION_MINOR).unwrap_err();
                assert!(message.contains(&format!("{major}.{TRITONBACKEND_API_VERSION_MINOR}")));
                assert!(message.contains(&format!(
                    "{TRITONBACKEND_API_VERSION_MAJOR}.{TRITONBACKEND_API_VERSION_MINOR}"
                )));
            }
        }

        #[test]
        fn an_older_server_is_rejected() {
            let message = validate_api_version(
                TRITONBACKEND_API_VERSION_MAJOR,
                TRITONBACKEND_API_VERSION_MINOR - 1,
            )
            .unwrap_err();
            assert!(message.contains("older than"));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::allocation::Allocation;
    use crate::backend_attribute::BackendAttribute;
    use crate::final_response::FinalResponse;
    use crate::inference_response::InferenceResponse;
    use crate::inference_trace::InferenceTrace;
    use crate::message::Message;
    use crate::metric::Metric;
    use crate::metric_args::MetricArgs;
    use crate::metric_family::MetricFamily;
    use crate::request_cancel::RequestCancel;
    use crate::request_release::RequestRelease;
    use crate::response_allocator::ResponseAllocator;
    use crate::response_factory::ResponseFactory;
    use crate::triton_parameter::TritonParameter;
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
            InferenceResult,
            InferenceStream,
            RequestCancel,
            Message,
            ResponseAllocator,
            Allocation,
            RequestRelease,
            FinalResponse,
            SharedInput,
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
            TritonParameter<'static>,
            MetricArgs,
            Model,
            ModelInstance,
            Backend,
            BackendAttribute,
            ResponseFactory,
            Input<'static>,
            Output<'static>,
            RawBuffer
        );
    }
}
