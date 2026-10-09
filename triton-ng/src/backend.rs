use crate::backend_handle::BackendHandle;
use crate::error::Error;
use crate::model::Model;
use crate::model_instance::ModelInstance;
use crate::request::Request;

pub trait Backend {
    /// Initialize a backend. This function is optional, a backend is not
    /// required to implement it. This function is called once when a
    /// backend is loaded to allow the backend to initialize any state
    /// associated with the backend. A backend has a single state that is
    /// shared across all models that use the backend.
    ///
    /// Corresponds to TRITONBACKEND_Initialize.
    fn initialize(_backend: &BackendHandle) -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a backend. This function is optional, a backend is
    /// not required to implement it. This function is called once, just
    /// before the backend is unloaded. All state associated with the
    /// backend should be freed and any threads created for the backend
    /// should be exited/joined before returning from this function.
    ///
    /// Corresponds to TRITONBACKEND_Finalize.
    fn finalize(_backend: &BackendHandle) -> Result<(), Error> {
        Ok(())
    }

    /// Initialize for a model. This function is optional, a backend is not
    /// required to implement it. This function is called once when a model
    /// that uses the backend is loaded, before any of its instances is
    /// created. Use [`Model::set_state`] to keep state shared by all
    /// instances of the model (tokenizers, vocabularies, parsed parameters).
    ///
    /// Corresponds to TRITONBACKEND_ModelInitialize.
    fn model_initialize(_model: &Model) -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a model. This function is optional, a backend is not
    /// required to implement it. The state stored with [`Model::set_state`]
    /// is dropped automatically right after this function returns.
    ///
    /// Corresponds to TRITONBACKEND_ModelFinalize.
    fn model_finalize(_model: &Model) -> Result<(), Error> {
        Ok(())
    }

    /// Initialize for a model instance. This function is optional, a
    /// backend is not required to implement it. This function is called
    /// once when a model instance is created to allow the backend to
    /// initialize any state associated with the instance.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceInitialize.
    fn model_instance_initialize(_instance: &ModelInstance) -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a model instance. This function is optional, a
    /// backend is not required to implement it. This function is called
    /// once for an instance, just before the corresponding model is
    /// unloaded from Triton. All state associated with the instance
    /// should be freed and any threads created for the instance should be
    /// exited/joined before returning from this function.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceFinalize.
    fn model_instance_finalize(_instance: &ModelInstance) -> Result<(), Error> {
        Ok(())
    }

    /// Execute a batch of one or more requests on a model instance. This
    /// function is required. Triton will not perform multiple
    /// simultaneous calls to this function for a given model 'instance';
    /// however, there may be simultaneous calls for different model
    /// instances (for the same or different models).
    ///
    /// Every request should get exactly one final response, either via
    /// [`crate::Response::send`] or [`Request::respond_error`]. Requests left
    /// without a response get an error response carrying the returned error
    /// (or a generic one on success). Requests are released by the
    /// `declare_backend!` glue, implementations must not release them.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceExecute.
    fn model_instance_execute(model: Model, requests: &[Request]) -> Result<(), Error>;
}

#[macro_export]
macro_rules! call_checked {
    ($res:expr) => {
        match $res {
            Err(err) => triton_ng::__macro_support::make_internal_error(&err.to_string()),
            Ok(_) => triton_ng::__macro_support::NULL_ERROR,
        }
    };
}

#[macro_export]
macro_rules! declare_backend {
    ($class:ident) => {
        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_Initialize(
            backend: *mut std::ffi::c_void,
        ) -> triton_ng::__macro_support::ErrorPtr {
            triton_ng::call_checked!($class::initialize(&unsafe {
                triton_ng::__macro_support::backend_handle(backend)
            }))
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_Finalize(
            backend: *mut std::ffi::c_void,
        ) -> triton_ng::__macro_support::ErrorPtr {
            triton_ng::call_checked!($class::finalize(&unsafe {
                triton_ng::__macro_support::backend_handle(backend)
            }))
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInitialize(
            model: *mut std::ffi::c_void,
        ) -> triton_ng::__macro_support::ErrorPtr {
            triton_ng::call_checked!($class::model_initialize(&unsafe {
                triton_ng::__macro_support::model(model)
            }))
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelFinalize(
            model: *mut std::ffi::c_void,
        ) -> triton_ng::__macro_support::ErrorPtr {
            let model = unsafe { triton_ng::__macro_support::model(model) };
            let result = $class::model_finalize(&model);
            triton_ng::__macro_support::clear_model_state(&model);
            triton_ng::call_checked!(result)
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceInitialize(
            instance: *mut std::ffi::c_void,
        ) -> triton_ng::__macro_support::ErrorPtr {
            triton_ng::call_checked!($class::model_instance_initialize(&unsafe {
                triton_ng::__macro_support::model_instance(instance)
            }))
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceFinalize(
            instance: *mut std::ffi::c_void,
        ) -> triton_ng::__macro_support::ErrorPtr {
            triton_ng::call_checked!($class::model_instance_finalize(&unsafe {
                triton_ng::__macro_support::model_instance(instance)
            }))
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceExecute(
            instance: *mut std::ffi::c_void,
            requests: *const *mut std::ffi::c_void,
            request_count: u32,
        ) -> triton_ng::__macro_support::ErrorPtr {
            match unsafe {
                triton_ng::__macro_support::model_and_requests(instance, requests, request_count)
            } {
                Err(err) => err,
                Ok((model, requests)) => {
                    let result = $class::model_instance_execute(model, &requests);
                    triton_ng::__macro_support::complete_requests(&requests, result)
                }
            }
        }
    };
}
