use crate::backend_handle::BackendHandle;
use crate::error::Error;
use crate::executor::Context;
use crate::model::Model;
use crate::model_instance::ModelInstance;
use crate::request::Request;
use std::future::Future;

/// A custom Triton backend, registered with [`crate::declare_backend!`].
///
/// # Examples
///
/// ```ignore
/// struct State {
///     encoder: String,
/// }
///
/// struct MyBackend;
///
/// impl Backend for MyBackend {
///     type State = State;
///
///     fn model_initialize(model: &Model) -> Result<State, Error> {
///         Ok(State { encoder: model.config_parameter_or("encoder", "encoder".to_string())? })
///     }
///
///     async fn model_instance_execute(ctx: Context<State>, requests: Vec<Request>) -> Result<(), Error> {
///         let mut inference = InferenceRequest::new(ctx.server(), &ctx.state().encoder, -1)?;
///         // ... add the inputs of all `requests` as one batch ...
///         let result = ctx.server().infer_async(inference).await?;
///         // ... answer every request from its rows of `result` ...
///         Ok(())
///     }
/// }
///
/// triton_ng::declare_backend!(MyBackend);
/// ```
pub trait Backend {
    /// State shared by all instances of a model, see [`Backend::model_initialize`].
    type State: Send + Sync + 'static;

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

    /// Initialize for a model. This function is called once when a model that
    /// uses the backend is loaded, before any of its instances is created. The
    /// returned state (tokenizers, vocabularies, parsed parameters) is shared
    /// by all instances of the model and reaches every batch through
    /// [`Context::state`].
    ///
    /// Corresponds to TRITONBACKEND_ModelInitialize.
    fn model_initialize(model: &Model) -> Result<Self::State, Error>;

    /// Finalize for a model. This function is optional, a backend is not
    /// required to implement it. The state returned by
    /// [`Backend::model_initialize`] is dropped right after this function
    /// returns, once no batch holds it.
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
    /// Every instance gets its own runtime for its batches, configured by
    /// the model parameters `worker_threads` (threads of the runtime, 1 by
    /// default) and `max_batches_in_flight` (see
    /// [`Backend::model_instance_execute`]).
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceInitialize.
    fn model_instance_initialize(_instance: &ModelInstance) -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a model instance. This function is optional, a
    /// backend is not required to implement it. This function is called
    /// once for an instance, just before the corresponding model is
    /// unloaded from Triton, after every batch of the instance has finished
    /// and its runtime has stopped.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceFinalize.
    fn model_instance_finalize(_instance: &ModelInstance) -> Result<(), Error> {
        Ok(())
    }

    /// Processes a batch of requests collected by Triton. The future runs on
    /// the runtime of the instance, `.await` on a BLS call frees its thread
    /// for the other batches of the instance. Triton gives the instance its
    /// next batch as soon as this batch is handed to the runtime, so an
    /// instance works on several batches at once; with the model parameter
    /// `max_batches_in_flight` the instance takes the next batch only when one
    /// of them has finished, and Triton keeps the requests in its queue.
    ///
    /// Process the batch as a whole and send it downstream as batched BLS
    /// calls: splitting it into tasks per request loses Triton's batching.
    ///
    /// The backend owns `requests` and must give every one exactly one final
    /// response, see [`Request`]. A request dropped without a response, also
    /// when the future returns an error or panics, gets a generic error
    /// response. The returned error is only logged: answer requests with
    /// [`Request::respond_error`] to give clients the reason.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceExecute.
    fn model_instance_execute(
        ctx: Context<Self::State>,
        requests: Vec<Request>,
    ) -> impl Future<Output = Result<(), Error>> + Send + 'static;
}

#[macro_export]
macro_rules! call_checked {
    ($res:expr) => {
        match triton_ng::__macro_support::catch(|| $res) {
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
            triton_ng::call_checked!(unsafe {
                triton_ng::__macro_support::model_initialize::<$class>(model)
            })
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelFinalize(
            model: *mut std::ffi::c_void,
        ) -> triton_ng::__macro_support::ErrorPtr {
            let model = unsafe { triton_ng::__macro_support::model(model) };
            let result = triton_ng::__macro_support::catch(|| $class::model_finalize(&model));
            triton_ng::__macro_support::clear_model_state(&model);
            triton_ng::call_checked!(result)
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceInitialize(
            instance: *mut std::ffi::c_void,
        ) -> triton_ng::__macro_support::ErrorPtr {
            let instance = unsafe { triton_ng::__macro_support::model_instance(instance) };
            triton_ng::call_checked!(
                $class::model_instance_initialize(&instance)
                    .and_then(|()| triton_ng::__macro_support::start_executor(&instance))
            )
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceFinalize(
            instance: *mut std::ffi::c_void,
        ) -> triton_ng::__macro_support::ErrorPtr {
            let instance = unsafe { triton_ng::__macro_support::model_instance(instance) };
            triton_ng::__macro_support::stop_executor(&instance);
            triton_ng::call_checked!($class::model_instance_finalize(&instance))
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceExecute(
            instance: *mut std::ffi::c_void,
            requests: *const *mut std::ffi::c_void,
            request_count: u32,
        ) -> triton_ng::__macro_support::ErrorPtr {
            unsafe {
                triton_ng::__macro_support::execute::<$class>(instance, requests, request_count)
            }
        }
    };
}
