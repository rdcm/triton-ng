use crate::auto_complete_config::AutoCompleteConfig;
use crate::backend::Backend;
use crate::backend_attributes::BackendAttributes;
use crate::error::Error;
use crate::executor::Context;
use crate::model::Model;
use crate::model_instance::ModelInstance;
use crate::request::Request;
use std::future::Future;

/// A custom Triton backend, registered with [`macro@crate::backend`].
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
/// #[triton_ng::backend]
/// impl CustomBackend for MyBackend {
///     fn model_initialize(model: &Model) -> Result<(), Error> {
///         let encoder = model.config_parameter_or("encoder", "encoder".to_string())?;
///         model.set_state(State { encoder })?;
///         Ok(())
///     }
///
///     async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///         let state = ctx.state::<State>()?;
///         let mut inference = InferenceRequest::new(ctx.server(), &state.encoder, -1)?;
///         // ... add the inputs of all `requests` as one batch ...
///         let result = ctx.server().infer_async(inference).await?;
///         // ... answer every request from its rows of `result` ...
///         Ok(())
///     }
/// }
/// ```
pub trait CustomBackend {
    /// Initialize a backend. This function is optional, a backend is not
    /// required to implement it. This function is called once when a
    /// backend is loaded to allow the backend to initialize any state
    /// associated with the backend. Use [`Backend::set_state`] to keep state
    /// shared across all models that use the backend; batches get it through
    /// [`Context::backend_state`]. This is also where the execution policy is
    /// set with [`Backend::set_execution_policy`].
    ///
    /// Corresponds to TRITONBACKEND_Initialize.
    fn initialize(_backend: &Backend) -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a backend. This function is optional, a backend is
    /// not required to implement it. This function is called once, just
    /// before the backend is unloaded. Any threads created for the backend
    /// should be exited/joined before returning from this function. The state
    /// stored with [`Backend::set_state`] is dropped right after this function
    /// returns, once no batch holds it.
    ///
    /// Corresponds to TRITONBACKEND_Finalize.
    fn finalize(_backend: &Backend) -> Result<(), Error> {
        Ok(())
    }

    /// Attributes of the backend Triton uses when it loads models: preferred
    /// instance groups for models without `instance_group` in their config,
    /// and whether instances of a model are initialized in parallel. This
    /// function is optional, the default sets nothing.
    ///
    /// Triton calls it right after [`CustomBackend::initialize`], and may call
    /// it again while the backend is loaded. Triton ignores a failure of it
    /// and keeps the attributes it had; triton-ng logs the error.
    ///
    /// Corresponds to TRITONBACKEND_GetBackendAttribute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn attributes(_backend: &Backend) -> Result<BackendAttributes, Error> {
    ///     Ok(BackendAttributes::new()
    ///         .preferred_instance_group(InstanceGroupKind::Cpu, 1, &[])?
    ///         .parallel_model_instance_loading(true))
    /// }
    /// ```
    fn attributes(_backend: &Backend) -> Result<BackendAttributes, Error> {
        Ok(BackendAttributes::new())
    }

    /// Completes the config of a model that is being loaded: fills in the max
    /// batch size, inputs and outputs, scheduler and `decoupled` that its
    /// `config.pbtxt` leaves out. This function is optional, the default
    /// changes nothing.
    ///
    /// Called right before [`CustomBackend::model_initialize`], unless
    /// Triton runs with `--disable-auto-complete-config`. Triton validates
    /// the completed config after `model_initialize`, so that hook already
    /// reads it through [`Model::config_json`] and [`Model::is_decoupled`].
    /// Models cannot be loaded or unloaded from it, see
    /// [`crate::Server::load_model`]. A failure fails the load of the model.
    ///
    /// Like `auto_complete_config` of python_backend.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn auto_complete_config(_model: &Model, config: &mut AutoCompleteConfig) -> Result<(), Error> {
    ///     config.set_max_batch_size(16)?;
    ///     config.set_dynamic_batching()?;
    ///     config.add_input("AUDIO", DataType::Fp32, &[-1])?;
    ///     config.add_output("TEXT", DataType::Bytes, &[1])?;
    ///     Ok(())
    /// }
    /// ```
    fn auto_complete_config(_model: &Model, _config: &mut AutoCompleteConfig) -> Result<(), Error> {
        Ok(())
    }

    /// Initialize for a model. This function is optional, a backend is not
    /// required to implement it. This function is called once when a model
    /// that uses the backend is loaded, before any of its instances is
    /// created. Use [`Model::set_state`] to keep state shared by all
    /// instances of the model (tokenizers, vocabularies, parsed parameters);
    /// batches get it through [`Context::state`].
    ///
    /// Corresponds to TRITONBACKEND_ModelInitialize.
    fn model_initialize(_model: &Model) -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a model. This function is optional, a backend is not
    /// required to implement it. The state stored with [`Model::set_state`]
    /// is dropped right after this function returns, once no batch holds it.
    ///
    /// Corresponds to TRITONBACKEND_ModelFinalize.
    fn model_finalize(_model: &Model) -> Result<(), Error> {
        Ok(())
    }

    /// Initialize for a model instance. This function is optional, a
    /// backend is not required to implement it. This function is called
    /// once when a model instance is created to allow the backend to
    /// initialize any state associated with the instance. Use
    /// [`ModelInstance::set_state`] to keep state of this instance only;
    /// batches get it through [`Context::instance_state`].
    ///
    /// Instances of a model are initialized one after another, unless
    /// [`CustomBackend::attributes`] enables parallel instance loading.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceInitialize.
    fn model_instance_initialize(_instance: &ModelInstance) -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a model instance. This function is optional, a
    /// backend is not required to implement it. This function is called
    /// once for an instance, just before the corresponding model is
    /// unloaded from Triton, after every batch of the instance has finished.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceFinalize.
    fn model_instance_finalize(_instance: &ModelInstance) -> Result<(), Error> {
        Ok(())
    }

    /// Processes a batch of requests collected by Triton. The future runs to
    /// its end on the Triton thread that called execute, which is blocked while
    /// the future waits for BLS calls: Triton gives the instance its next
    /// batch only after that, and keeps the requests in its queue meanwhile,
    /// where they are batched together. Batches of a model run in parallel
    /// on its instances (`instance_group` count).
    ///
    /// The future may await several BLS calls at once (`join!`), they all
    /// run in Triton meanwhile. There is no async runtime around it: timers,
    /// I/O and spawning of tokio or other runtimes do not work in it. A
    /// blocking call blocks the execution thread, including every instance
    /// sharing it under [`crate::ExecutionPolicy::DeviceBlocking`].
    ///
    /// Process the batch as a whole and send it downstream as batched BLS
    /// calls: splitting it into tasks per request loses Triton's batching.
    ///
    /// The backend owns `requests` and must give every one exactly one final
    /// response, see [`Request`]; a request of a decoupled model may get a
    /// stream of responses before it, see [`Request::into_sender`]. The batch
    /// is in flight until its future ends, also while it streams responses.
    /// A request dropped without a final response, also
    /// when the future returns an error or panics, gets a generic error
    /// response. The returned error is only logged: answer requests with
    /// [`Request::respond_error`] to give clients the reason.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceExecute.
    fn model_instance_execute(
        ctx: Context,
        requests: Vec<Request>,
    ) -> impl Future<Output = Result<(), Error>> + Send + 'static;
}
