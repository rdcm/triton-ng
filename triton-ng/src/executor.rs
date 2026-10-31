use crate::error::Error;
use crate::log;
use crate::model::Model;
use crate::server::Server;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;
use tokio_util::task::TaskTracker;

/// Model parameter: worker threads of every instance's runtime.
const WORKER_THREADS: &str = "worker_threads";
/// Model parameter: batches an instance processes at once, unlimited when absent or 0.
const MAX_BATCHES_IN_FLIGHT: &str = "max_batches_in_flight";

/// What a batch task gets from its model: the server for BLS calls and the
/// state returned by [`crate::Backend::model_initialize`].
///
/// `Context` is `Send + Clone` and holds no Triton pointers, so it can be used
/// across `.await` and passed to other tasks.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context<State>, requests: Vec<Request>) -> Result<(), Error> {
///     let mut inference = InferenceRequest::new(ctx.server(), &ctx.state().encoder, -1)?;
///     // ... add the inputs of all `requests` as one batch ...
///     let result = ctx.server().infer_async(inference).await?;
///     // ... answer every request from its rows of `result` ...
///     Ok(())
/// }
/// ```
pub struct Context<S> {
    server: Server,
    state: Arc<S>,
    device_id: i32,
}

impl<S> Context<S> {
    pub(crate) fn new(server: Server, state: Arc<S>, device_id: i32) -> Self {
        Self {
            server,
            state,
            device_id,
        }
    }

    /// The server to run BLS calls on, see [`Server::infer_async`].
    pub fn server(&self) -> &Server {
        &self.server
    }

    /// The state returned by [`crate::Backend::model_initialize`].
    pub fn state(&self) -> &S {
        &self.state
    }

    /// Device of the instance that runs the batch (GPU id for `KIND_GPU`).
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl<S> Clone for Context<S> {
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
            state: Arc::clone(&self.state),
            device_id: self.device_id,
        }
    }
}

/// Runtime of one model instance, with the batches it runs.
pub(crate) struct InstanceExecutor {
    runtime: Runtime,
    tasks: TaskTracker,
    /// One permit per batch in flight, `None` when the number is not limited.
    slots: Option<Arc<Semaphore>>,
}

impl InstanceExecutor {
    /// Creates the runtime of an instance of `model`, configured by the
    /// `worker_threads` and `max_batches_in_flight` model parameters.
    pub(crate) fn new(model: &Model, thread_name: String) -> Result<Self, Error> {
        let workers = model.config_parameter_or(WORKER_THREADS, 1usize)?;
        if workers == 0 {
            return Err(format!("{WORKER_THREADS} must be at least 1").into());
        }
        let max_in_flight = model.config_parameter_or(MAX_BATCHES_IN_FLIGHT, 0usize)?;

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .thread_name(thread_name)
            .enable_all()
            .build()?;

        Ok(Self {
            runtime,
            tasks: TaskTracker::new(),
            slots: (max_in_flight > 0).then(|| Arc::new(Semaphore::new(max_in_flight))),
        })
    }

    /// Runs the batch `future` on this instance. Waits for a free slot first
    /// when `max_batches_in_flight` batches are already in progress, so Triton
    /// keeps the next requests in its queue.
    pub(crate) fn spawn(&self, future: impl Future<Output = Result<(), Error>> + Send + 'static) {
        let permit = self.slots.as_ref().and_then(|slots| {
            self.runtime
                .block_on(Arc::clone(slots).acquire_owned())
                .inspect_err(|e| log::error(&format!("failed to wait for a free batch slot: {e}")))
                .ok()
        });

        self.tasks.spawn_on(
            async move {
                if let Err(e) = future.await {
                    log::error(&format!("model instance execute failed: {e}"));
                }
                drop(permit);
            },
            self.runtime.handle(),
        );
    }

    /// Waits for every batch of this instance and stops its runtime.
    pub(crate) fn shutdown(self) {
        self.tasks.close();
        self.runtime.block_on(self.tasks.wait());
        self.runtime.shutdown_timeout(Duration::from_secs(5));
    }
}

/// Server of `model` and the device of the instance, as passed to batch tasks.
pub(crate) fn context<S: Send + Sync + 'static>(
    model: &Model,
    device_id: i32,
) -> Result<Context<S>, Error> {
    let state = Arc::clone(model.state::<Arc<S>>()?);
    Ok(Context::new(model.get_server()?, state, device_id))
}
