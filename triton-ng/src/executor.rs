use crate::error::Error;
use crate::model::{Model, SharedState};
use crate::server::Server;
use std::any::Any;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::sync::{AcquireError, Semaphore};
use tokio_util::task::TaskTracker;

/// Model parameter: worker threads of every instance's runtime.
const WORKER_THREADS: &str = "worker_threads";
/// Model parameter: batches an instance processes at once, unlimited when absent or 0.
const MAX_BATCHES_IN_FLIGHT: &str = "max_batches_in_flight";

/// What a batch task gets from its model: the server for BLS calls and the
/// state stored with [`Model::set_state`].
///
/// `Context` is `Send + Clone` and holds no Triton pointers, so it can be used
/// across `.await` and passed to other tasks.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///     let state = ctx.state::<State>()?;
///     let mut inference = InferenceRequest::new(ctx.server(), &state.encoder, -1)?;
///     // ... add the inputs of all `requests` as one batch ...
///     let result = ctx.server().infer_async(inference).await?;
///     // ... answer every request from its rows of `result` ...
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct Context {
    server: Server,
    state: Option<SharedState>,
    device_id: i32,
}

impl Context {
    /// The server to run BLS calls on, see [`Server::infer_async`].
    pub fn server(&self) -> &Server {
        &self.server
    }

    /// The state stored with [`Model::set_state`].
    ///
    /// Fails when the model has no state or it is not a `T`.
    pub fn state<T: Any>(&self) -> Result<&T, Error> {
        self.state
            .as_deref()
            .ok_or("model state is not initialized")?
            .downcast_ref::<T>()
            .ok_or_else(|| "model state has unexpected type".into())
    }

    /// Device of the instance that runs the batch (GPU id for `KIND_GPU`).
    pub fn device_id(&self) -> i32 {
        self.device_id
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
        Ok(Self::with_limits(workers, max_in_flight, thread_name)?)
    }

    /// `workers` must be at least 1; `max_in_flight` 0 means unlimited.
    fn with_limits(
        workers: usize,
        max_in_flight: usize,
        thread_name: String,
    ) -> std::io::Result<Self> {
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
    /// keeps the next requests in its queue. Fails, without running `future`,
    /// only when the slots are closed, which never happens before shutdown.
    pub(crate) fn spawn(
        &self,
        future: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), AcquireError> {
        let permit = match &self.slots {
            Some(slots) => Some(self.runtime.block_on(Arc::clone(slots).acquire_owned())?),
            None => None,
        };

        self.tasks.spawn_on(
            async move {
                future.await;
                drop(permit);
            },
            self.runtime.handle(),
        );
        Ok(())
    }

    /// Waits for every batch of this instance and stops its runtime.
    pub(crate) fn shutdown(self) {
        self.tasks.close();
        self.runtime.block_on(self.tasks.wait());
        self.runtime.shutdown_timeout(Duration::from_secs(5));
    }
}

/// Server of `model` and the device of the instance, as passed to batch tasks.
pub(crate) fn context(model: &Model, device_id: i32) -> Result<Context, Error> {
    Ok(Context {
        server: model.get_server()?,
        state: model.shared_state()?,
        device_id,
    })
}

#[cfg(test)]
mod tests {
    use super::InstanceExecutor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use tokio::sync::oneshot;

    fn hold(
        executor: &InstanceExecutor,
        active: &Arc<AtomicUsize>,
        started: &mpsc::Sender<()>,
    ) -> oneshot::Sender<()> {
        let (release, released) = oneshot::channel();
        let active = Arc::clone(active);
        let started = started.clone();
        executor
            .spawn(async move {
                active.fetch_add(1, Ordering::SeqCst);
                started.send(()).unwrap();
                released.await.unwrap();
                active.fetch_sub(1, Ordering::SeqCst);
            })
            .unwrap();
        release
    }

    #[test]
    fn batch_starts_only_after_a_batch_in_progress_finishes_when_all_slots_are_busy() {
        let executor = InstanceExecutor::with_limits(2, 2, "test".to_string()).unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let (started, started_rx) = mpsc::channel();
        let first = hold(&executor, &active, &started);
        let second = hold(&executor, &active, &started);
        started_rx.recv().unwrap();
        started_rx.recv().unwrap();

        let (seen, seen_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let active = Arc::clone(&active);
                executor
                    .spawn(async move {
                        seen.send(active.load(Ordering::SeqCst)).unwrap();
                    })
                    .unwrap();
            });
            first.send(()).unwrap();
        });

        assert_eq!(seen_rx.recv().unwrap(), 1);
        second.send(()).unwrap();
        executor.shutdown();
    }

    #[test]
    fn batches_are_not_limited_without_max_batches_in_flight() {
        let executor = InstanceExecutor::with_limits(1, 0, "test".to_string()).unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let (started, started_rx) = mpsc::channel();
        let releases: Vec<_> = (0..8).map(|_| hold(&executor, &active, &started)).collect();
        (0..8).for_each(|_| started_rx.recv().unwrap());

        assert_eq!(active.load(Ordering::SeqCst), 8);
        releases
            .into_iter()
            .for_each(|release| release.send(()).unwrap());
        executor.shutdown();
    }

    #[test]
    fn shutdown_waits_for_batches_in_progress() {
        let executor = InstanceExecutor::with_limits(1, 0, "test".to_string()).unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let (started, started_rx) = mpsc::channel();
        let release = hold(&executor, &active, &started);
        started_rx.recv().unwrap();

        std::thread::scope(|scope| {
            scope.spawn(|| executor.shutdown());
            release.send(()).unwrap();
        });

        assert_eq!(active.load(Ordering::SeqCst), 0);
    }
}
