use crate::{Error, Model, RequestView};
use std::sync::Arc;

/// Decides which requests Triton puts into one batch, for a model whose
/// config names a batching strategy library.
///
/// Triton calls [`BatchStrategy::initialize`] once when the model loads, then
/// [`BatchStrategy::batch`] when it starts forming a batch and
/// [`BatchStrategy::include`] for each queued request in turn, oldest first,
/// until the batch is full. Answering `false` ends the batch before that
/// request: it stays in the queue and can head the next batch, the requests
/// behind it wait with it. Triton asks the strategy only for a model with
/// `dynamic_batching`, and may start a batch and abandon it again before one
/// is handed to a model instance.
///
/// A strategy that answers `false` for the request at the head of the queue
/// forms no batch at all: Triton asks again, with the same request, and the
/// queue does not move. Only a timeout takes a request out of the queue, the
/// timeout the client set or the `default_queue_policy` of the model, so a
/// model whose config sets neither stays stalled for good.
///
/// Triton finalizes the batcher before it stops its batching thread, and it
/// does not finalize a batch it started but never formed when the model
/// unloads: that batch state, and the reference to the strategy it holds, is
/// leaked. Work that must happen belongs in the calls, not in a `Drop`.
///
/// The library exports the entry points of this trait through
/// [`macro@crate::batch_strategy`]; a crate that uses the attribute is the
/// batching strategy library of the model, one crate per strategy.
///
/// # Examples
///
/// ```ignore
/// struct ByKey;
///
/// struct Batch {
///     started: bool,
///     key: Option<String>,
/// }
///
/// #[triton_ng::batch_strategy]
/// impl BatchStrategy for ByKey {
///     type Batch = Batch;
///
///     fn initialize(model: &Model) -> Result<Self, Error> {
///         Ok(Self)
///     }
///
///     fn batch(&self) -> Batch {
///         Batch { started: false, key: None }
///     }
///
///     fn include(&self, batch: &mut Batch, request: &RequestView<'_>) -> Result<bool, Error> {
///         let key = match request.parameter("batch_key")? {
///             Some(Parameter::String(key)) => Some(key),
///             _ => None,
///         };
///         if !batch.started {
///             // The request at the head of the queue joins the batch whatever
///             // its key: a batch needs its head.
///             batch.started = true;
///             batch.key = key;
///             return Ok(true);
///         }
///         Ok(key == batch.key)
///     }
/// }
/// ```
pub trait BatchStrategy: Sized + Send + Sync + 'static {
    /// What the strategy keeps about the batch it is forming.
    ///
    /// One is created when Triton starts a batch and dropped when the batch
    /// is handed to the model instance, so it never sees the batch execute
    /// and cannot report what the batch turned out to be.
    type Batch: Send;

    /// Reads the config of `model` and returns the strategy Triton keeps for
    /// it, until it unloads the model. Called once, on the thread that loads
    /// the model.
    ///
    /// Fails when the config of the model does not fit the strategy; Triton
    /// then fails to load the model.
    fn initialize(model: &Model) -> Result<Self, Error>;

    /// A batch about to be formed. Called on Triton's batching thread when
    /// the previous batch is complete, possibly several times before one is
    /// handed to a model instance.
    fn batch(&self) -> Self::Batch;

    /// Whether `request` joins `batch`. Called on Triton's batching thread
    /// for each queued request, oldest first.
    ///
    /// Fails when the request cannot be read. Triton logs the error and ends
    /// the batch before the request, as it does for `false`: at the head of
    /// the queue, a failing `include` keeps the model from forming a batch.
    fn include(&self, batch: &mut Self::Batch, request: &RequestView<'_>) -> Result<bool, Error>;
}

/// A batch Triton is forming, as the batching strategy library keeps it: the
/// strategy, which [`BatchStrategy::include`] needs, and the batch state.
pub(crate) struct BatchState<S: BatchStrategy> {
    pub(crate) strategy: Arc<S>,
    pub(crate) batch: S::Batch,
}
