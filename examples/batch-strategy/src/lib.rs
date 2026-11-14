//! Minimal by-key batching strategy from heliar-k's original contribution in PR #34.
//! Accept the head of the queue, then include adjacent requests with the same key.

use triton_ng::{BatchStrategy, Error, Model, Parameter, RequestView};

/// The request parameter whose value groups the requests of a batch.
const BATCH_KEY: &str = "batch_key";

/// Groups the requests of a batch by `batch_key`, the key of the request at
/// the head of the queue first.
struct ByKey;

/// What the strategy keeps about the batch Triton is forming: the key of the
/// request that heads it.
struct Batch {
    /// Whether the head of the batch has been seen, which is what tells a
    /// request without `batch_key` from the state before the first one.
    started: bool,
    key: Option<String>,
}

#[triton_ng::batch_strategy]
impl BatchStrategy for ByKey {
    type Batch = Batch;

    fn initialize(_model: &Model) -> Result<Self, Error> {
        Ok(Self)
    }

    fn batch(&self) -> Batch {
        Batch {
            started: false,
            key: None,
        }
    }

    fn include(&self, batch: &mut Batch, request: &RequestView<'_>) -> Result<bool, Error> {
        let key = match request.parameter(BATCH_KEY)? {
            Some(Parameter::String(key)) => Some(key),
            _ => None,
        };
        if !batch.started {
            // The request at the head of the queue joins the batch whatever
            // its key: a strategy that refuses it forms no batch at all.
            batch.started = true;
            batch.key = key;
            return Ok(true);
        }
        Ok(key == batch.key)
    }
}
