use crate::error::TritonError;
use triton_ng_macros::triton_call;

/// Reports the statistics of one `TRITONBACKEND_ModelInstanceExecute` call
/// to Triton: those of a request when it is released, those of the batch
/// when it is done. They are what `nv_inference_*` metrics and
/// `/v2/models/<model>/stats` show for the model.
///
/// Borrows the `TRITONBACKEND_ModelInstance`: a model is not finalized while
/// a request of it is in flight, so the instance outlives every request of
/// the batch. `Send + Sync`.
pub(crate) struct BatchStatistics {
    instance: *mut triton_ng_sys::TRITONBACKEND_ModelInstance,
    exec_start_ns: u64,
}

// SAFETY: the instance is only given to Triton's reporting functions, which
// may be called from any thread (core backend_model_instance.cc,
// ModelInstanceReportStatistics: the aggregator takes its own mutex).
unsafe impl Send for BatchStatistics {}
unsafe impl Sync for BatchStatistics {}

impl BatchStatistics {
    /// Starts the clock of a batch of `instance`.
    pub(crate) fn new(instance: *mut triton_ng_sys::TRITONBACKEND_ModelInstance) -> Self {
        Self {
            instance,
            exec_start_ns: now_ns(),
        }
    }

    /// Reports `request` as done, with the time since the batch started as
    /// its compute time. Triton needs the request alive: call it before the
    /// request is released.
    pub(crate) fn report_request(
        &self,
        request: *mut triton_ng_sys::TRITONBACKEND_Request,
        success: bool,
    ) -> Result<(), TritonError> {
        let end_ns = now_ns();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceReportStatistics(
            self.instance,
            request,
            success,
            self.exec_start_ns,
            self.exec_start_ns,
            end_ns,
            end_ns,
        ))
    }

    /// Reports the batch as done: one execution of `batch_size` elements,
    /// the sum of the batch sizes of its requests.
    pub(crate) fn report_batch(&self, batch_size: u64) -> Result<(), TritonError> {
        let end_ns = now_ns();
        triton_call!(
            triton_ng_sys::TRITONBACKEND_ModelInstanceReportBatchStatistics(
                self.instance,
                batch_size,
                self.exec_start_ns,
                self.exec_start_ns,
                end_ns,
                end_ns,
            )
        )
    }

    /// The instance that runs the batch, for the statistics of a response.
    pub(crate) fn instance(&self) -> *mut triton_ng_sys::TRITONBACKEND_ModelInstance {
        self.instance
    }
}

/// Nanoseconds of the clock Triton stamps its requests with:
/// `std::chrono::steady_clock`, which is `CLOCK_MONOTONIC` on Linux.
pub(crate) fn now_ns() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `time` is a plain struct that clock_gettime fills; it cannot
    // fail for CLOCK_MONOTONIC on Linux and leaves zeros otherwise.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    u64::try_from(time.tv_sec)
        .unwrap_or(0)
        .saturating_mul(1_000_000_000)
        .saturating_add(u64::try_from(time.tv_nsec).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::now_ns;

    #[test]
    fn the_clock_does_not_go_back() {
        let first = now_ns();
        let second = now_ns();
        assert!(first > 0 && second >= first);
    }
}
