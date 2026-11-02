use crate::TritonError;
use triton_ng_macros::triton_call;

/// The buckets of a histogram metric, `TRITONSERVER_MetricArgs` passed to
/// `TRITONSERVER_MetricNewWithArgs`, which copies them into the Prometheus
/// histogram (core `r25.09`, `MetricFamily::Add`).
///
/// Owns the args and deletes them on drop. Neither `Send` nor `Sync`: they
/// live only while a metric is made.
pub(crate) struct MetricArgs {
    ptr: *mut triton_ng_sys::TRITONSERVER_MetricArgs,
}

impl MetricArgs {
    pub(crate) fn histogram(buckets: &[f64]) -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONSERVER_MetricArgsNew(&mut _))?;
        let args = Self { ptr };
        triton_call!(triton_ng_sys::TRITONSERVER_MetricArgsSetHistogram(
            args.ptr,
            buckets.as_ptr(),
            buckets.len() as u64,
        ))?;
        Ok(args)
    }

    pub(crate) fn as_ptr(&self) -> *const triton_ng_sys::TRITONSERVER_MetricArgs {
        self.ptr
    }
}

impl Drop for MetricArgs {
    fn drop(&mut self) {
        if let Err(e) = triton_call!(triton_ng_sys::TRITONSERVER_MetricArgsDelete(self.ptr)) {
            crate::log::error(&format!("failed to delete metric args: {e}"));
        }
    }
}
