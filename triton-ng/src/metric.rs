use crate::error::{ErrorCode, TritonError};
use crate::metric_args::MetricArgs;
use crate::metric_family::{FamilyHandle, MetricKind, validate_label_name};
use crate::metric_label::MetricLabel;
use std::collections::BTreeSet;
use std::sync::Arc;
use triton_ng_macros::triton_call;

/// A `TRITONSERVER_Metric`: one labelled time series of a family.
///
/// Owns the metric and deletes it on drop, before it lets its family go:
/// Triton invalidates the metrics of a deleted family without
/// synchronization with their updates, so the family must outlive them.
/// `Send + Sync`.
pub(crate) struct MetricHandle {
    ptr: *mut triton_ng_sys::TRITONSERVER_Metric,
    /// Dropped after `Drop` deletes the metric.
    family: Arc<FamilyHandle>,
}

// SAFETY: the Prometheus counter, gauge and histogram behind a metric are
// updated and read atomically or under their own lock, so a metric can be
// used from many threads at once. Triton reads the family pointer of the
// metric unsynchronized, but it changes only when the family is deleted,
// which `family` prevents.
unsafe impl Send for MetricHandle {}
unsafe impl Sync for MetricHandle {}

impl MetricHandle {
    pub(crate) fn new(
        family: &Arc<FamilyHandle>,
        labels: &[(&str, &str)],
    ) -> Result<Self, TritonError> {
        validate_labels(labels).map_err(|e| {
            TritonError::with_code(
                ErrorCode::InvalidArg,
                format!("metric of family '{}': {e}", family.name()),
            )
        })?;

        let labels = labels
            .iter()
            .map(|(name, value)| MetricLabel::new(name, value))
            .collect::<Result<Vec<_>, _>>()?;
        let mut label_ptrs: Vec<_> = labels.iter().map(MetricLabel::as_ptr).collect();

        let args = match family.kind() {
            MetricKind::Histogram(buckets) => Some(MetricArgs::histogram(buckets)?),
            MetricKind::Counter | MetricKind::Gauge => None,
        };
        let args_ptr = args.as_ref().map_or(std::ptr::null(), MetricArgs::as_ptr);

        let ptr = triton_call!(triton_ng_sys::TRITONSERVER_MetricNewWithArgs(
            &mut _,
            family.as_ptr(),
            label_ptrs.as_mut_ptr(),
            label_ptrs.len() as u64,
            args_ptr,
        ))?;
        Ok(Self {
            ptr,
            family: Arc::clone(family),
        })
    }

    pub(crate) fn increment(&self, value: f64) -> Result<(), TritonError> {
        triton_call!(triton_ng_sys::TRITONSERVER_MetricIncrement(self.ptr, value))
    }

    pub(crate) fn set(&self, value: f64) -> Result<(), TritonError> {
        triton_call!(triton_ng_sys::TRITONSERVER_MetricSet(self.ptr, value))
    }

    pub(crate) fn observe(&self, value: f64) -> Result<(), TritonError> {
        triton_call!(triton_ng_sys::TRITONSERVER_MetricObserve(self.ptr, value))
    }

    pub(crate) fn value(&self) -> Result<f64, TritonError> {
        let mut value = 0.0;
        triton_call!(triton_ng_sys::TRITONSERVER_MetricValue(
            self.ptr, &mut value
        ))?;
        Ok(value)
    }
}

impl Drop for MetricHandle {
    fn drop(&mut self) {
        if let Err(e) = triton_call!(triton_ng_sys::TRITONSERVER_MetricDelete(self.ptr)) {
            crate::log::error(&format!(
                "failed to delete a metric of family '{}': {e}",
                self.family.name()
            ));
        }
    }
}

/// Label names must be valid Prometheus names and distinct: Triton keeps
/// only the last value of a repeated name. Values may be any string without
/// a NUL byte.
fn validate_labels(labels: &[(&str, &str)]) -> Result<(), String> {
    let mut names = BTreeSet::new();
    for (name, value) in labels {
        validate_label_name(name)?;
        if value.contains('\0') {
            return Err(format!("value of label '{name}' contains a NUL byte"));
        }
        if !names.insert(*name) {
            return Err(format!("label '{name}' is given twice"));
        }
    }
    Ok(())
}

/// Values a counter may be incremented by: finite and not negative.
pub(crate) fn validate_counter_increment(value: f64) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "a counter is incremented by a finite non-negative value, not {value}"
        ));
    }
    Ok(())
}

/// Values a gauge or histogram takes: finite. NaN or an infinity would stick
/// in the sum or value for good.
pub(crate) fn validate_finite(value: f64) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("metric value {value} is not finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_counter_increment, validate_finite, validate_labels};

    #[test]
    fn distinct_valid_labels_are_accepted() {
        assert!(validate_labels(&[]).is_ok());
        assert!(validate_labels(&[("model", "encoder"), ("gpu", "0")]).is_ok());
        assert!(validate_labels(&[("model", "")]).is_ok());
    }

    #[test]
    fn repeated_or_invalid_labels_are_refused() {
        assert!(validate_labels(&[("model", "a"), ("model", "b")]).is_err());
        assert!(validate_labels(&[("bad-name", "a")]).is_err());
        assert!(validate_labels(&[("model", "a\0b")]).is_err());
    }

    #[test]
    fn counters_only_grow_by_finite_values() {
        assert!(validate_counter_increment(0.0).is_ok());
        assert!(validate_counter_increment(2.5).is_ok());
        for value in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(validate_counter_increment(value).is_err(), "{value}");
        }
    }

    #[test]
    fn gauges_and_histograms_take_finite_values() {
        assert!(validate_finite(-3.0).is_ok());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(validate_finite(value).is_err(), "{value}");
        }
    }
}
