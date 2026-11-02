use crate::error::{ErrorCode, TritonError};
use crate::utils::cstring_from_str;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use triton_ng_macros::triton_call;

/// What a metric family counts, with the buckets of a histogram.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MetricKind {
    Counter,
    Gauge,
    Histogram(Vec<f64>),
}

impl MetricKind {
    fn to_sys(&self) -> triton_ng_sys::TRITONSERVER_MetricKind {
        match self {
            Self::Counter => {
                triton_ng_sys::TRITONSERVER_metrickind_enum_TRITONSERVER_METRIC_KIND_COUNTER
            }
            Self::Gauge => {
                triton_ng_sys::TRITONSERVER_metrickind_enum_TRITONSERVER_METRIC_KIND_GAUGE
            }
            Self::Histogram(_) => {
                triton_ng_sys::TRITONSERVER_metrickind_enum_TRITONSERVER_METRIC_KIND_HISTOGRAM
            }
        }
    }
}

/// Everything two families of the same name must agree on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FamilySpec {
    pub(crate) kind: MetricKind,
    pub(crate) description: String,
}

/// A `TRITONSERVER_MetricFamily`, shared by every family value of its name
/// in the process.
///
/// Families of one name must be one Triton object: Triton registers them all
/// into the same Prometheus family, but counts the users of each Prometheus
/// metric per Triton family, so deleting a metric of one family would free
/// the metric with the same labels that another family still uses (core
/// `r25.09`, `MetricFamily::Remove`).
///
/// Owns the family and deletes it on drop, once no metric holds it: a metric
/// of a deleted family is invalidated without synchronization with its
/// updates. `Send + Sync`.
pub(crate) struct FamilyHandle {
    ptr: *mut triton_ng_sys::TRITONSERVER_MetricFamily,
    name: String,
    spec: FamilySpec,
}

// SAFETY: Triton adds and removes the metrics of a family under the locks of
// the family and of its Prometheus family, so metrics of one family can be
// made and deleted from any threads at once. The family itself is deleted
// only by the last owner.
unsafe impl Send for FamilyHandle {}
unsafe impl Sync for FamilyHandle {}

/// The live families by name, see `FamilyHandle`. Holds no strong
/// references, so a family is deleted with its last user.
static FAMILIES: Mutex<BTreeMap<String, Weak<FamilyHandle>>> = Mutex::new(BTreeMap::new());

fn families() -> MutexGuard<'static, BTreeMap<String, Weak<FamilyHandle>>> {
    // The map holds only weak references, it is consistent even after a
    // panic of another thread.
    FAMILIES.lock().unwrap_or_else(PoisonError::into_inner)
}

impl FamilyHandle {
    /// The family `name`: the live one, when its spec is the same, or a new
    /// one.
    pub(crate) fn get_or_create(name: &str, spec: FamilySpec) -> Result<Arc<Self>, TritonError> {
        validate_family(name, &spec)
            .map_err(|e| TritonError::with_code(ErrorCode::InvalidArg, e))?;

        let mut families = families();
        if let Some(existing) = families.get(name).and_then(Weak::upgrade) {
            // The last reference must not be dropped under the lock: `Drop`
            // takes it.
            drop(families);
            if existing.spec != spec {
                return Err(TritonError::with_code(
                    ErrorCode::AlreadyExists,
                    format!(
                        "metric family '{name}' already exists as {:?}, not {spec:?}",
                        existing.spec
                    ),
                ));
            }
            return Ok(existing);
        }

        let name_cstr = cstring_from_str(name)?;
        let description_cstr = cstring_from_str(&spec.description)?;
        let ptr = triton_call!(triton_ng_sys::TRITONSERVER_MetricFamilyNew(
            &mut _,
            spec.kind.to_sys(),
            name_cstr.as_ptr(),
            description_cstr.as_ptr(),
        ))?;
        let family = Arc::new(Self {
            ptr,
            name: name.to_string(),
            spec,
        });
        families.insert(name.to_string(), Arc::downgrade(&family));
        Ok(family)
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONSERVER_MetricFamily {
        self.ptr
    }

    pub(crate) fn kind(&self) -> &MetricKind {
        &self.spec.kind
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for FamilyHandle {
    fn drop(&mut self) {
        {
            // A family of the same name made after this one lost its last
            // reference is alive and stays registered.
            let mut families = families();
            if families
                .get(&self.name)
                .is_some_and(|family| family.strong_count() == 0)
            {
                families.remove(&self.name);
            }
        }
        if let Err(e) = triton_call!(triton_ng_sys::TRITONSERVER_MetricFamilyDelete(self.ptr)) {
            crate::log::error(&format!(
                "failed to delete metric family '{}': {e}",
                self.name
            ));
        }
    }
}

/// Checks what Triton would refuse by throwing from inside Prometheus, so
/// that the backend gets an error naming the problem.
pub(crate) fn validate_family(name: &str, spec: &FamilySpec) -> Result<(), String> {
    validate_metric_name(name)?;
    if name.starts_with("nv_") {
        return Err(format!(
            "metric family '{name}': the 'nv_' prefix is reserved for the metrics of Triton"
        ));
    }
    if spec.description.contains('\0') {
        return Err(format!(
            "description of metric family '{name}' contains a NUL byte"
        ));
    }
    if let MetricKind::Histogram(buckets) = &spec.kind {
        validate_buckets(buckets).map_err(|e| format!("metric family '{name}': {e}"))?;
    }
    Ok(())
}

/// Prometheus metric names: `[a-zA-Z_:][a-zA-Z0-9_:]*`.
fn validate_metric_name(name: &str) -> Result<(), String> {
    let mut chars = name.chars();
    let valid = chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_' || first == ':')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':');
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid metric family name '{name}', expected [a-zA-Z_:][a-zA-Z0-9_:]*"
        ))
    }
}

/// Prometheus label names: `[a-zA-Z_][a-zA-Z0-9_]*`, without the reserved
/// `__` prefix.
pub(crate) fn validate_label_name(name: &str) -> Result<(), String> {
    let mut chars = name.chars();
    let valid = chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.starts_with("__");
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid label name '{name}', expected [a-zA-Z_][a-zA-Z0-9_]* not starting with '__'"
        ))
    }
}

/// Histogram buckets: upper bounds, finite and strictly increasing; the
/// `+Inf` bucket is always added.
fn validate_buckets(buckets: &[f64]) -> Result<(), String> {
    if buckets.is_empty() {
        return Err("a histogram needs at least one bucket".to_string());
    }
    if let Some(bucket) = buckets.iter().find(|bucket| !bucket.is_finite()) {
        return Err(format!("histogram bucket {bucket} is not finite"));
    }
    if let Some(pair) = buckets.windows(2).find(|pair| pair[0] >= pair[1]) {
        return Err(format!(
            "histogram buckets must be strictly increasing, {} is followed by {}",
            pair[0], pair[1]
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{FamilySpec, MetricKind, validate_family, validate_label_name};

    fn spec(kind: MetricKind) -> FamilySpec {
        FamilySpec {
            kind,
            description: "requests".to_string(),
        }
    }

    #[test]
    fn prometheus_metric_names_are_accepted() {
        for name in ["requests_total", "_hidden", "asr:latency_seconds", "a1"] {
            assert!(
                validate_family(name, &spec(MetricKind::Counter)).is_ok(),
                "{name}"
            );
        }
    }

    #[test]
    fn invalid_metric_names_are_refused() {
        for name in [
            "",
            "1st",
            "with-dash",
            "with space",
            "привет",
            "nv_inference_count",
        ] {
            assert!(
                validate_family(name, &spec(MetricKind::Gauge)).is_err(),
                "{name}"
            );
        }
    }

    #[test]
    fn description_with_nul_is_refused() {
        let spec = FamilySpec {
            kind: MetricKind::Counter,
            description: "bad\0".to_string(),
        };
        assert!(validate_family("requests_total", &spec).is_err());
    }

    #[test]
    fn label_names_follow_prometheus() {
        for name in ["model", "_x", "gpu0"] {
            assert!(validate_label_name(name).is_ok(), "{name}");
        }
        for name in ["", "0gpu", "__name", "a-b", "a:b"] {
            assert!(validate_label_name(name).is_err(), "{name}");
        }
    }

    #[test]
    fn histogram_buckets_are_finite_and_strictly_increasing() {
        let histogram = |buckets: &[f64]| {
            validate_family(
                "latency_seconds",
                &spec(MetricKind::Histogram(buckets.to_vec())),
            )
        };
        assert!(histogram(&[0.01, 0.1, 1.0]).is_ok());
        assert!(histogram(&[-1.0]).is_ok());
        assert!(histogram(&[]).is_err());
        assert!(histogram(&[0.1, 0.1]).is_err());
        assert!(histogram(&[1.0, 0.1]).is_err());
        assert!(histogram(&[0.1, f64::NAN]).is_err());
        assert!(histogram(&[0.1, f64::INFINITY]).is_err());
    }
}
