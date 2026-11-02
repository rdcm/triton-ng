//! Backend metrics, exported by Triton next to its own on the metrics
//! endpoint (`:8002/metrics`).
//!
//! A family is a metric name with its kind and description; a metric is one
//! series of it with its labels. Families of one name are one family in the
//! process, so every model can create the families it needs in
//! `model_initialize` and get the same family. A metric keeps its family
//! alive, and the family is deleted with its last metric and family value.
//! Every type is `Send + Sync + Clone`, cheap to clone, and can be kept in the
//! state of a model or instance.

use crate::error::{ErrorCode, TritonError};
use crate::metric::{Metric, validate_counter_increment, validate_finite};
use crate::metric_family::{FamilySpec, MetricFamily, MetricKind};
use std::sync::Arc;

fn family(
    name: &str,
    description: &str,
    kind: MetricKind,
) -> Result<Arc<MetricFamily>, TritonError> {
    MetricFamily::get_or_create(
        name,
        FamilySpec {
            kind,
            description: description.to_string(),
        },
    )
}

fn invalid_arg(message: String) -> TritonError {
    TritonError::with_code(ErrorCode::InvalidArg, message)
}

/// A family of counters: values that only grow, like the number of requests.
///
/// # Examples
///
/// ```ignore
/// let requests = CounterFamily::new("asr_requests_total", "Requests answered by the ASR backend")?;
/// let answered = requests.counter(&[("model", model.name()?.as_str()), ("status", "ok")])?;
/// answered.increment(1.0)?;
/// ```
#[derive(Clone)]
pub struct CounterFamily {
    family: Arc<MetricFamily>,
}

impl CounterFamily {
    /// The counter family `name`, created on first use.
    ///
    /// Fails when `name` is not a valid Prometheus name (`[a-zA-Z_:][a-zA-Z0-9_:]*`),
    /// starts with the `nv_` prefix of Triton's own metrics, or a family of
    /// this name already exists with another kind or description.
    pub fn new(name: &str, description: &str) -> Result<Self, TritonError> {
        Ok(Self {
            family: family(name, description, MetricKind::Counter)?,
        })
    }

    /// The counter of this family with `labels`, starting at zero. Counters
    /// with the same labels share one value.
    ///
    /// Fails when a label name is not a valid Prometheus label name or is
    /// given twice, or a value contains a NUL byte.
    pub fn counter(&self, labels: &[(&str, &str)]) -> Result<Counter, TritonError> {
        Ok(Counter {
            metric: Arc::new(Metric::new(&self.family, labels)?),
        })
    }
}

/// A counter of a [`CounterFamily`].
///
/// # Examples
///
/// ```ignore
/// state.answered.increment(requests.len() as f64)?;
/// ```
#[derive(Clone)]
pub struct Counter {
    metric: Arc<Metric>,
}

impl Counter {
    /// Adds `value` to the counter. Fails when `value` is negative, NaN or
    /// infinite.
    pub fn increment(&self, value: f64) -> Result<(), TritonError> {
        validate_counter_increment(value).map_err(invalid_arg)?;
        self.metric.increment(value)
    }

    /// The current value.
    pub fn value(&self) -> Result<f64, TritonError> {
        self.metric.value()
    }
}

/// A family of gauges: values that go up and down, like the number of
/// requests in flight.
///
/// # Examples
///
/// ```ignore
/// let in_flight = GaugeFamily::new("asr_batches_in_flight", "Batches the instance works on")?;
/// let gauge = in_flight.gauge(&[("model", "asr")])?;
/// gauge.increment(1.0)?;
/// // ... the batch ...
/// gauge.decrement(1.0)?;
/// ```
#[derive(Clone)]
pub struct GaugeFamily {
    family: Arc<MetricFamily>,
}

impl GaugeFamily {
    /// The gauge family `name`, created on first use. Fails like
    /// [`CounterFamily::new`].
    pub fn new(name: &str, description: &str) -> Result<Self, TritonError> {
        Ok(Self {
            family: family(name, description, MetricKind::Gauge)?,
        })
    }

    /// The gauge of this family with `labels`, starting at zero. Fails like
    /// [`CounterFamily::counter`].
    pub fn gauge(&self, labels: &[(&str, &str)]) -> Result<Gauge, TritonError> {
        Ok(Gauge {
            metric: Arc::new(Metric::new(&self.family, labels)?),
        })
    }
}

/// A gauge of a [`GaugeFamily`].
///
/// # Examples
///
/// ```ignore
/// state.cached_buffers.set(cache.len() as f64)?;
/// ```
#[derive(Clone)]
pub struct Gauge {
    metric: Arc<Metric>,
}

impl Gauge {
    /// Sets the gauge to `value`. Fails when `value` is NaN or infinite.
    pub fn set(&self, value: f64) -> Result<(), TritonError> {
        validate_finite(value).map_err(invalid_arg)?;
        self.metric.set(value)
    }

    /// Adds `value`, which may be negative. Fails when `value` is NaN or
    /// infinite.
    pub fn increment(&self, value: f64) -> Result<(), TritonError> {
        validate_finite(value).map_err(invalid_arg)?;
        self.metric.increment(value)
    }

    /// Subtracts `value`. Fails when `value` is NaN or infinite.
    pub fn decrement(&self, value: f64) -> Result<(), TritonError> {
        validate_finite(value).map_err(invalid_arg)?;
        self.metric.increment(-value)
    }

    /// The current value.
    pub fn value(&self) -> Result<f64, TritonError> {
        self.metric.value()
    }
}

/// A family of histograms: the distribution of observed values, like
/// latencies, counted into buckets.
///
/// # Examples
///
/// ```ignore
/// let latency = HistogramFamily::new(
///     "asr_encoder_seconds",
///     "Duration of the BLS call to the encoder",
///     &[0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
/// )?;
/// let encoder = latency.histogram(&[("model", "asr")])?;
/// ```
#[derive(Clone)]
pub struct HistogramFamily {
    family: Arc<MetricFamily>,
}

impl HistogramFamily {
    /// The histogram family `name` with the upper bounds `buckets`, created
    /// on first use; Prometheus adds the `+Inf` bucket.
    ///
    /// Fails like [`CounterFamily::new`], when the family exists with other
    /// buckets, and when `buckets` is empty, not strictly increasing or has
    /// a NaN or infinite bound.
    pub fn new(name: &str, description: &str, buckets: &[f64]) -> Result<Self, TritonError> {
        Ok(Self {
            family: family(name, description, MetricKind::Histogram(buckets.to_vec()))?,
        })
    }

    /// The histogram of this family with `labels`, with nothing observed.
    /// Fails like [`CounterFamily::counter`].
    pub fn histogram(&self, labels: &[(&str, &str)]) -> Result<Histogram, TritonError> {
        Ok(Histogram {
            metric: Arc::new(Metric::new(&self.family, labels)?),
        })
    }
}

/// A histogram of a [`HistogramFamily`].
///
/// # Examples
///
/// ```ignore
/// let started = Instant::now();
/// let result = ctx.server().infer_async(inference).await?;
/// state.encoder.observe(started.elapsed().as_secs_f64())?;
/// ```
#[derive(Clone)]
pub struct Histogram {
    metric: Arc<Metric>,
}

impl Histogram {
    /// Counts `value` into its bucket. Fails when `value` is NaN or
    /// infinite.
    pub fn observe(&self, value: f64) -> Result<(), TritonError> {
        validate_finite(value).map_err(invalid_arg)?;
        self.metric.observe(value)
    }
}
