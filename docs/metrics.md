# Metrics

A backend exports its own metrics through Triton: they appear next to Triton's metrics on its
Prometheus endpoint (`:8002/metrics`), with no exporter of the backend's own.

```rust ignore
struct State {
    answered: Counter,
    encoder_seconds: Histogram,
}

fn model_initialize(model: &Model) -> Result<(), Error> {
    let name = model.name()?;
    let labels = [("model", name.as_str())];
    model.set_state(State {
        answered: CounterFamily::new("asr_answered_total", "Requests answered")?
            .counter(&labels)?,
        encoder_seconds: HistogramFamily::new(
            "asr_encoder_seconds",
            "Duration of the BLS call to the encoder",
            &[0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )?
        .histogram(&labels)?,
    })?;
    Ok(())
}

async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let started = Instant::now();
    let result = ctx.server().infer_async(inference).await?;
    state.encoder_seconds.observe(started.elapsed().as_secs_f64())?;
    // ... answer the requests ...
    state.answered.increment(requests.len() as f64)?;
    Ok(())
}
```

**Kinds.** A family is a metric name with its kind, description and, for a histogram, buckets.
Every kind has its own types, so a counter cannot be set and a gauge cannot be observed:

| Family | Metric | Operations |
|---|---|---|
| `CounterFamily::new(name, description)` | `Counter` | `increment` by a finite value ≥ 0, `value` |
| `GaugeFamily::new(name, description)` | `Gauge` | `set`, `increment`, `decrement` by a finite value, `value` |
| `HistogramFamily::new(name, description, buckets)` | `Histogram` | `observe` a finite value |

A metric is one series of its family with its labels: `family.counter(&[("model", "asr")])`.
Metrics with the same labels share one value. A NaN or an infinity is refused, it would stick in
the value or the sum of a histogram for good.

**Names.** Everything Prometheus would refuse is refused with an `InvalidArg` error before it
reaches Triton: metric names other than `[a-zA-Z_:][a-zA-Z0-9_:]*`, label names other than
`[a-zA-Z_][a-zA-Z0-9_]*` or starting with `__`, a label given twice, NUL bytes, and buckets that
are empty, not strictly increasing or not finite. The `nv_` prefix belongs to Triton's own metrics
and is refused too: sharing a Triton family would let the backend free Triton's series. Prefix the
names with the name of the backend.

**One family per name.** Families of one name are one family in the process: every model of the
backend can create the families it needs in `model_initialize` and gets the same family. Asking
for an existing name with another kind, description or buckets fails with `AlreadyExists`.
Triton registers families of one name into one Prometheus family but counts the users of each
series per family, so two Triton families of one name would let one of them free a series the
other still updates; triton-ng never creates the second one.

**Lifetimes.** A metric keeps its family alive, so a family is never deleted while one of its
metrics is in use; Triton would invalidate the metric without synchronization with its updates.
The family is deleted with its last metric and family value. Every type is `Send + Sync`, and
`Clone` shares the same metric: keep them in the state of the model or of the instance. A series is
removed from the endpoint when its last metric is dropped, so the metrics of an unloaded model
disappear and start from zero when it is loaded again.

[`example/custom-backend`](https://github.com/rdcm/triton-ng/blob/main/example/custom-backend/src/lib.rs) counts answered requests, the
requests it holds and the duration of its BLS calls, labelled with the model; the integration
tests read them from `:8002/metrics`.

## Inference statistics of the model

Triton's own `nv_inference_*` metrics and `/v2/models/<model>/stats` only show what a backend
reports, and triton-ng reports every batch and every request:

- when a request is released, as one successful request when its final response went out
  without an error (`nv_inference_request_success`, `nv_inference_count` with its batch size),
  otherwise as a failed one (`nv_inference_request_failure`); the queue time is from its arrival
  in the queue of the model until `model_instance_execute` was called, the compute time from
  that call until the request was answered (`nv_inference_queue_duration_us`,
  `nv_inference_compute_infer_duration_us`; the input and output durations stay zero);
- when the batch is done, as one execution of as many elements as its requests had in their batch
  dimension (`nv_inference_exec_count`, `batch_stats` of `/v2/models/<model>/stats`).

`nv_inference_count / nv_inference_exec_count` is the average batch size of the model, and
the queue time tells how long requests wait for an instance: the two numbers to watch when
choosing `instance_group` count, see [Batching and concurrency](https://github.com/rdcm/triton-ng/blob/main/docs/batching_and_concurrency.md).
A rescheduled request (`Request::reschedule`) is reported when its last step is released.
