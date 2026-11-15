# About

[![crates.io](https://img.shields.io/crates/v/triton-ng.svg)](https://crates.io/crates/triton-ng)
[![docs.rs](https://img.shields.io/docsrs/triton-ng)](https://docs.rs/triton-ng)

A Rust SDK for writing custom backends for NVIDIA Triton Inference Server.
Backends run as shared libraries inside Triton. The project also includes an async gRPC client
for calling a running Triton server.

Targets **Triton 25.09 / backend API 1.19** and **Rust 1.99**. The API checks compatibility at
backend startup. The examples below describe the **current repository**, including additions
after 0.3.1; use the Git dependency until those additions are released. See
[CHANGELOG](https://github.com/rdcm/triton-ng/blob/main/CHANGELOG.md) for release boundaries.

| Crate | Description |
|---|---|
| [triton-ng](https://crates.io/crates/triton-ng) | Safe backend API |
| [triton-ng-client](https://crates.io/crates/triton-ng-client) | Async gRPC client |
| [triton-ng-sys](https://crates.io/crates/triton-ng-sys) | Raw FFI bindings |
| [triton-ng-macros](https://crates.io/crates/triton-ng-macros) | Backend and batching entry points |

## Getting started

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
triton-ng = { git = "https://github.com/rdcm/triton-ng.git" }
```

Implement one hook to call another model and return its output. This complete backend is also
[examples/minimal-backend](https://github.com/rdcm/triton-ng/tree/main/examples/minimal-backend),
the minimal backend example (the separate batching-strategy example is introduced below):

```rust ignore
use triton_ng::{Context, CustomBackend, DataType, Error, InferenceRequest, Request, Response};

struct MyBackend;

#[triton_ng::backend]
impl CustomBackend for MyBackend {
    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        for request in requests {
            let data = request.get_input("INPUT")?.as_bytes()?;

            let mut inference = InferenceRequest::new(ctx.server(), "minimal_backend_target", -1)?;
            inference.add_input_with_data("INPUT0", DataType::Fp32, &[1, 10], data)?;
            inference.add_requested_output("OUTPUT0")?;
            let result = match ctx.server().infer_async(inference).await {
                Ok(result) => result,
                Err(e) => {
                    request.respond_error_with_code(e.code(), &e.to_string())?;
                    continue;
                }
            };

            let mut response = Response::new(request)?;
            response
                .create_output("OUTPUT", DataType::Fp32, &[1, 10])?
                .write_fp32_vec(&result.output("OUTPUT0")?.as_fp32_vec()?)?;
            response.send()?;
        }
        Ok(())
    }
}
```

### gRPC client

Call the custom backend from your application using the gRPC client:

```rust ignore
use triton_ng_client::{InferInput, InferOptions, TritonClient, TritonClientConfig};

type ClientError = Box<dyn std::error::Error + Send + Sync>;

async fn call_example() -> Result<(), ClientError> {
    let client = TritonClient::new(TritonClientConfig::new("http://localhost:8001")).await?;
    let result = client.infer(
        "minimal_backend", None,
        [InferInput::fp32("INPUT", vec![1, 10], vec![1.0; 10])],
        ["OUTPUT"], InferOptions::default(),
    ).await?;
    for output in result.outputs {
        println!("{}: {:?}", output.name, output.data);
    }
    Ok(())
}
```

The examples below are independent functions or hooks; import their SDK types from
`triton_ng`. Combine state fields when using several examples in one backend.
`make build` compile-checks the Rust snippets in this README and every guide.

## Compose models with BLS

Forward an input to another model without copying it:

```rust ignore
async fn forward_input(ctx: &Context, request: &Request) -> Result<InferenceResult, Error> {
    let shared = request.get_input("INPUT")?.share()?;
    let mut call = InferenceRequest::new(ctx.server(), "minimal_backend_target", -1)?;
    call.add_input_with_buffer("INPUT0", DataType::Fp32, &[1, 10], shared)?;
    call.add_requested_output("OUTPUT0")?;
    Ok(ctx.server().infer_async(call).await?)
}
```

To replace an input prepared by another helper and request only the embedding:

```rust ignore
fn replace_audio(call: &mut InferenceRequest, samples: i64, audio: Vec<u8>) -> Result<(), Error> {
    call.remove_input("AUDIO")?;
    call.add_input_with_data("AUDIO", DataType::Fp32, &[1, samples], audio)?;
    call.remove_all_requested_outputs()?;
    call.add_requested_output("EMBEDDING")?;
    Ok(())
}
```

[Request editing and buffer lifetimes](https://github.com/rdcm/triton-ng/blob/main/docs/request_metadata_and_tracing.md#editing-a-bls-request-before-submission).

## Streaming

Return tokens as they become available. Configure the model with
`model_transaction_policy { decoupled: true }`:

```rust ignore
fn send_tokens(request: Request) -> Result<(), Error> {
    let mut sender = request.into_sender()?;
    for token in [10, 20, 30] {
        let mut response = sender.response()?;
        response.create_output("TOKEN", DataType::Int32, &[1])?.write_i32_vec(&[token])?;
        response.send()?;
    }
    sender.finish()?;
    Ok(())
}
```

Call a streaming model from another backend with `Server::infer_stream`, or from an
application with `TritonClient::infer_stream`. See [streaming and cancellation](https://github.com/rdcm/triton-ng/blob/main/docs/streaming.md)
for complete examples.

## Metrics and response statistics

Count completed requests on Triton's Prometheus endpoint. Create the counter once in
model initialization and call `count_request` from execute:

```rust ignore
fn model_initialize(model: &Model) -> Result<(), Error> {
    let name = model.name()?;
    let counter = CounterFamily::new("minimal_requests_total", "Requests handled")?
        .counter(&[("model", name.as_str())])?;
    model.set_state(counter)?;
    Ok(())
}

fn count_request(ctx: &Context) -> Result<(), Error> {
    ctx.state::<Counter>()?.increment(1.0)?;
    Ok(())
}
```

Request and batch statistics are automatic. Enable per-response statistics when needed:

```rust ignore
fn answer_with_statistics(request: Request) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    response.create_output("OUTPUT", DataType::Fp32, &[1, 10])?.write_fp32_vec(&[0.0; 10])?;
    response.enable_statistics();
    response.send()?;
    Ok(())
}
```

[Metric types, error responses, and statistics](https://github.com/rdcm/triton-ng/blob/main/docs/metrics.md).

## Tracing and metadata

Add a trace activity and connect a BLS call to its client request:

```rust ignore
async fn traced_call(ctx: &Context, request: &Request) -> Result<InferenceResult, Error> {
    request.trace()?.report_activity("decode")?;
    let mut call = InferenceRequest::new(ctx.server(), "minimal_backend_target", -1)?;
    call.set_parent_trace(request)?;
    if let Some(id) = request.id()? { call.set_id(&id)?; }
    call.set_parameter("language", "en")?;
    call.add_input_with_data("INPUT0", DataType::Fp32, &[1, 10], request.get_input("INPUT")?.as_bytes()?)?;
    Ok(ctx.server().infer_async(call).await?)
}
```

[Request IDs, sequence flags, priority, timeouts, and tracing](https://github.com/rdcm/triton-ng/blob/main/docs/request_metadata_and_tracing.md).

## Memory buffers

Choose memory for each output, then take the buffers back for reuse. This example takes a
prepared request for a model with three FP32 outputs of shape `[1, 10]`:

```rust ignore
async fn reuse_outputs(
    ctx: &Context,
    mut call: InferenceRequest,
    device_id: i32,
) -> Result<(RamBuffer, PinnedRamBuffer, GpuBuffer), Error> {
    call.add_requested_output_into("CPU_OUTPUT", ctx.memory().ram(40)?)?;
    call.add_requested_output_into("PINNED_OUTPUT", ctx.memory().pinned(40)?)?;
    call.add_requested_output_into("GPU_OUTPUT", ctx.memory().gpu(device_id, 40)?)?;
    let mut result = ctx.server().infer_async(call).await?;
    Ok((
        result.take_ram("CPU_OUTPUT")?,
        result.take_pinned("PINNED_OUTPUT")?,
        result.take_gpu("GPU_OUTPUT")?,
    ))
}
```

Use any one of these buffer types, or combine them as above. Pinned and GPU allocations require
Triton's corresponding memory pools. A `GpuBuffer` can become the next BLS call's input without
a host copy; CPU access to its data requires your own CUDA code.

Report memory held by your model so clients can inspect it in model statistics:

```rust ignore
fn model_initialize(model: &Model) -> Result<(), Error> {
    let allocation = model.memory_manager()?.pinned(4096)?;
    model.set_state(allocation)?;
    model.report_memory_usage(&[BufferAttributes::new(MemoryType::CpuPinned, 4096)?])?;
    Ok(())
}
```

For per-instance allocations use `ModelInstance::report_memory_usage`.
[Memory ownership, GPU chaining, and caching](https://github.com/rdcm/triton-ng/blob/main/docs/memory_buffers.md).

## Configure backend and instances

Keep a setting shared by every model of your backend:

```rust ignore
fn initialize(backend: &Backend) -> Result<(), Error> {
    let language = backend.config()?.parameter_or("language", String::from("en"))?;
    backend.set_state(language)?;
    backend.set_execution_policy(ExecutionPolicy::Blocking)?;
    let (_, directory) = backend.artifacts()?;
    println!("Backend files: {}", directory.display());
    Ok(())
}

fn language(ctx: &Context) -> Result<&str, Error> {
    Ok(ctx.backend_state::<String>()?.as_str())
}
```

Inspect an instance and save its host policy for later input reads:

```rust ignore
fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
    println!("{} passive={} profiles={:?} secondary={:?}",
        instance.name()?, instance.is_passive()?, instance.profile_names()?, instance.secondary_devices()?);
    instance.set_state(instance.host_policy()?)?;
    Ok(())
}
```

[Backend configuration, execution policies, and instance settings](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md).

### Host-policy input data

Read the input storage selected for this instance:

```rust ignore
fn read_policy(ctx: &Context, request: &Request) -> Result<Vec<u8>, Error> {
    let policy = ctx.instance_state::<HostPolicy>()?;
    request.get_input("INPUT")?.as_bytes_for_host_policy(policy.name())
}
```

Supply different input bytes for a BLS target's host policy:

```rust ignore
fn policy_call(ctx: &Context, default: Vec<u8>, local: &[u8]) -> Result<InferenceRequest, Error> {
    let mut call = InferenceRequest::new(ctx.server(), "minimal_backend_target", -1)?;
    call.add_input_with_data("INPUT0", DataType::Fp32, &[1, 10], default)?;
    call.append_input_data_with_host_policy("INPUT0", local, "cpu")?;
    Ok(call)
}
```

Each input above contains ten FP32 values (40 bytes). A policy without supplied data uses
the ordinary input. [Policy selection and zero-copy appends](https://github.com/rdcm/triton-ng/blob/main/docs/request_metadata_and_tracing.md#appending-policy-specific-input-data).

## Sequence state and iterative requests

Count requests in a sequence. Declare a state tensor with an initial value of zero,
read its `StateConfig` with `Model::sequence_states`, then update it on each request:

```rust ignore
fn increment(mut request: Request, config: &StateConfig) -> Result<(), Error> {
    let previous = request.get_input(config.input_name())?.as_fp32_vec()?;
    let previous = previous.first().copied().ok_or("state has no counter")?;
    let mut state = State::new(&mut request, config, &[1])?;
    state.write_bytes(&(previous + 1.0).to_ne_bytes())?;
    state.update()?;
    let mut response = Response::new(request)?;
    response.create_output("OUTPUT", DataType::Fp32, &[1])?.write_fp32_vec(&[previous + 1.0])?;
    response.send()?;
    Ok(())
}
```

This example uses `max_batch_size: 0` and state dimensions `[1]`.
[Complete sequence-state configuration](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md#sequence-state).

Return a partial result and schedule another step of a decoupled iterative sequence:

```rust ignore
fn next_step(request: Request) -> Result<(), Error> {
    let mut sender = request.into_sender()?;
    let mut response = sender.response()?;
    response.create_output("STEP", DataType::Int32, &[1])?.write_i32_vec(&[1])?;
    response.send()?;
    sender.reschedule()?;
    Ok(())
}
```

Triton r25.09 allows only one reschedule per request; finish the response on the second step.
[Two-step execution and model configuration](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md#rescheduling-requests).

## Custom batching

Group adjacent requests with the same `batch_key` parameter:

```rust ignore
struct ByKey;
struct Batch { started: bool, key: Option<String> }

#[triton_ng::batch_strategy]
impl BatchStrategy for ByKey {
    type Batch = Batch;
    fn initialize(_model: &Model) -> Result<Self, Error> { Ok(Self) }
    fn batch(&self) -> Batch { Batch { started: false, key: None } }
    fn include(&self, batch: &mut Batch, request: &RequestView<'_>) -> Result<bool, Error> {
        let key = match request.parameter("batch_key")? {
            Some(Parameter::String(value)) => Some(value),
            _ => None,
        };
        if !batch.started {
            batch.started = true;
            batch.key = key;
            return Ok(true);
        }
        Ok(batch.key == key)
    }
}
```

Build this as a separate library and set `TRITON_BATCH_STRATEGY_PATH` in the model config.
[Runnable example](https://github.com/rdcm/triton-ng/tree/main/examples/batch-strategy) ·
[Queue behavior and safe model unload](https://github.com/rdcm/triton-ng/blob/main/docs/batching_and_concurrency.md).

## Comparison with python_backend

Against python_backend **r25.09**:

| Area | python_backend | triton-ng |
|---|---|---|
| Runtime | Python stub process per instance; shared-memory communication | Rust shared library in the Triton process |
| BLS | `exec` / `async_exec`, decoupled iterator | `infer_async` futures and `infer_stream` |
| Streaming | Response sender and final flags | Consuming `ResponseSender` and typed response ownership |
| Memory | NumPy, DLPack and CUDA IPC | Owned RAM/pinned/GPU buffers; shared request inputs |
| Isolation | A model crash generally affects its stub | Panics are contained; a native crash affects Triton |
| Best fit | Python/framework code inside the backend | Typed orchestration around Triton framework models |

[Full comparison](https://github.com/rdcm/triton-ng/blob/main/docs/python_backend.md), including
feature gaps and cases where Python is the better choice.

## Benchmarks

The recorded benchmark compares five small BLS workloads on the same Triton 25.09 server,
two CPU instances per backend, with a Ryzen 9 9950X3D. At concurrency 64:

| Workload | triton-ng requests/s | python_backend requests/s | Ratio |
|---|---:|---:|---:|
| Argmax, no BLS | 119,907 | 52,716 | 2.27× |
| One BLS call | 43,422 | 17,819 | 2.44× |
| Concurrent BLS calls | 74,316 | 13,317 | 5.58× |
| Decoupled responses | 12,802 | 3,670 | 3.49× |
| Consuming a BLS stream | 15,056 | 3,253 | 4.63× |

These are the recorded measurements, not a promise for large GPU models; heavy inference
reduces the backend's share of total time. Streaming latency in these results is time to first
response. [Method, environment, all concurrency levels, and latency tables](https://github.com/rdcm/triton-ng/blob/main/docs/benchmarks.md)
are in the benchmark guide.

## Documentation

Start with the topics you need; the guides and this README are also included in
[the Rust API documentation](https://docs.rs/triton-ng).

| Guide | What you will learn |
|---|---|
| [Execution model and thread safety](https://github.com/rdcm/triton-ng/blob/main/docs/execution_model_and_thread_safety.md) | Instances and their threads, state of models and instances, request ownership, error codes, BLS futures, threads, cancellation, shutdown, batching |
| [Batching and concurrency](https://github.com/rdcm/triton-ng/blob/main/docs/batching_and_concurrency.md) | How Triton forms batches and why a batch runs on the Triton thread of its instance, BLS calls joined inside a batch, tuning with `instance_group`, measurements |
| [Streaming](https://github.com/rdcm/triton-ng/blob/main/docs/streaming.md) | Decoupled models: streams of responses from a backend, BLS streams, cancellation of BLS calls, the client side, how it is implemented and its limitations |
| [Model parameters](https://github.com/rdcm/triton-ng/blob/main/docs/model_parameters.md) | The `parameters` map of a model config and how a backend reads it |
| [Request metadata and tracing](https://github.com/rdcm/triton-ng/blob/main/docs/request_metadata_and_tracing.md) | ID, correlation ID, sequence flags and timeout of a request; setting them, the priority and parameters on BLS calls; BLS calls as children of the trace of a request |
| [Metrics](https://github.com/rdcm/triton-ng/blob/main/docs/metrics.md) | Counters, gauges and histograms of the backend on Triton's Prometheus endpoint; the inference statistics triton-ng reports for every model |
| [Model management and configuration](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md) | The backend config from `--backend-config`, backend attributes (preferred instance groups, parallel instance loading), completing model configs, loading and unloading models from a backend, rescheduling requests of iterative sequences, the state a sequence keeps between its requests |
| [Memory buffers](https://github.com/rdcm/triton-ng/blob/main/docs/memory_buffers.md) | RAM, pinned and GPU buffers from Triton's memory manager for BLS inputs and outputs, caching them in a backend, passing inputs of a request on without a copy, reporting the memory a model and its instances hold |
| [C API coverage](https://github.com/rdcm/triton-ng/blob/main/docs/c_api_coverage.md) | All 276 C functions: Rust and Python coverage, public APIs, internal support, and missing wrappers |
| [Benchmarks](https://github.com/rdcm/triton-ng/blob/main/docs/benchmarks.md) | python_backend and triton-ng on the same BLS workloads with `perf_analyzer`: method, models, environment, results |
| [python_backend comparison](https://github.com/rdcm/triton-ng/blob/main/docs/python_backend.md) | Runtime differences, feature coverage, isolation, and when to use Python |

## Development

Install Rust, Docker, the NVIDIA Container Toolkit, and a compatible NVIDIA driver.
The test suite also requires `cargo-nextest`. From the repository root:

```bash
make build
make download-model
make docker-env-up
make tests
make docker-env-down
```

Wait until Triton reports the models ready before running the tests. After changing a
backend, rebuild and restart the service. See the [Makefile](https://github.com/rdcm/triton-ng/blob/main/Makefile) for all development
commands.

## License

MIT
