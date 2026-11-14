# About

[![crates.io](https://img.shields.io/crates/v/triton-ng.svg)](https://crates.io/crates/triton-ng)
[![docs.rs](https://img.shields.io/docsrs/triton-ng)](https://docs.rs/triton-ng)

Write NVIDIA Triton backends in Rust, or call a running Triton server with the async gRPC client.
A backend runs as a shared library inside Triton: use BLS to compose models, stream responses,
keep data on the GPU between calls, and let Rust types enforce request and response ownership.

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

The remaining backend snippets are small hook/function bodies. Import the SDK types they use
from `triton_ng`. State examples are independent: combine their fields in your own model or
instance state when using several features. Each snippet is compiled against the SDK to keep it aligned with the API.

## Compose models with BLS

`Server::infer_async` submits immediately and returns a future. Joining futures allows the BLS
calls of a batch to run together; batches of one instance still execute sequentially on its
Triton thread. Dropping a pending future requests cancellation.

An input can be shared without copying. The SDK holds back the parent's final response until
the last shared input is dropped; its lifetime does not depend on the BLS response arriving.

```rust ignore
async fn forward_input(ctx: &Context, request: &Request) -> Result<InferenceResult, Error> {
    let shared = request.get_input("INPUT")?.share()?;
    let mut call = InferenceRequest::new(ctx.server(), "minimal_backend_target", -1)?;
    call.add_input_with_buffer("INPUT0", DataType::Fp32, &[1, 10], shared)?;
    call.add_requested_output("OUTPUT0")?;
    Ok(ctx.server().infer_async(call).await?)
}
```

### Edit a request before submission

Helpers can add and remove inputs or requested outputs. Removing a missing input returns
`INVALID_ARG`; removing a missing requested output succeeds. An empty requested-output list
selects **all outputs**. Removed input owners stay alive until request drop/release; removed
SDK output buffers are dropped immediately.

```rust ignore
fn rebuild_call(ctx: &Context) -> Result<InferenceRequest, Error> {
    let mut call = InferenceRequest::new(ctx.server(), "minimal_backend_target", -1)?;
    call.add_input_with_data("INPUT0", DataType::Fp32, &[1, 10], vec![0; 40])?;
    call.remove_input("INPUT0")?;
    call.remove_all_inputs()?; // Already empty is fine.
    call.add_input_with_data("INPUT0", DataType::Fp32, &[1, 10], vec![0; 40])?;
    call.add_requested_output("OUTPUT0")?;
    call.remove_requested_output("OUTPUT0")?;
    call.remove_all_requested_outputs()?;
    Ok(call)
}
```

## Streaming

A decoupled model declares `model_transaction_policy { decoupled: true }`. Each request owns
one `ResponseSender`: send any number of responses, then send the final flag exactly once.

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

BLS can consume that model's stream. `next` is available directly on `InferenceStream`:

```rust ignore
async fn read_tokens(ctx: &Context) -> Result<(), Error> {
    let mut call = InferenceRequest::new(ctx.server(), "tokens", -1)?;
    call.add_input_with_data("INPUT0", DataType::Fp32, &[1, 10], vec![0; 40])?;
    let mut stream = ctx.server().infer_stream(call);
    while let Some(result) = stream.next().await {
        println!("{:?}", result?.output("TOKEN")?.as_i32_vec()?);
    }
    Ok(())
}
```

The gRPC client's `infer_stream(...).await?` exposes the same response sequence to applications;
its stream is consumed with `futures::StreamExt`. Drop a BLS stream or call `cancel()` to request
cancellation. Cancellation is cooperative: check `request.is_cancelled()` between work steps.
A normal `infer_async` call to a decoupled model rejects zero or multiple responses.

## Metrics and response statistics

Counters, gauges, and histograms are exported on Triton's Prometheus endpoint. Create metric
families once, keep the metric in model/instance state, and update it from execute:

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

Request and batch inference statistics are reported automatically. **Response** statistics are
opt-in, independently for each response; enabling twice does not report twice:

```rust ignore
fn answer_with_statistics(request: Request) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    response.create_output("OUTPUT", DataType::Fp32, &[1, 10])?.write_fp32_vec(&[0.0; 10])?;
    response.enable_statistics();
    response.send()?;
    Ok(())
}

fn reject_with_statistics(mut request: Request) -> Result<(), Error> {
    request.enable_statistics();
    request.respond_error_with_code(ErrorCode::InvalidArg, "invalid input")?;
    Ok(())
}
```

## Tracing and metadata

Preserve a client's ID and attach BLS work to the parent's trace. A custom activity is recorded
only when tracing is enabled; invalid collector names are rejected instead of corrupting JSON.

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

Correlation IDs, sequence flags, timeouts, priority, and typed parameters are also available.
The [metadata and tracing guide](https://github.com/rdcm/triton-ng/blob/main/docs/request_metadata_and_tracing.md)
explains tracing ownership and the r25.09 validation-failure limitation.

## Memory and instance configuration

Allocate RAM, pinned RAM, or GPU memory through `ctx.memory()`. `add_requested_output_into`
lets a BLS model write into a supplied buffer; take the written buffer from the result for reuse:

```rust ignore
async fn reuse_output(ctx: &Context) -> Result<RamBuffer, Error> {
    let mut call = InferenceRequest::new(ctx.server(), "minimal_backend_target", -1)?;
    call.add_input_with_data("INPUT0", DataType::Fp32, &[1, 10], vec![0; 40])?;
    call.add_requested_output_into("OUTPUT0", ctx.memory().ram(40)?)?;
    let mut result = ctx.server().infer_async(call).await?;
    result.take_ram("OUTPUT0")
}
```

GPU output buffers are handed back with `take_gpu` and can become another BLS input without a
host copy. Reading GPU data on the CPU requires your CUDA code; the SDK does not dereference
device pointers or depend on the CUDA runtime.

### Report model and instance memory

Reports describe allocations you own; reporting does not allocate memory. Keep reported memory
alive in state and replace the report when its usage changes:

```rust ignore
fn model_initialize(model: &Model) -> Result<(), Error> {
    let allocation = model.memory_manager()?.pinned(4096)?;
    model.set_state(allocation)?;
    model.report_memory_usage(&[BufferAttributes::new(MemoryType::CpuPinned, 4096)?])?;
    Ok(())
}
```

`ModelInstance::report_memory_usage` does the same for one instance. Triton aggregates records
by memory type and ID. Metadata such as passive-instance status, optimization profiles, secondary
devices, and the selected host policy is available in instance initialization:

```rust ignore
fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
    println!("{} passive={} profiles={:?} secondary={:?}",
        instance.name()?, instance.is_passive()?, instance.profile_names()?, instance.secondary_devices()?);
    instance.set_state(instance.host_policy()?)?;
    Ok(())
}

fn initialize(backend: &Backend) -> Result<(), Error> {
    let (kind, directory) = backend.artifacts()?;
    println!("{kind:?}: {}", directory.display());
    backend.set_execution_policy(ExecutionPolicy::Blocking)?;
    Ok(())
}
```

Backend config, preferred instance groups, parallel model loading, and auto-completed configs
are described in the model-management guide. `DeviceBlocking` shares a device execution thread
between GPU instances; CPU instances retain independent execution threads.

### Read and supply host-policy input data

A policy name selects existing storage. When that policy has no data, r25.09 falls back to
ordinary input. The owned metadata and host copies can outlive the request. They work for
`Request` and the read-only `RequestView` used by custom batching:

```rust ignore
fn read_policy(ctx: &Context, request: &Request) -> Result<Vec<u8>, Error> {
    let policy = ctx.instance_state::<HostPolicy>()?;
    let input = request.get_input("INPUT")?;
    let properties = input.properties_for_host_policy(policy.name())?;
    println!("{:?}: {} buffers", properties.shape, properties.buffer_count);
    if properties.buffer_count > 0 {
        let first = input.buffer_for_host_policy(policy.name(), 0)?;
        println!("First segment: {} bytes", first.len());
    }
    input.as_bytes_for_host_policy(policy.name())
}

fn policy_call(ctx: &Context) -> Result<InferenceRequest, Error> {
    let mut call = InferenceRequest::new(ctx.server(), "minimal_backend_target", -1)?;
    call.add_input_with_data("INPUT0", DataType::Fp32, &[1, 10], vec![0; 40])?;
    call.append_input_data_with_host_policy("INPUT0", &[0; 40], "pinned")?;
    let mut local = ctx.memory().pinned(40)?;
    local.copy_from_slice(&[0; 40]);
    call.append_input_buffer_with_host_policy("INPUT0", local, "cpu")?;
    Ok(call)
}
```

`buffer_for_host_policy(name, index)` copies one segment instead of the full input.
`append_input_buffer_with_host_policy` transfers ownership without copying and accepts pinned
buffers or shared inputs. Each policy list describes a full tensor. An explicitly empty policy
list suppresses fallback. These APIs do not allocate on a NUMA node themselves.

## Sequence state and iterative requests

With `sequence_batching.state` and an `initial_state` in the config, the sequence state of one
request becomes the next request's input. Read the configuration once with `Model::sequence_states`
and keep it in model state. This counter example uses `max_batch_size: 0` and state dims `[1]`.
A state borrows its request mutably and is consumed by `update`:

```rust ignore
fn increment(mut request: Request, config: &StateConfig) -> Result<(), Error> {
    let previous = request.get_input(config.input_name())?.as_fp32_vec()?;
    let previous = previous.first().copied().ok_or("state has no counter")?;
    let mut state = State::new(&mut request, config, &[1])?;
    state.write_bytes(&(previous + 1.0).to_ne_bytes())?;
    println!("State storage: {:?}", state.buffer_attributes()?);
    state.update()?;
    let mut response = Response::new(request)?;
    response.create_output("OUTPUT", DataType::Fp32, &[1])?.write_fp32_vec(&[previous + 1.0])?;
    response.send()?;
    Ok(())
}
```

The SDK rejects configurations from another model, invalid shapes, and shared state inputs;
a safe update also rejects an unwritten buffer. Shapes passed to `State::new` exclude batching:
the SDK prepends the per-sequence batch dimension when needed. Device writes require the
explicit unsafe completion contract of `update_after_device_write`.

An iterative sequence can return its request to the scheduler instead of finishing it:

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

This fragment uses `max_batch_size: 0`; the model must declare a decoupled iterative sequence. r25.09 only reschedules a request once;
see the guide before designing a multi-step loop.

## Custom batching

Group adjacent requests by a shared parameter using a custom batching strategy:

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

See [examples/batch-strategy](https://github.com/rdcm/triton-ng/tree/main/examples/batch-strategy)
for the complete example.

Point `TRITON_BATCH_STRATEGY_PATH` in the model config at that library. `false` ends a batch
before the current request; it cannot skip to a matching request farther in the queue. A head
that always refuses stalls until a configured timeout. Callback errors retain partial mutations.

The SDK retires unfinished states at batcher finalization and protects in-flight callbacks with
`Arc`. **r25.09 still closes the strategy library before joining its scheduler**: concurrent
library unmapping during a callback requires a core fix. Stop producers and drain callbacks
before unload. See [the lifecycle and failure guide](https://github.com/rdcm/triton-ng/blob/main/docs/batching_and_concurrency.md).

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
| [C API coverage](https://github.com/rdcm/triton-ng/blob/main/docs/c_api_coverage.md) | What of `tritonbackend.h` and the server functions a backend calls triton-ng wraps, and what it does not |
| [Benchmarks](https://github.com/rdcm/triton-ng/blob/main/docs/benchmarks.md) | python_backend and triton-ng on the same BLS workloads with `perf_analyzer`: method, models, environment, results |
| [python_backend comparison](https://github.com/rdcm/triton-ng/blob/main/docs/python_backend.md) | Runtime differences, feature coverage, isolation, and when to use Python |

## Development

Run these commands from the repository root:

| Command | Purpose |
|---|---|
| `make build` | Build the workspace in release mode, including examples and README snippet checks |
| `make download-model` | Download the MNIST model and create model version directories |
| `make docker-env-up` | Start local Triton after building and downloading the models |
| `make docker-env-down` | Stop the local Triton service |
| `make docker-env-clean` | Stop local Triton and remove its volumes |
| `make docker-logs` | Show local Triton logs |
| `make tests` | Run unit and integration tests; integration tests require local Triton |
| `make format` | Sort workspace manifests and format Rust code |
| `make lint` | Run Clippy |
| `make doc` | Build workspace documentation with warnings treated as errors |
| `make update-submodules` | Restore submodules to the branches in `.gitmodules` and refresh vendored headers |
| `make gen-grpc-client` | Regenerate the gRPC client from protobuf definitions |
| `make setup-protoc` | Install the protobuf compiler using `dnf` |
| `make benchmarks` | Run the benchmark suite in a separate Triton container |
| `make benchmarks-env-up` | Build and start the benchmark service on ports 9000–9002 |
| `make benchmarks-env-down` | Stop the benchmark service |
| `make publish DRY_RUN=1` | Package and check workspace crates without publishing |
| `make publish` | Publish workspace crates to crates.io |

User examples live in `examples/`; feature and failure fixtures live in
[tests/fixtures](https://github.com/rdcm/triton-ng/tree/main/triton-ng-client/tests/fixtures).
Models and Compose stay at the root. The local service requires Docker, NVIDIA Container
Toolkit, and a compatible NVIDIA driver because it includes GPU fixtures. Integration tests
run serially and read traces from `target/traces`.

## License

MIT
