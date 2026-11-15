# triton-ng documentation

Start with [Getting started](https://github.com/rdcm/triton-ng#getting-started) for the complete
minimal backend, then work through BLS, streaming, metrics, tracing, memory, sequence state,
and custom batching in the [main README](https://github.com/rdcm/triton-ng/blob/main/README.md).
The examples describe the current Git API; crates.io/docs.rs may still describe 0.3.1.

## Guides

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
a coverage matrix for every area of the #17 audit and cases where Python is the better choice.

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
are in the benchmark guide. Reproduce with `make benchmarks` (CPU-only benchmark container,
ports 9000–9002); the full suite takes about 15 minutes. These results were not remeasured for
this documentation reorganization.

## Examples and verification

User examples are the [minimal backend](https://github.com/rdcm/triton-ng/tree/main/examples/minimal-backend)
and [minimal custom batching strategy](https://github.com/rdcm/triton-ng/tree/main/examples/batch-strategy).
The root `minimal_batching` model runs them together.
Each feature or failure group has its own [test fixture](https://github.com/rdcm/triton-ng/tree/main/triton-ng-client/tests/fixtures).
The test model selects its fixture at initialization; no shared request-parameter dispatcher
selects unrelated test behavior. `models/` and `docker-compose.yml` stay in the root.

`make build` compiles every Rust snippet in the root README and these guides using the
compile-only [`tools/readme-checks`](https://github.com/rdcm/triton-ng/tree/main/tools/readme-checks) tool. Each code block is checked
independently with SDK types, `Arc`, `Mutex`, and `Instant` in scope. The examples are functions
or hooks; import those types when copying them into your backend.

`make doc` renders the guides and checks rustdoc warnings; it does not execute examples.
`make tests` runs unit and integration tests against the root Triton service. Runtime examples
require their documented model configs and inputs; compilation alone does not validate those
configs or GPU behavior. The benchmark tables are recorded results, not part of these checks.
