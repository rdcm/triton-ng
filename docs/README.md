# triton-ng documentation

How to write custom Triton backends with triton-ng. Start with the [main README](https://github.com/rdcm/triton-ng/blob/main/README.md)
for a minimal backend, the gRPC client and running the example.

| Section | What it covers |
|---|---|
| [Execution model and thread safety](https://github.com/rdcm/triton-ng/blob/main/docs/execution_model_and_thread_safety.md) | Instances and their threads, state of models and instances, request ownership, error codes, BLS futures, threads, cancellation, shutdown, batching |
| [Batching and concurrency](https://github.com/rdcm/triton-ng/blob/main/docs/batching_and_concurrency.md) | How Triton forms batches and why a batch runs on the Triton thread of its instance, BLS calls joined inside a batch, tuning with `instance_group`, measurements |
| [Streaming](https://github.com/rdcm/triton-ng/blob/main/docs/streaming.md) | Decoupled models: streams of responses from a backend, BLS streams, cancellation of BLS calls, the client side, how it is implemented and its limitations |
| [Model parameters](https://github.com/rdcm/triton-ng/blob/main/docs/model_parameters.md) | The `parameters` map of a model config and how a backend reads it |
| [Request metadata and tracing](https://github.com/rdcm/triton-ng/blob/main/docs/request_metadata_and_tracing.md) | ID, correlation ID, sequence flags and timeout of a request; setting them, the priority and parameters on BLS calls; BLS calls as children of the trace of a request |
| [Metrics](https://github.com/rdcm/triton-ng/blob/main/docs/metrics.md) | Counters, gauges and histograms of the backend on Triton's Prometheus endpoint; the inference statistics triton-ng reports for every model |
| [Model management and configuration](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md) | The backend config from `--backend-config`, backend attributes (preferred instance groups, parallel instance loading), completing model configs, loading and unloading models from a backend, rescheduling requests of iterative sequences |
| [Memory buffers](https://github.com/rdcm/triton-ng/blob/main/docs/memory_buffers.md) | RAM, pinned and GPU buffers from Triton's memory manager for BLS inputs and outputs, caching them in a backend, passing inputs of a request on without a copy |
| [Benchmarks](https://github.com/rdcm/triton-ng/blob/main/docs/benchmarks.md) | python_backend and triton-ng on the same BLS workloads with `perf_analyzer`: method, models, environment, results |

[`example/custom-backend`](https://github.com/rdcm/triton-ng/blob/main/example/custom-backend/src/lib.rs) is the MNIST example backend that
uses all of it.
