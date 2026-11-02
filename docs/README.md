# triton-ng documentation

How to write custom Triton backends with triton-ng. Start with the [main README](../README.md)
for a minimal backend, the gRPC client and running the example.

| Section | What it covers |
|---|---|
| [Execution model and thread safety](execution_model_and_thread_safety.md) | Instances and their runtimes, state of models and instances, request ownership, error codes, BLS futures, threads, cancellation, shutdown, batching |
| [Streaming](streaming.md) | Decoupled models: streams of responses from a backend, BLS streams, cancellation of BLS calls, the client side, how it is implemented and its limitations |
| [Model parameters](model_parameters.md) | `worker_threads` and `max_batches_in_flight` in the model config |
| [Request metadata and tracing](request_metadata_and_tracing.md) | ID, correlation ID, sequence flags and timeout of a request; setting them, the priority and parameters on BLS calls; BLS calls as children of the trace of a request |
| [Metrics](metrics.md) | Counters, gauges and histograms of the backend on Triton's Prometheus endpoint |
| [Model management and configuration](model_management_and_configuration.md) | The backend config from `--backend-config`, backend attributes (preferred instance groups, parallel instance loading), completing model configs, loading and unloading models from a backend, rescheduling requests of iterative sequences |
| [Memory buffers](memory_buffers.md) | RAM, pinned and GPU buffers from Triton's memory manager for BLS inputs and outputs, caching them in a backend, passing inputs of a request on without a copy |

[`example/custom-backend`](../example/custom-backend/src/lib.rs) is the MNIST example backend that
uses all of it.
