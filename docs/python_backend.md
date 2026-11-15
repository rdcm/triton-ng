# Comparison with python_backend

[python_backend](https://github.com/triton-inference-server/python_backend) is NVIDIA's backend for
models written in Python, and the usual way to write BLS orchestration around framework models.
triton-ng covers the same ground from Rust. What matches and what differs, against python_backend
r25.09:

| | python_backend | triton-ng |
|---|---|---|
| Runs | a Python stub process per instance; requests and BLS calls cross an IPC boundary | in the Triton process: a `.so` that calls the C API |
| `execute` | `def` or `async def`; async decoupled requests can overlap on the instance’s event loop | `async fn`; the batch future runs on the Triton thread of its instance, BLS calls awaited together run at once |
| BLS | `exec()`; `async_exec()` is the blocking `exec` on asyncio's thread pool | `Server::infer_async`, a future woken by Triton's response callback |
| BLS to a decoupled model | `exec(decoupled=True)`, an iterator | `Server::infer_stream`, a `Stream` |
| Cancelling a BLS call | `iterator.cancel()` | `stream.cancel()`, or dropping the future or the stream |
| Decoupled models | `get_response_sender()`, `send(flags=FINAL)` | `Request::into_sender`; the types enforce exactly one final flag |
| Request metadata | id, correlation id, flags, timeout, parameters as JSON | the same, typed, one parameter at a time |
| BLS request metadata | id, correlation id, flags, timeout, parameters, trace | the same, plus priority and `f64` parameters |
| Errors | `TritonError(message, code)` | `respond_error_with_code`; the code of a failed BLS call is passed on |
| Rescheduling | `set_release_flags(RESCHEDULE)` | `Request::reschedule`, with the checks r25.09 does not make |
| Metrics, statistics | `MetricFamily`; statistics reported by the C++ side | `CounterFamily`, `GaugeFamily`, `HistogramFamily`; statistics reported by triton-ng |
| Model loading, `auto_complete_config` | `pb_utils.load_model`, `ModelConfig` | `Server::load_model`, `AutoCompleteConfig` |
| Backend config and attributes | not exposed | `Backend::config`, `CustomBackend::attributes` |
| Memory | numpy arrays; CPU tensors copied into shared memory, GPU tensors through DLPack and CUDA IPC | RAM, pinned and GPU buffers of Triton's memory manager; BLS outputs taken over and request inputs passed on without a copy |
| Data types | Triton-supported NumPy dtypes; DLPack for framework tensors | Typed helpers for fp32, fp64, i32, i64, u64, bool and strings; raw bytes for other Triton datatypes |
| GPU outputs of a response, GPU inputs read on the CPU | yes, copied | no: triton-ng has no CUDA runtime |
| Custom `is_ready` hook | Not available in r25.09 | Not available with the targeted r25.09 C API |
| PyTorch, TensorFlow, JAX inside the model | yes | no: the model goes to its framework backend and is called over BLS |
| A crash in the model | takes down the stub; Triton notices | panics are caught at the FFI boundary; a segfault takes down Triton |

When python_backend is the better choice: the model runs PyTorch, TensorFlow or JAX inside
`model.py` and cannot be moved to a framework backend; the pre- or post-processing is numpy code
that you do not want to reimplement in Rust; responses or BLS inputs must be GPU
tensors read or written on the CPU side; or a bug in the model code must not take Triton down.

## Coverage of the backend API additions

“Not exposed” means there is no corresponding public Python hook or `pb_utils` method in
r25.09, even if the C++ backend or Triton scheduler uses the capability internally.
“Supported” describes the current Rust Git API, including changes after 0.3.1.

| Area from #17 | python_backend r25.09 | triton-ng API and coverage | Limits / example |
|---|---|---|---|
| Model and instance memory usage | No user API for reporting owned allocations | Supported: `Model::report_memory_usage`, `ModelInstance::report_memory_usage` | Reporting only; [memory guide](https://github.com/rdcm/triton-ng/blob/main/docs/memory_buffers.md#reporting-memory-usage) |
| Instance host policy | Used internally when reading tensors; no explicit Python accessor | Supported: `ModelInstance::host_policy` | Returns configuration, not measured affinity; [instance policy](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md#instance-host-policy) |
| Host-policy input reads | No public choice of policy or individual policy buffer | Supported: `Input::properties_for_host_policy`, `buffer_for_host_policy`, `as_bytes_for_host_policy`, also through `RequestView` | Owned CPU copies; GPU reads rejected; [policy inputs](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md#instance-host-policy) |
| Custom trace activities | No public custom-activity API | Supported: `Request::trace`, `Trace::report_activity` | No-op when not traced; collector names validated; [tracing](https://github.com/rdcm/triton-ng/blob/main/docs/request_metadata_and_tracing.md#traces) |
| Per-response statistics | No per-response opt-in API; request/batch statistics are reported internally | Supported: `enable_statistics` on `Request`, `Response`, `StreamResponse` | Opt-in; [statistics](https://github.com/rdcm/triton-ng/blob/main/docs/metrics.md) |
| Sequence state buffers | Sequence metadata is available; no Python `TRITONBACKEND_State*` wrapper | Supported: `Model::sequence_states`, `State::new`, `write_bytes`, `buffer_attributes`, `update` | Initial state required; device writes need an unsafe completion contract; [state](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md#sequence-state) |
| Custom batching | Triton can load a native strategy library; no Python batching callback | Supported: `BatchStrategy`, `RequestView`, `#[batch_strategy]` | Separate library; r25.09 unload limitation; [batching](https://github.com/rdcm/triton-ng/blob/main/docs/batching_and_concurrency.md#custom-batching) |
| Instance details | Initialization exposes basic instance kind/device; no equivalent profile/passivity/secondary-device accessors | Supported: `ModelInstance::profile_names`, `is_passive`, `secondary_devices` | No instance-enumeration API in r25.09; [coverage](https://github.com/rdcm/triton-ng/blob/main/docs/c_api_coverage.md) |
| Input/output attributes and indexed access | No equivalent low-level buffer-attribute API | Partial: `State::buffer_attributes`; input/response attribute and indexed-access gaps remain | [Exact missing C functions](https://github.com/rdcm/triton-ng/blob/main/docs/c_api_coverage.md#not-wrapped) |
| Backend artifacts, state, execution policy | No backend-wide Python hook; instance state lives on the Python model object | Supported: `Backend::artifacts`, `set_state`, `state`, `set_execution_policy`; `Context::backend_state` | Backend state must be `Send + Sync`; [configuration](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md#backend-config) |
| BLS input/output removal | Construct a new `InferenceRequest`; no equivalent remove methods | Supported: `InferenceRequest::remove_input`, `remove_all_inputs`, `remove_requested_output`, `remove_all_requested_outputs` | Removed input owners remain until release; [mutation](https://github.com/rdcm/triton-ng/blob/main/docs/request_metadata_and_tracing.md#editing-a-bls-request-before-submission) |
| BLS host-policy appends | No public policy-specific append method | Supported: `append_input_data_with_host_policy`, `append_input_buffer_with_host_policy` | Full tensor per policy; no NUMA allocator; [appends](https://github.com/rdcm/triton-ng/blob/main/docs/request_metadata_and_tracing.md#appending-policy-specific-input-data) |

Sources: the [r25.09 Python API and async execution documentation](https://github.com/triton-inference-server/python_backend/blob/r25.09/README.md),
[Python bindings](https://github.com/triton-inference-server/python_backend/blob/r25.09/src/pb_stub.cc),
and [backend implementation](https://github.com/triton-inference-server/python_backend/blob/r25.09/src/python_be.cc).
The original audit is [#17](https://github.com/rdcm/triton-ng/issues/17);
[C API coverage](https://github.com/rdcm/triton-ng/blob/main/docs/c_api_coverage.md) maps the Rust support back to C functions.
