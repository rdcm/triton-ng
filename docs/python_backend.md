# Comparison with python_backend

[python_backend](https://github.com/triton-inference-server/python_backend) is NVIDIA's backend for
models written in Python, and the usual way to write BLS orchestration around framework models.
triton-ng covers the same ground from Rust. What matches and what differs, against python_backend
r25.09:

| | python_backend | triton-ng |
|---|---|---|
| Runs | a Python stub process per instance; every batch, tensor and BLS call crosses to it through shared memory | in the Triton process: a `.so` that calls the C API |
| `execute` | `def` or `async def`; the batches of an instance run one after another | `async fn`; the batch future runs on the Triton thread of its instance, BLS calls awaited together run at once |
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
| Data types | any numpy dtype | fp32, fp64, i32, i64, u64, bool, strings and raw bytes |
| GPU outputs of a response, GPU inputs read on the CPU | yes, copied | no: triton-ng has no CUDA runtime |
| `is_ready` | yes | no, needs r26.04 |
| PyTorch, TensorFlow, JAX inside the model | yes | no: the model goes to its framework backend and is called over BLS |
| A crash in the model | takes down the stub; Triton notices | panics are caught at the FFI boundary; a segfault takes down Triton |

When python_backend is the better choice: the model runs PyTorch, TensorFlow or JAX inside
`model.py` and cannot be moved to a framework backend; the pre- or post-processing is numpy code
with no equivalent in Rust yet (fp16, bf16, u8 tensors); responses or BLS inputs must be GPU
tensors read or written on the CPU side; or a bug in the model code must not take Triton down.

This comparison targets r25.09. For custom batching and sequence-state ownership, see
[Batching and concurrency](https://github.com/rdcm/triton-ng/blob/main/docs/batching_and_concurrency.md) and
[Model management](https://github.com/rdcm/triton-ng/blob/main/docs/model_management_and_configuration.md).
