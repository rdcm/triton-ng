# Benchmarks

python_backend and triton-ng on the same BLS workloads, measured with `perf_analyzer`: the
method, the models, how to run them, the environment and the results.

## Method

python_backend has no benchmark suite of its own; its examples are driven by hand with
`tritonclient`. Triton's standard tool for the throughput and latency of a model is
`perf_analyzer`, part of the Triton SDK and on PyPI as `perf-analyzer`, and these benchmarks use
it with its default measurement procedure:

- gRPC, asynchronous requests, a fixed number of requests in flight (`--concurrency-range N`), at
  concurrency 1, 4, 16 and 64;
- measurement windows of 5 s (`--measurement-interval 5000`); a result is accepted when three
  consecutive windows agree within 10 % in both throughput and average latency
  (`--stability-percentage 10`), after at most ten windows (`--max-trials 10`);
- random input data, one request per inference (`-b 1`): batches are formed by Triton's dynamic
  batcher, as in production;
- decoupled models are driven over the gRPC stream (`--streaming`). perf_analyzer counts requests
  per second for them and measures latency to the first response of a request.

The driver, `benches/benchmarks.py`, runs perf_analyzer for every benchmark, model and
concurrency level, keeps its CSV reports and logs in `target/benchmarks/` and prints the tables
below (also `target/benchmarks/results.md`). Both backends are loaded into one Triton and
measured one after another by the same client.

## Models

Every benchmark is a pair of models: `<name>_rs` on triton-ng (`benches/backend.rs`, backend
`bench_backend`) and `<name>_py` on python_backend (`benches/models/<name>_py/1/model.py`),
with the same model config apart from the backend name (`benches/models/*/config.pbtxt`): CPU,
two instances, dynamic batching without `max_queue_delay_microseconds`, so that at low concurrency
the latency is the backend's and not the batcher's wait for company; under load the batches are
formed by Triton's back-pressure either way, see
[Batching and concurrency](https://github.com/rdcm/triton-ng/blob/main/docs/batching_and_concurrency.md). The Rust and the Python model do the same work step by step: read the input,
call the same model over BLS, take the argmax of its logits, answer.

| Benchmark | Model | What it measures |
|---|---|---|
| `argmax` | `max_batch_size: 8`, dynamic batching without a queue delay | A request in and a response out with no BLS: the backend's own cost per request. The answer is the argmax of the 28×28 input |
| `infer` | the same | One BLS call per request, awaited one after another: `Server::infer_async` against `InferenceRequest.exec()` |
| `infer_concurrent` | the same | The BLS calls of a batch in flight together: `join_all` of `infer_async` futures against `async def execute` with `async_exec()` and `asyncio.gather` |
| `decoupled` | `max_batch_size: 0`, decoupled, input `[N, 1, 28, 28]` | A decoupled model: one BLS call and one streamed response per image, N = 4, then the final flag: `ResponseSender` against `get_response_sender()` |
| `stream` | `max_batch_size: 0`, input `[N, 1, 28, 28]` | Reading a stream over BLS: the model calls the `decoupled` model of its own backend and answers with the N digits: `Server::infer_stream` against `exec(decoupled=True)` |

The BLS target is tiny on purpose, so that the numbers are the cost of the backends themselves; a
heavy model on a GPU takes most of the time in inference and the difference shrinks with its
share. It is `bench_onnx`, the MNIST ONNX model of the examples
(`models/mnist_onnx/1/model.onnx`, copied by `make benchmarks`), on four CPU instances with one
thread each (`intra_op_thread_count: 1`): 0.1 ms per call, so that it is not the bottleneck and
does not spin on every core, which made the measurements unstable with the default thread count.

## Running

Requirements: Docker, [uv](https://docs.astral.sh/uv/), the model from `make download-model`.
No GPU is needed.

```bash
make benchmarks
```

`make benchmarks` builds the bench backend (`make build`), starts a Triton of its own from
`benches/docker-compose.yml` (container `triton-bench`, ports 9000–9002, so the development
Triton of `make docker-env-up` can keep running), runs the driver with `uv run` (perf_analyzer is
installed from PyPI into `benches/.venv`) and stops the container. That Triton loads only
`benches/models` and runs without verbose logging and tracing. A full run takes about 15 minutes.

`make benchmarks-env-up` and `make benchmarks-env-down` start and stop that Triton on their own,
for partial runs:

```bash
cd benches && uv run benchmarks.py --only infer,stream --concurrency 16,64
```

## Environment

- CPU: AMD Ryzen 9 9950X3D 16-Core Processor, 32 threads
- RAM: 30 GiB
- GPU: NVIDIA GeForce RTX 5080, 580.178.04 (not used by the benchmark models)
- OS: Linux-7.2.15-100.fc43.x86_64-x86_64-with-glibc2.42
- Triton: 2.61.0 (nvcr.io/nvidia/tritonserver:25.09-py3)
- perf_analyzer: perf-analyzer 2.60.0 from PyPI (the binary reports Perf Analyzer Version 0.0.0 (commit 3a49a23))
- Concurrency: 1, 4, 16, 64; measurement window 5000 ms, stability 10 %

## Results

Throughput in requests per second, latency as perf_analyzer reports it (p50 / p99); `Speedup` is triton-ng over python_backend.

### `argmax`

| Concurrency | triton-ng, infer/s | python_backend, infer/s | Speedup | triton-ng p50 / p99, ms | python_backend p50 / p99, ms |
|---|---|---|---|---|---|
| 1 | 20770 | 12224 | 1.70× | 0.04 / 0.07 | 0.07 / 0.15 |
| 4 | 49846 | 34489 | 1.45× | 0.07 / 0.12 | 0.10 / 0.20 |
| 16 | 89336 | 47514 | 1.88× | 0.16 / 0.27 | 0.32 / 0.52 |
| 64 | 119907 | 52716 | 2.27× | 0.52 / 0.81 | 1.19 / 1.52 |

### `infer`

| Concurrency | triton-ng, infer/s | python_backend, infer/s | Speedup | triton-ng p50 / p99, ms | python_backend p50 / p99, ms |
|---|---|---|---|---|---|
| 1 | 11543 | 4967 | 2.32× | 0.08 / 0.13 | 0.19 / 0.29 |
| 4 | 26816 | 16175 | 1.66× | 0.14 / 0.22 | 0.23 / 0.44 |
| 16 | 42150 | 18331 | 2.30× | 0.36 / 0.57 | 0.87 / 1.36 |
| 64 | 43422 | 17819 | 2.44× | 1.45 / 1.74 | 3.56 / 4.35 |

### `infer_concurrent`

| Concurrency | triton-ng, infer/s | python_backend, infer/s | Speedup | triton-ng p50 / p99, ms | python_backend p50 / p99, ms |
|---|---|---|---|---|---|
| 1 | 12662 | 3168 | 4.00× | 0.07 / 0.12 | 0.30 / 0.48 |
| 4 | 25640 | 8129 | 3.15× | 0.14 / 0.23 | 0.47 / 0.80 |
| 16 | 59565 | 12114 | 4.92× | 0.25 / 0.40 | 1.33 / 1.92 |
| 64 | 74316 | 13317 | 5.58× | 0.84 / 1.09 | 4.80 / 5.66 |

### `decoupled`

| Concurrency | triton-ng, infer/s | python_backend, infer/s | Speedup | triton-ng p50 / p99, ms | python_backend p50 / p99, ms |
|---|---|---|---|---|---|
| 1 | 5907 | 1588 | 3.72× | 0.15 / 0.23 | 0.58 / 0.80 |
| 4 | 13740 | 3756 | 3.66× | 0.26 / 0.38 | 1.01 / 1.33 |
| 16 | 12867 | 3670 | 3.51× | 1.20 / 1.57 | 4.30 / 5.28 |
| 64 | 12802 | 3670 | 3.49× | 4.86 / 6.56 | 17.32 / 21.60 |

### `stream`

| Concurrency | triton-ng, infer/s | python_backend, infer/s | Speedup | triton-ng p50 / p99, ms | python_backend p50 / p99, ms |
|---|---|---|---|---|---|
| 1 | 5970 | 1249 | 4.78× | 0.15 / 0.25 | 0.78 / 1.05 |
| 4 | 14890 | 3243 | 4.59× | 0.25 / 0.37 | 1.21 / 1.60 |
| 16 | 14647 | 3213 | 4.56× | 1.05 / 1.42 | 4.92 / 6.68 |
| 64 | 15056 | 3253 | 4.63× | 4.13 / 5.57 | 19.62 / 23.70 |

## Reading the results

- **At concurrency 1** the whole round trip of a request through perf_analyzer, gRPC, Triton's
  frontend and the backend takes tens of microseconds, and the backend's share is visible even
  there: 1.7× on `argmax`, 2.3× on `infer`, 3.7× on `decoupled`. From concurrency 16 on the
  instances are busy and the throughput is the backend's.
- **`argmax`**, with no BLS at all, is the cost of the backend itself: 120k against 53k requests
  per second at concurrency 64. python_backend runs a stub process per instance: every batch is
  written to shared memory, the stub is woken over an inter-process condition variable, it reads
  the inputs as numpy arrays, writes the outputs back to shared memory and the C++ side copies
  them into the responses. triton-ng reads the inputs and writes the outputs through the C API
  in the Triton process.
- **`infer`**: a BLS call of python_backend is the same round trip again, stub → Triton →
  stub, with the input and the output copied through shared memory. triton-ng submits the
  request with `TRITONSERVER_ServerInferAsync` and is woken by the response callback; the output
  buffer of the response is taken over without a copy. 43k against 18k at concurrency 64.
- **`infer_concurrent`**: the BLS calls of a batch in flight together take triton-ng from 43k to
  74k, and python_backend from 18k down to 13k. `async_exec` is the blocking `exec` run on
  asyncio's default thread pool, so every BLS call in flight holds a thread and the event loop
  adds its own cost; the futures of triton-ng hold nothing and are woken by Triton.
- **`decoupled`** multiplies the BLS cost by the four images and adds a response per image:
  12.8k against 3.7k requests per second, 3.5×. **`stream`** reads that stream over BLS and
  answers the client once: 14.6k against 3.2k, 4.6×. python_backend reads the stream through the
  same shared memory channel, response by response; triton-ng is faster here than on `decoupled`
  itself because the four responses stay inside Triton and the gRPC frontend sends one.
- **Latency** follows throughput: at concurrency 64 the requests queue for the two instances,
  and the queue drains as fast as the instances answer, so p50 is the throughput's inverse
  times the concurrency on both sides.

## Limitations

- The client runs on the same machine as the server and takes a share of the CPU; the absolute
  numbers depend on the machine, the ratios much less.
- The tensors are small (3 KiB per image), so the cost per call dominates. The copy through
  shared memory of python_backend grows with the tensor, the in-process call of triton-ng does
  not; with large tensors the compute of the BLS target dominates both.
- CPU only: GPU inputs and outputs are not measured. python_backend moves GPU BLS tensors
  through CUDA IPC, triton-ng passes Triton's buffers on, see
  [Memory buffers](https://github.com/rdcm/triton-ng/blob/main/docs/memory_buffers.md).
- One run of one machine; numbers vary by a few percent between runs, see the stability
  criterion above.
- The latency of the decoupled model is to its first response, as perf_analyzer measures it.
