# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-04

A batch runs on the Triton thread of its instance instead of a tokio runtime, which restores
Triton's batching under load; models of a triton-ng backend report their inference statistics;
the README and the guides are on docs.rs with every public item documented. Breaking: the
runtime model parameters, `InferenceFuture` and the public modules are gone. Targets Triton
`r25.09` and Rust 1.99. All four crates are published as 0.3.0.

### Changed ([#13])

- **Breaking:** a batch runs to its end on the Triton thread that called execute, with a minimal
  `block_on` instead of a tokio runtime per instance. The instance takes its next batch only after
  that, so Triton keeps the requests in its queue and batches them as for a synchronous backend;
  batches of a model run in parallel on its instances (`instance_group` count). triton-ng starts no
  threads of its own. Timers, I/O and spawning of tokio or other runtimes no longer work inside
  `model_instance_execute`; a blocking call (e.g. `Server::load_model`) blocks only its instance.
- **Breaking:** the model parameters `worker_threads` and `max_batches_in_flight` are removed.
  triton-ng no longer reads them, batches of a model run in parallel on its instances.
- **Breaking:** `InferenceFuture` is removed. `Server::infer_async` returns
  `impl Future<Output = Result<InferenceResult, TritonError>> + Send + 'static`, still submitting
  the request at the call. Dropping the future now cancels the request (`InferenceFuture::cancel`
  is gone), as dropping an `InferenceStream` does.
- The result of `infer_async` goes through a tokio oneshot channel instead of a hand-written slot;
  triton-ng depends on tokio only for its `sync` channels and no longer on `tokio-util`.

- **Breaking:** the modules of `triton-ng` are private: every type is at the crate root
  (`triton_ng::Request`, as the examples always showed), only `triton_ng::log` stays a module.
  The crate page on docs.rs now lists the types instead of re-exports.

### Added

- The crate documentation on docs.rs includes the README and the guides of `docs/` (module
  `docs`); every public item is documented and `#![deny(missing_docs)]` keeps it so. `make doc`
  builds the documentation as docs.rs does, with warnings as errors.
- Inference statistics: every request is reported to Triton when it is released (as successful
  when its final response went out without an error, with the time since its batch started as
  compute time) and every batch when it is done, so `nv_inference_*` and
  `/v2/models/<model>/stats` show the requests, batches, queue and compute times of the models of
  a triton-ng backend. See [Metrics](docs/metrics.md).
- `Model::max_batch_size`.
- [Batching and concurrency](docs/batching_and_concurrency.md): how Triton forms batches, why a
  batch runs on the Triton thread, tuning with `instance_group`, measurements.

## [0.2.0] - 2026-11-02

Custom backends come close to feature parity with Triton's python_backend: streaming, request
metadata and tracing, backend metrics, memory-manager buffers, model management, and passing
request inputs on to BLS calls without a copy. Targets Triton `r25.09`
(`nvcr.io/nvidia/tritonserver:25.09`) and Rust 1.99. All four crates are published as 0.2.0.

### Added

- **Memory buffers** ([#6], [#11])
  - `Context::memory()` gives Triton's memory manager: `ram(len)` returns a `RamBuffer`,
    `pinned(len)` a `PinnedRamBuffer`, `gpu(device_id, len)` an `UninitGpuBuffer` from Triton's
    CUDA pool. triton-ng does not use the CUDA runtime.
  - `add_input_with_buffer` takes any `InputBuffer`: `Vec<u8>`, the buffer types above, a
    `SharedInput`, or an `Arc` of any of them.
  - `add_requested_output_into` writes a BLS output into your buffer, and
    `InferenceResult::take_gpu` / `take_pinned` / `take_ram` hand it back for reuse. GPU data
    stays on the GPU across BLS calls.
  - `Input::share()` passes a request input in CPU or GPU memory on as a BLS input without a
    copy. While a shared input is alive, the final response is held back and sent when the last
    one is dropped, because in r25.09 the gRPC and HTTP frontends free input data once the final
    response is sent.
  - Instance state: `ModelInstance::set_state` and `Context::instance_state::<T>()`.
- **Error codes** ([#7])
  - Every `TritonError` has an `ErrorCode`.
  - `Request::respond_error_with_code` answers with any code; clients get it as the gRPC status.
  - A failed BLS call keeps the code Triton or the called model gave it, for example `NotFound`
    for an unknown model.
- **Request metadata, tracing and metrics** ([#8])
  - `Request::id`, `correlation_id`, `flags` and `timeout`.
  - `InferenceRequest::set_id`, `set_correlation_id`, `set_flags`, `set_timeout`, `set_priority`
    and `set_parameter`. Setters refuse values Triton would misread.
  - `InferenceResult::parameter` reads the parameters the called model set on its response.
  - `InferenceRequest::set_parent_trace` makes a BLS call a child of the client request's trace.
    The parent is released only after its children are done.
  - `CounterFamily`, `GaugeFamily` and `HistogramFamily` publish metrics on Triton's Prometheus
    endpoint. Names, labels, buckets and values are validated before they reach Triton.
- **Streaming** ([#9])
  - `Request::into_sender()` returns a `ResponseSender` for decoupled models: any number of
    `StreamResponse`s, then exactly one final flag (`finish`, `final_response`, `respond_*`).
  - `Server::infer_stream` returns an `InferenceStream`, a `futures_core::Stream` of BLS results.
  - Cancellation with `InferenceStream::cancel`, dropping the stream, or `InferenceFuture::cancel`.
  - `TritonClient::infer_stream` in the gRPC client.
- **Model management and configuration** ([#10])
  - `Backend::config()` returns the `--backend-config` settings.
  - Optional hooks `CustomBackend::attributes` (preferred instance groups, parallel instance
    loading) and `CustomBackend::auto_complete_config` (completes model configs following
    python_backend's rules).
  - `Server::load_model(name, config, files)` and `Server::unload_model(name, unload_dependents)`,
    like `pb_utils.load_model` and `pb_utils.unload_model`.
  - `Request::reschedule` and `ResponseSender::reschedule` for `iterative_sequence` models. Two
    r25.09 pitfalls that would drop the request silently are refused with an error: rescheduling
    a request of a model without `iterative_sequence`, and rescheduling a request more than once.
- New guides in [`docs/`](https://github.com/rdcm/triton-ng/tree/0.2.0/docs): memory buffers,
  streaming, request metadata and tracing, metrics, model management and configuration; the
  execution model guide has a table of every handle, checked by a unit test.

### Changed

- **Breaking:** `trait Backend` is now `trait CustomBackend` ([#8]). A typical backend needs only
  this rename:

  ```diff
  -use triton_ng::backend::Backend;
  +use triton_ng::CustomBackend;

   #[triton_ng::backend]
  -impl Backend for MyBackend {
  +impl CustomBackend for MyBackend {
  ```

- **Breaking:** `BackendHandle` is now `Backend`, the struct passed to `initialize` / `finalize`
  ([#8]).
- **Breaking:** `InstanceKind` is now `InstanceGroupKind`; `ModelInstance::kind` fails on an
  unknown kind instead of returning `Auto` ([#10]).
- **Breaking:** `Model::location` is replaced by `Model::directory() -> PathBuf`, which fails when
  the repository is not on the local file system ([a470ec8]).
- **Breaking:** `OutputTensor::data` is replaced by `OutputTensor::bytes()`, which fails for
  outputs in GPU memory ([#6]).
- **Breaking:** `Error` is `Box<dyn Error + Send + Sync>`, so a `Result` can be held across
  `.await` ([#6]).
- **Breaking:** `InferenceResult` has private fields and can no longer be built with a struct
  literal ([#8]).
- `Server::infer_async` fails when a model sends more than one response, instead of merging them
  ([#9]).
- BLS outputs in CPU memory are handed to the result without a second copy ([#7]).
- `Input::bytes()` borrows the tensor from the request when Triton keeps it in one buffer, which
  is the usual case ([#7]).
- One response allocator per backend instead of one per BLS call ([#7]).

Items are still re-exported from the crate root, so `use triton_ng::{...}` imports keep working
where the names did not change. New hooks have default implementations.

### Removed

- **Breaking:** `InferenceResult::error`, which was always `None`: a failed call resolves to
  `Err(TritonError)` ([#7]).
- **Breaking:** the `cuda` feature, whose `cudaMalloc` outputs could not be read: use
  memory-manager buffers ([#6]).
- **Breaking:** `ResponseAllocator` and `InferenceResponse` are crate-internal ([#6]).

## [0.1.0] - 2026-11-02

The first release on crates.io: a safe Rust API for writing custom Triton backends, built as a
`.so` that Triton loads, and an async gRPC client for calling a running Triton server. Targets
Triton `r25.09` (`nvcr.io/nvidia/tritonserver:25.09`) and Rust 1.99.

### Added

- Crates `triton-ng` (safe API for custom backends), `triton-ng-macros`
  (`#[triton_ng::backend]`, which generates the `TRITONBACKEND_*` entry points), `triton-ng-sys`
  (raw bindings to `tritonbackend.h` generated by bindgen from vendored headers) and
  `triton-ng-client` (async gRPC client on tonic).
- **Execution model**
  - `model_instance_execute` is an `async fn`. Each model instance has its own multi-threaded
    tokio runtime, and `ModelInstanceExecute` returns to Triton at once, so one instance works on
    several batches at the same time.
  - Model parameters `worker_threads` (threads of each instance runtime, default 1) and
    `max_batches_in_flight` (batches one instance works on at once; extra requests wait in
    Triton's queue).
  - `Server::infer_async` submits a BLS request and returns an `InferenceFuture` woken by
    Triton's callback; no thread waits for a model.
  - Shutdown waits for every batch of an instance before `model_instance_finalize` runs.
  - Panics are caught at the FFI boundary and reported to Triton as errors.
- **Requests and responses**
  - Exactly one final response: `Response::new`, `Request::respond_error` and
    `Request::respond_cancelled` consume the request. A request dropped without a response gets
    an error response automatically.
  - `Request::get_input` returns an `Input` with `as_bytes`, `as_string`, `as_strings`, `as_u64`,
    `as_fp32_vec`, `as_i32_vec`, `as_i64_vec` and `properties`; `Request::input_names` and
    `output_names` list the tensors.
  - `Request::parameter` and `Request::is_cancelled`.
  - `Response::create_output` with `write_bytes`, `write_string(s)`, `write_fp32_vec`,
    `write_fp64_vec`, `write_i32_vec`, `write_i64_vec`, `write_bool_vec` and `write_u64`. Sizes
    are checked against the shape, and every created output must be written before `send`.
  - `Response::set_parameter`.
- **Models, instances and the server**
  - `Model::set_state` and `Context::state::<T>()` for state shared by all instances of a model.
  - `Model::config_parameter_or::<T>`, `required_config_parameter` and `config_json`.
  - `Model::name`, `version`, `location`, `path` and `load_file`.
  - `ModelInstance::name`, `kind` and `device_id`.
  - `Server::infer_async`, `is_ready`, `is_model_ready`, `model_metadata` and `api_version`.
  - `log::info`, `warn`, `error` and `verbose` write to Triton's log.
- **gRPC client** `TritonClient` for the KServe v2 gRPC API and Triton's extensions:
  - health and metadata: `server_live`, `server_ready`, `server_metadata`, `model_ready`,
    `model_metadata`, `model_config`, `model_statistics`;
  - `infer` with typed `InferInput` and decoded `InferResponse` outputs;
  - model repository: `repository_index`, `repository_model_load`, `repository_model_unload`;
  - system and CUDA shared memory: `*_status`, `*_register`, `*_unregister`;
  - `trace_settings` and `log_settings`;
  - TLS with `TritonClientConfig::with_tls`.
- Builds without submodules, Triton sources or `protoc`: the headers are vendored and the gRPC
  code is pre-generated.
- Experimental feature `cuda`: allocates BLS outputs with the CUDA runtime.

### Fixed

Fixed before the first release; the bugs affected backends that depended on the git repository.

- Memory leak on every BLS call into an onnxruntime model: the response allocator had no query
  function, so `TRITONBACKEND_RequestOutputBufferProperties` returned an error that the
  onnxruntime backend never freed. The allocator now registers a query function.
- An empty output crashed Triton's HTTP frontend because it never got a buffer. Every created
  output now gets one, empty ones included.

[Unreleased]: https://github.com/rdcm/triton-ng/compare/0.3.0...HEAD
[0.3.0]: https://github.com/rdcm/triton-ng/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/rdcm/triton-ng/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/rdcm/triton-ng/releases/tag/0.1.0
[a470ec8]: https://github.com/rdcm/triton-ng/commit/a470ec8b87eb2405e6191df19fb1469243897828
[#6]: https://github.com/rdcm/triton-ng/pull/6
[#7]: https://github.com/rdcm/triton-ng/pull/7
[#8]: https://github.com/rdcm/triton-ng/pull/8
[#9]: https://github.com/rdcm/triton-ng/pull/9
[#10]: https://github.com/rdcm/triton-ng/pull/10
[#11]: https://github.com/rdcm/triton-ng/pull/11
[#13]: https://github.com/rdcm/triton-ng/pull/13
