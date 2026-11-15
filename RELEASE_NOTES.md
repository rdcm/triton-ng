# triton-ng 0.4.0

This release expands the backend API and reorganizes the documentation and examples.
It targets **Triton 25.09 / backend API 1.19** and **Rust 1.99**.
All four public crates move from 0.3.1 to 0.4.0.

## New since 0.3.1

- **Custom batching:** implement `BatchStrategy` with `#[triton_ng::batch_strategy]` and
  inspect queued requests through `RequestView` to decide which requests form a batch.
- **Sequence state:** read configured state, write validated updates, and access buffer
  attributes through `State` and `StateConfig`.
- **Observability:** report model and instance memory usage, emit custom trace activities,
  and opt in to per-response statistics, including streaming and error responses.
- **Backend configuration:** access instance profiles, passive status, secondary devices,
  host-policy settings, backend artifacts, shared state, and execution policy.
- **Host-policy inputs and BLS:** read policy-specific input data, append copied or shared
  policy-specific buffers, and remove inputs or requested outputs from an inference request.
- **Compatibility checks:** validate Triton's backend API version during initialization.

## Documentation and examples

The README introduces features from a minimal backend through streaming, metrics, sequence
state, and custom batching. The guides include a python_backend comparison and a function-by-function
coverage reference for all 276 functions in the r25.09 backend/server headers. README and guide
Rust snippets are compile-checked during the build.

`examples/` contains a minimal backend and a minimal by-key batching strategy. Test scenarios
live in dedicated fixtures, with integration tests grouped by feature and isolated for parallel
execution. Benchmark summaries reuse the existing measurements; this release does not introduce
new benchmark results.

## Upgrading from 0.3.1

Update the triton-ng crates used by your application to `0.4.0` together. The required Rust and
Triton versions remain unchanged.

`Input` now has an owner parameter: `Input<'a, P = Request>`. Existing `Input<'a>` annotations
keep their default owner. Inputs obtained through the new `RequestView` are read-only and cannot
be shared into BLS requests.

Paths under `examples/` have changed. Use `examples/minimal-backend` and
`examples/batch-strategy` for runnable examples; test-only backends and strategies are under
`triton-ng-client/tests/fixtures/`.

## Known limitation

Triton r25.09 can close a batching strategy library before its scheduler has finished running
callbacks. Stop producers and drain requests before unloading that library. The SDK retains
Rust callback state safely, but concurrent library unmapping requires a Triton core fix.
See [batching limitations](https://github.com/rdcm/triton-ng/blob/main/docs/batching_and_concurrency.md).

[Full changelog](https://github.com/rdcm/triton-ng/blob/main/CHANGELOG.md#040---2026-11-15)
