# triton-ng 0.4.0

This release expands the backend API and refreshes the documentation and examples.
All four crates are versioned **0.4.0**, targeting **Triton 25.09 / backend API 1.19**
and **Rust 1.99**.

## New since 0.3.1

- **Custom batching:** `BatchStrategy`, `#[triton_ng::batch_strategy]`, and read-only
  `RequestView` access to queued requests.
- **Sequence state:** configuration, validated state updates, and buffer attributes.
- **Observability:** model and instance memory reports, custom trace activities, and
  opt-in per-response statistics for unary, streaming, and error responses.
- **Backend and instance APIs:** profiles, secondary devices, passive status, host-policy
  settings, artifacts, shared state, and execution policy.
- **Host-policy inputs and BLS:** read policy-specific input data, append copied or shared
  buffers, and remove inputs or requested outputs.
- **Compatibility checks:** validate the backend API version during initialization.

## Documentation and examples

The README and guides introduce features progressively, with compile-checked Rust snippets,
a python_backend comparison, and a coverage reference for all 276 C API functions in the
r25.09 backend/server headers.

Minimal backend and by-key batching examples live in `examples/`. Test scenarios have moved
to dedicated fixtures, with integration tests isolated for parallel execution.

## Upgrading

Update the triton-ng crates you use to `0.4.0` together. Rust and Triton requirements remain
unchanged. `Input` now has an owner parameter, `Input<'a, P = Request>`; existing `Input<'a>`
annotations keep their default owner. Inputs from `RequestView` cannot be shared into BLS.

## Known limitation

Triton r25.09 can unload a batching strategy library while scheduler callbacks are active.
Stop producers and drain requests before unloading it. Fully concurrent library unload
requires a Triton core fix; SDK state protection does not protect the library mapping.

## Thanks

Special thanks to [@heliar-k](https://github.com/heliar-k) for custom batching, sequence state,
tracing, response statistics, many other backend API additions, and the detailed API coverage
review that helped shape this release.

[Full changelog](https://github.com/rdcm/triton-ng/compare/0.3.1...0.4.0)
