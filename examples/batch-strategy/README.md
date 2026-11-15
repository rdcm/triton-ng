# Minimal custom batching strategy

The by-key strategy from [heliar-k's original PR #34](https://github.com/rdcm/triton-ng/pull/34),
without the test fixture's failure modes, logging, or lifecycle instrumentation.

`ByKey` accepts the first request of a batch, then includes adjacent requests with the same
string `batch_key`. Missing/non-string keys form their own group. For a queued `A, A, B, A`,
the strategy groups `[A, A]`, `[B]`, `[A]`; it cannot look past the `B`.

Build and run from the workspace root:

```bash
make build
make download-model
make docker-env-up
```

The build produces `target/release/libtriton_example_batch_strategy.so`. Root Compose mounts it
at `/strategies/example-batchstrategy.so`. The `minimal_batching` model combines this strategy
with the minimal Rust backend: send FP32 `INPUT` with shape `[1, 10]` and a string `batch_key`
parameter, and receive the same data in `OUTPUT`. Multiple requests can share a batch; each
request still has batch size one. The BLS target remains `minimal_backend_target`.

To attach the strategy to another model, enable dynamic batching and configure the path:

```text
max_batch_size: 4
dynamic_batching { max_queue_delay_microseconds: 200000 }
parameters: {
  key: "TRITON_BATCH_STRATEGY_PATH"
  value: { string_value: "/strategies/example-batchstrategy.so" }
}
```

Accepting the head is required for progress: refusing it leaves the queue stalled until a
configured timeout. Always drain active requests/callbacks before unloading the library;
Triton r25.09's library shutdown ordering limitation still applies.

Integration tests use this exact library in `batch_strategy_example` to observe batch
composition. `minimal_batching` separately verifies the minimal backend and strategy together.
The lifecycle/error-injection strategy stays in `triton-ng-client/tests/fixtures/batch-strategy`.
