# Integration-test fixtures

These are test assets, not user examples. User examples live in
[`examples/minimal-backend`](../../../examples/minimal-backend) and
[`examples/batch-strategy`](../../../examples/batch-strategy).

- `backend/`: shared SDK setup and feature-specific Rust modules. The model's `fixture`
  parameter selects one execute implementation once at initialization. There is no execute
  function that switches across unrelated tests using request parameters.
- `batch-strategy/`: grouping and lifecycle/failure fixtures for custom batching.
- `readme-checks`: compile-only extraction of every Rust block in the root README. No snippet
  is copied into a separate test source that can drift from the visible documentation.

Root `models/` contains fixture configs and root `docker-compose.yml` mounts their libraries.
Run `make build`, `make download-model`, `make docker-env-up`, then `make tests` from the root.
The client integration test group runs serially because model lifecycle, traces and shared
memory are server-wide. Root make target names and library names are preserved.

## Feature fixtures

Tests use `Config::fixture("name")` to select `fixture_name`. Existing dedicated models keep
explicit names (for example `mnist_stream`, `sequence_state`, `device_blocking`). Per-request
parameters inside a fixture select variants of that feature, such as an invalid shape or an
error response; they do not route into another feature's handler.

| Fixture module | Models / test group |
|---|---|
| [backend/src/backend_config.rs](backend/src/backend_config.rs) | `fixture_backend_config` |
| [backend/src/backend_state.rs](backend/src/backend_state.rs) | `fixture_backend_state`, `fixture_backend_state_other` |
| [backend/src/batching.rs](backend/src/batching.rs) | `batch_strategy`, `batch_strategy_lifecycle`, `batch_strategy_init_error`, `batch_strategy_init_panic`, `batch_strategy_batch_panic`, `batch_strategy_include_error`, `batch_strategy_include_panic`, `batch_strategy_drop_panic`, `batch_strategy_refuse` |
| [backend/src/bls_mutation.rs](backend/src/bls_mutation.rs) | `fixture_bls_mutation` |
| [backend/src/bls_streaming.rs](backend/src/bls_streaming.rs) | `fixture_bls_streaming` |
| [backend/src/cancellation.rs](backend/src/cancellation.rs) | `fixture_cancellation` |
| [backend/src/concurrency.rs](backend/src/concurrency.rs) | `cpu_parallel`, `device_blocking` |
| [backend/src/errors.rs](backend/src/errors.rs) | `fixture_errors` |
| [backend/src/host_policy.rs](backend/src/host_policy.rs) | `host_policy_pinned`, `host_policy_unknown`, `host_policy_default`, `host_policy_batching` |
| [backend/src/inference.rs](backend/src/inference.rs) | `mnist`, `mnist_auto` |
| [backend/src/input_forward.rs](backend/src/input_forward.rs) | `identity_forward` |
| [backend/src/instance_details.rs](backend/src/instance_details.rs) | `instance_details` |
| [backend/src/iterative_sequence.rs](backend/src/iterative_sequence.rs) | `mnist_steps` |
| [backend/src/memory_buffers.rs](backend/src/memory_buffers.rs) | `fixture_memory_buffers` |
| [backend/src/memory_usage.rs](backend/src/memory_usage.rs) | `memory_usage` |
| [backend/src/metadata.rs](backend/src/metadata.rs) | `fixture_metadata` |
| [backend/src/model_control.rs](backend/src/model_control.rs) | `fixture_model_control` |
| [backend/src/policy_append.rs](backend/src/policy_append.rs) | `fixture_policy_append` |
| [backend/src/reschedule_refusal.rs](backend/src/reschedule_refusal.rs) | `fixture_reschedule_refusal` |
| [backend/src/response_statistics.rs](backend/src/response_statistics.rs) | `fixture_response_statistics` |
| [backend/src/sender_refusal.rs](backend/src/sender_refusal.rs) | `fixture_sender_refusal` |
| [backend/src/sequence.rs](backend/src/sequence.rs) | `sequence_state`, `sequence_state_other`, `sequence_state_batched` |
| [backend/src/shared_inputs.rs](backend/src/shared_inputs.rs) | `fixture_shared_inputs` |
| [backend/src/streaming.rs](backend/src/streaming.rs) | `mnist_stream` |
| [backend/src/tracing.rs](backend/src/tracing.rs) | `fixture_tracing` |

`backend/src/lib.rs` contains common initialization/state and the model-level dispatch;
`common.rs` holds ordinary BLS request/response helpers. Those helpers do not select test
scenarios. The strategy lifecycle log stays in `target/traces` alongside Triton's trace files.

## Limits of the checks

The README fixture checks compilation, not execution of every sample hook. Integration tests
exercise the minimal example and the feature fixtures. No new benchmark measurements are
claimed by this refactor. The documented r25.09 tracing and strategy-unload limitations remain.
