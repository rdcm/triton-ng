# Integration-test fixtures

These are test assets, not user examples. User examples live in
[`examples/minimal-backend`](../../../examples/minimal-backend) and
[`examples/batch-strategy`](../../../examples/batch-strategy).

- `backend/`: shared SDK setup and feature-specific Rust modules. The model's `fixture`
  parameter selects one execute implementation once at initialization. There is no execute
  function that switches across unrelated tests using request parameters.
- `batch-strategy/`: grouping and lifecycle/failure fixtures for custom batching.

Root `models/` contains fixture configs and root `docker-compose.yml` mounts their libraries.
Run `make build`, `make download-model`, `make docker-env-up`, then `make tests` from the root.
Tests run concurrently. Each test creates private model instances and shared-memory names;
fixture guards unload their models on teardown, and the test runner removes generated
directories after all tests finish (Triton may scan the whole repository during a test). Triton
uses `--strict-readiness=false` so expected model-load failures do not change server health
for unrelated tests; model readiness is tested independently.

## Client test groups

Client assertions are split by capability into `tests/*_tests.rs` files. Shared helpers live
in [tests/common](../common/mod.rs). Nextest runs up to eight tests concurrently to bound
server resource use, with no serial test groups or locks around whole tests.

## Feature fixtures

Tests use `Config::isolated(test_name, backend, models)` to create private copies of the
selected models. `config.model("template")` returns that test's model name. Templates stay
in root `models/`; generated `models/test_*/` directories are ignored by Git and removed
after the complete test run. Read-only BLS targets remain shared. Per-request
parameters inside a fixture select variants of that feature, such as an invalid shape or an
error response; they do not route into another feature's handler.

| Fixture module | Models / test group |
|---|---|
| [backend/src/fixtures/backend_config.rs](backend/src/fixtures/backend_config.rs) | `fixture_backend_config` |
| [backend/src/fixtures/backend_state.rs](backend/src/fixtures/backend_state.rs) | `fixture_backend_state`, `fixture_backend_state_other` |
| [backend/src/fixtures/batching.rs](backend/src/fixtures/batching.rs) | `batch_strategy`, `batch_strategy_lifecycle`, `batch_strategy_init_error`, `batch_strategy_init_panic`, `batch_strategy_batch_panic`, `batch_strategy_include_error`, `batch_strategy_include_panic`, `batch_strategy_drop_panic`, `batch_strategy_refuse` |
| [backend/src/fixtures/bls_mutation.rs](backend/src/fixtures/bls_mutation.rs) | `fixture_bls_mutation` |
| [backend/src/fixtures/bls_streaming.rs](backend/src/fixtures/bls_streaming.rs) | `fixture_bls_streaming` |
| [backend/src/fixtures/cancellation.rs](backend/src/fixtures/cancellation.rs) | `fixture_cancellation` |
| [backend/src/fixtures/concurrency.rs](backend/src/fixtures/concurrency.rs) | `cpu_parallel`, `device_blocking` |
| [backend/src/fixtures/errors.rs](backend/src/fixtures/errors.rs) | `fixture_errors` |
| [backend/src/fixtures/host_policy.rs](backend/src/fixtures/host_policy.rs) | `host_policy_pinned`, `host_policy_unknown`, `host_policy_default`, `host_policy_batching` |
| [backend/src/fixtures/inference.rs](backend/src/fixtures/inference.rs) | `mnist`, `mnist_auto` |
| [backend/src/fixtures/input_forward.rs](backend/src/fixtures/input_forward.rs) | `identity_forward` |
| [backend/src/fixtures/instance_details.rs](backend/src/fixtures/instance_details.rs) | `instance_details` |
| [backend/src/fixtures/iterative_sequence.rs](backend/src/fixtures/iterative_sequence.rs) | `mnist_steps` |
| [backend/src/fixtures/memory_buffers.rs](backend/src/fixtures/memory_buffers.rs) | `fixture_memory_buffers` |
| [backend/src/fixtures/memory_usage.rs](backend/src/fixtures/memory_usage.rs) | `memory_usage` |
| [backend/src/fixtures/metadata.rs](backend/src/fixtures/metadata.rs) | `fixture_metadata` |
| [backend/src/fixtures/model_control.rs](backend/src/fixtures/model_control.rs) | `fixture_model_control` |
| [backend/src/fixtures/policy_append.rs](backend/src/fixtures/policy_append.rs) | `fixture_policy_append` |
| [backend/src/fixtures/reschedule_refusal.rs](backend/src/fixtures/reschedule_refusal.rs) | `fixture_reschedule_refusal` |
| [backend/src/fixtures/response_statistics.rs](backend/src/fixtures/response_statistics.rs) | `fixture_response_statistics` |
| [backend/src/fixtures/sender_refusal.rs](backend/src/fixtures/sender_refusal.rs) | `fixture_sender_refusal` |
| [backend/src/fixtures/sequence.rs](backend/src/fixtures/sequence.rs) | `sequence_state`, `sequence_state_other`, `sequence_state_batched` |
| [backend/src/fixtures/shared_inputs.rs](backend/src/fixtures/shared_inputs.rs) | `fixture_shared_inputs` |
| [backend/src/fixtures/streaming.rs](backend/src/fixtures/streaming.rs) | `mnist_stream` |
| [backend/src/fixtures/tracing.rs](backend/src/fixtures/tracing.rs) | `fixture_tracing` |

`backend/src/lib.rs` contains common initialization/state;
`backend/src/fixtures/mod.rs` declares the feature modules and model-level dispatch;
`common.rs` holds ordinary BLS request/response helpers. Those helpers do not select test
scenarios. The strategy lifecycle log stays in `target/traces` alongside Triton's trace files.

## Limits of the checks

The README fixture checks compilation, not execution of every sample hook. Integration tests
exercise the minimal example and the feature fixtures. No new benchmark measurements are
claimed by this refactor. The documented r25.09 tracing and strategy-unload limitations remain.
