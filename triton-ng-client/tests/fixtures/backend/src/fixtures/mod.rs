use super::*;

#[path = "backend_config.rs"]
mod backend_config;
use backend_config::echo_backend_config;
pub(super) use backend_config::execution_policy;
#[path = "backend_state.rs"]
mod backend_state;
#[path = "batching.rs"]
mod batching;
#[path = "bls_mutation.rs"]
mod bls_mutation;
use bls_mutation::wait_for_input_release;
#[path = "bls_streaming.rs"]
mod bls_streaming;
#[path = "cancellation.rs"]
mod cancellation;
use cancellation::{hold_until_cancelled, work_step};
#[path = "common.rs"]
mod common;
use common::{answer_done, argmax, host_inference, infer_image, respond};
#[path = "concurrency.rs"]
mod concurrency;
#[path = "errors.rs"]
mod errors;
#[path = "host_policy.rs"]
mod host_policy;
#[path = "inference.rs"]
mod inference;
#[path = "input_forward.rs"]
mod input_forward;
use input_forward::forward_to_identity;
#[path = "instance_details.rs"]
mod instance_details;
#[path = "iterative_sequence.rs"]
mod iterative_sequence;
use iterative_sequence::stream_steps;
#[path = "memory_buffers.rs"]
mod memory_buffers;
#[path = "memory_usage.rs"]
mod memory_usage;
#[path = "metadata.rs"]
mod metadata;
#[path = "model_control.rs"]
mod model_control;
#[path = "policy_append.rs"]
mod policy_append;
#[path = "reschedule_refusal.rs"]
mod reschedule_refusal;
#[path = "response_statistics.rs"]
mod response_statistics;
#[path = "sender_refusal.rs"]
mod sender_refusal;
#[path = "sequence.rs"]
mod sequence;
use sequence::count_sequence;
#[path = "shared_inputs.rs"]
mod shared_inputs;
use shared_inputs::kept_inputs;
#[path = "streaming.rs"]
mod streaming;
use streaming::{answer_with_streamed_digits, cancel_stream, stream_digits, stream_inference};
#[path = "tracing.rs"]
mod tracing;

pub(super) use memory_usage::{initialize_buffers, report_model};

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let _in_flight = InFlight::enter(state);
    match state.fixture.as_str() {
        "errors" => errors::execute(ctx, requests).await,
        "cancellation" => cancellation::execute(ctx, requests).await,
        "tracing" => tracing::execute(ctx, requests).await,
        "bls_streaming" => bls_streaming::execute(ctx, requests).await,
        "sender_refusal" => sender_refusal::execute(ctx, requests).await,
        "shared_inputs" => shared_inputs::execute(ctx, requests).await,
        "response_statistics" => response_statistics::execute(ctx, requests).await,
        "model_control" => model_control::execute(ctx, requests).await,
        "reschedule_refusal" => reschedule_refusal::execute(ctx, requests).await,
        "backend_config" => backend_config::execute(ctx, requests).await,
        "bls_mutation" => bls_mutation::execute(ctx, requests).await,
        "policy_append" => policy_append::execute(ctx, requests).await,
        "host_policy" => host_policy::execute(ctx, requests).await,
        "metadata" => metadata::execute(ctx, requests).await,
        "instance_details" => instance_details::execute(ctx, requests).await,
        "backend_state" => backend_state::execute(ctx, requests).await,
        "concurrency" => concurrency::execute(ctx, requests).await,
        "memory_buffers" => memory_buffers::execute(ctx, requests).await,
        "batching" => batching::execute(ctx, requests).await,
        "streaming" => stream_digits(ctx, state, requests).await,
        "sequence" => count_sequence(state, requests),
        "iterative_sequence" => stream_steps(state, requests),
        "input_forward" => forward_to_identity(ctx, state, requests).await,
        "inference" | "memory_usage" => inference::execute(ctx, requests).await,
        other => Err(format!("unknown test fixture '{other}'").into()),
    }
}
