# Request metadata and tracing

## Metadata of a request

Besides inputs and parameters, a request carries metadata that clients set on the gRPC
`ModelInferRequest`, and `Request` reads it:

| Getter | Client sets | Without it |
|---|---|---|
| `Request::id()` | `id` | `None` |
| `Request::correlation_id()` | `sequence_id` parameter, a number or a string | `None` |
| `Request::flags()` | `sequence_start`, `sequence_end` parameters | both `false` |
| `Request::timeout()` | `timeout` parameter, in microseconds | `None` |

The ID comes back in the response and is in Triton's logs and traces, so it is the ID to log.
The correlation ID and the flags are what stateful models use to tell sequences apart: a
`CorrelationId` is `Int(u64)` or `String`, and Triton has no zero or empty ID, so these are
`None`. `flags()` fails when Triton sets a flag this version of triton-ng does not know, rather
than ignore what may change the meaning of the request.

## Metadata of a BLS request

`InferenceRequest` sets the same metadata on a BLS call, and a model called with it reads it with
the getters above:

```rust
let mut inference = InferenceRequest::new(ctx.server(), "decoder", -1)?;
if let Some(id) = request.id()? {
    inference.set_id(&id)?;
}
if let Some(session) = request.correlation_id()? {
    inference.set_correlation_id(session)?;
    inference.set_flags(request.flags()?)?;
}
if let Some(timeout) = request.timeout()? {
    inference.set_timeout(timeout)?;
}
inference.set_priority(1)?;
inference.set_parameter("language", "ru")?;
```

Every setter refuses what Triton would read as something else instead of passing it on: a zero
or empty correlation ID (no ID), a string ID longer than the 128 bytes Triton allows, a zero
timeout (no timeout) or one that does not fit `u64` microseconds, and a parameter set twice
(Triton keeps both, and the called model would see one of them). The priority applies only to
models with `priority_levels` in their dynamic batching config: 1 is the highest, 0 or a value
past the levels is the default of the model.

`InferenceResult::parameter` reads the parameters the called model set on its response with
`Response::set_parameter`. Reading a BYTES parameter fails; the other parameters of the response
are still read.

## Traces

When Triton traces a client request (`--trace-config`, or the trace settings of the model), BLS
calls made for it are invisible in the trace unless they are its children. One call makes a BLS
request a child of the trace of a request:

```rust
let mut inference = InferenceRequest::new(ctx.server(), "encoder", -1)?;
inference.set_parent_trace(&requests[0])?;
let result = ctx.server().infer_async(inference).await?;
```

The child gets its own trace ID with `parent_id` set to the trace ID of the request, and Triton
writes it together with the parent. A request that is not traced, or a Triton built without
tracing, makes it a no-op. A batched BLS request has one parent: pick one request of the batch.

**Lifetimes.** Triton's frontend frees the trace data shared by a trace and its children with the
parent, and a child that reports after that writes into freed memory. So a request with a traced
child is released to Triton only after the child BLS request and its response are done, even when
the backend answers and drops the request before awaiting the BLS call. Answering is not delayed:
only the release, which frees the inputs of the request and lets the model unload.

**A submission that fails.** `TRITONSERVER_ServerInferAsync` takes the child trace only after the
request passes its checks, and its error does not tell whether it did. triton-ng then leaves the
trace to Triton: a submission rejected before the trace was taken leaks one small trace object
(about 200 bytes), instead of risking a double free, and triton-ng logs a warning. It happens only
when the request is traced, has a child trace from `set_parent_trace`, and Triton rejects it for
its content with `INVALID_ARG`, which in core `r25.09` is one of:

| Check of `InferenceRequest::Normalize` | Rejected request |
|---|---|
| requested outputs | an output the model config does not have |
| number of inputs | fewer than the required inputs of the model, or more than all its inputs |
| input names | an input the model config does not have |
| batch dimension, when `max_batch_size > 0` | an input without dimensions; inputs whose first dimensions differ; a first dimension greater than `max_batch_size` (shape tensors are exempt) |
| datatype | an input whose datatype differs from the config |
| shape | a shape without the batch dimension of another rank than `dims`, or a dimension that differs from a fixed one in `dims`; a `-1` dimension in the request |
| byte size | data whose total size is not the element count times the element size (except non-linear format inputs) |
| BYTES data in CPU memory | a length prefix split between buffers or running past the data, more or fewer strings than elements |
| correlation ID, for a model with a `CONTROL_SEQUENCE_CORRID` control | a string ID to a model with an integer control, or an integer ID to one with a string control |

Every one of them fails the same BLS call every time it is made, so a leak is a sign of a broken
backend or of a backend out of sync with the config of the model it calls. Nothing else leaks: an
unknown model fails `InferenceRequest::new` before there is a trace; a model that is not ready or
unloading, a full queue, a missing correlation ID of a sequence model, a timeout in the queue, a
failed execution and a cancellation all fail after Triton took the trace, and Triton releases it.

[`example/custom-backend`](../example/custom-backend/src/lib.rs) makes every BLS call to
`mnist_onnx` a child of its request. With the `echo_metadata` parameter it answers with the
metadata of the request in response parameters, and with `forward_metadata` it calls itself over
BLS with metadata derived from the request and passes the echo on. The integration tests trace
`mnist` into `target/traces` (see `docker-compose.yml`) and look for the child traces there.
