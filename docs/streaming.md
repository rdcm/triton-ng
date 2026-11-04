# Streaming

A decoupled model answers one request with any number of responses: tokens of an LLM, partial
results of streaming speech recognition, one result per item of a request. triton-ng covers both
sides of it: a backend can answer its requests with a stream (`ResponseSender`), and a backend can
read the stream of another model over BLS (`Server::infer_stream`). `triton-ng-client` reads
streams with `TritonClient::infer_stream`.

## Decoupled models

A model is decoupled when its config says so:

```
model_transaction_policy {
  decoupled: true
}
```

`Model::is_decoupled` reads it, and triton-ng reads it once for every instance. Only requests of a
decoupled model can get a stream; requests of other models get exactly one response, as everywhere
else in triton-ng.

What Triton allows for a decoupled model:

| Way to call it | Works |
|---|---|
| gRPC `ModelStreamInfer` (`TritonClient::infer_stream`) | yes |
| BLS `Server::infer_stream` | yes |
| BLS `Server::infer_async` | only when the model sends exactly one response, see [Reading a stream over BLS](#reading-a-stream-over-bls) |
| gRPC `ModelInfer` (`TritonClient::infer`) | no, Triton rejects it |
| HTTP `/v2/models/<name>/infer` | no: "HTTP end point doesn't support models with decoupled transaction policy" |
| Response cache | no, Triton refuses to load a decoupled model with `response_cache` enabled |

## Sending a stream

`Request::into_sender` turns a request into a `ResponseSender`, which sends any number of responses
and then exactly one final flag:

```rust
async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
    for request in requests {
        let text = request.get_input("TEXT")?.as_string()?;
        let mut sender = request.into_sender()?;

        for token in tokenize(&text) {
            if sender.is_cancelled()? {
                sender.respond_cancelled()?;
                return Ok(());
            }
            let mut response = sender.response()?;
            response
                .create_output("TOKEN", DataType::Int64, &[1])?
                .write_i64_vec(&[token])?;
            response.send()?;
        }
        sender.finish()?;
    }
    Ok(())
}
```

| Method | Sends | Final |
|---|---|---|
| `sender.response()` then `StreamResponse::send` | a response | no |
| `sender.final_response()` then `Response::send` | a response | yes |
| `sender.finish()` | nothing but the final flag | yes |
| `sender.respond_error(message)`, `respond_error_with_code(code, message)` | an error | yes |
| `sender.respond_cancelled()` | a CANCELLED error | yes |
| `sender.reschedule()` | nothing, the request comes back in a later batch | no, see [Rescheduling requests](model_management_and_configuration.md#rescheduling-requests) |
| dropping the sender | a generic error ("did not produce a final response") | yes |

- **Exactly one final flag.** Every final method consumes the sender, so nothing can be sent after
  the final flag, and a dropped sender sends one. Triton does not check this itself: core passes
  any number of responses and flags on to the frontend.
- **One response at a time.** `StreamResponse` borrows its sender mutably: responses are built
  and sent one after another, and the stream cannot be finished while a response is being built.
  A `StreamResponse` dropped without being sent is deleted, and the stream goes on.
- **Every response is a whole response.** It has its own outputs, every one written exactly once,
  and its own parameters (`StreamResponse::set_parameter`).
- **Errors end the stream.** Triton allows a decoupled model to send an error without the final
  flag; triton-ng has no such response. A model that must report a failed item and go on puts it
  into the outputs or parameters of a response.
- **The sender owns the request.** The inputs stay readable for the whole stream
  (`sender.request()`), and BLS calls can be children of the trace of the request
  (`inference.set_parent_trace(sender.request())`). The request is released when the sender is
  dropped, after the final flag. While an input of the request passed on with `Input::share` is
  alive, the final flag waits for it, see [Memory buffers](memory_buffers.md).
- **A model that is not decoupled** fails `into_sender`, and the request is answered with that
  error: its clients wait for exactly one response. A decoupled model can still answer with
  `Response::new(request)`: that is a stream of one response.

### Streams and batches

The stream of a request lives in the future of its batch, so **a batch is in flight until every
stream of it is finished**, and a long stream keeps its instance busy for as long as it lasts: run
as many instances as streams should go on at once. Keep the streams in the batch future, there is
no runtime to spawn tasks on. To stream several requests of a
batch at once, poll them together in that future (e.g. `futures::future::join_all`) rather than
spawning a task per request.

### Cancellation

A client cancels a stream by dropping its gRPC call; Triton then cancels every request of the call.
`ResponseSender::is_cancelled` tells whether the request is cancelled: check it between responses
and answer with `respond_cancelled`. Responses sent after the cancellation are not an error, the
client no longer reads them.

Core keeps the flag as a plain `bool` written by the cancelling thread and read by the backend
without synchronization, so a cancellation may be seen one check later. triton-ng cannot make this
stronger.

### How it is implemented

Checked against core `r25.09`:

- `TRITONBACKEND_ResponseNew(request)` and `TRITONBACKEND_ResponseNewFromFactory` create responses
  from the same response factory of the request. The sender owns the request, so it creates every
  response from the request; a `ResponseFactory` is needed only for the final flag without a
  response.
- `finish` creates a `TRITONBACKEND_ResponseFactory`, calls `ResponseFactorySendFlags(FINAL)`, which
  calls the response callback of the request with a null response, and deletes the factory. Like
  every final response, it goes through the `FinalResponse` of the request, which holds it back
  while a `SharedInput` of the request is alive.
  `ResponseFactoryNew` and `ResponseFactoryDelete` only copy and drop a `shared_ptr` to the factory
  of the request.
- `Response` and `StreamResponse` share one builder of the `TRITONBACKEND_Response` (`ResponseBody`):
  outputs, parameters, the check that every output is written, and the deletion of an unsent
  response. They differ in who owns them and in how they are sent: a `StreamResponse` at once, a
  `Response` through the `FinalResponse` of its request.

## Reading a stream over BLS

`Server::infer_stream` submits a request and returns an `InferenceStream` of its results, one per
response, in the order the model sent them:

```rust
let mut inference = InferenceRequest::new(ctx.server(), "tokenizer_stream", -1)?;
inference.add_input_with_data("TEXT", DataType::Bytes, &[1], text)?;
inference.add_requested_output("TOKEN")?;
inference.set_parent_trace(sender.request())?;

let mut stream = ctx.server().infer_stream(inference);
while let Some(result) = stream.next().await {
    let tokens = result?.output("TOKEN")?.as_i64_vec()?;
    // ...
}
```

- **Items.** Every response is one item: `Ok(InferenceResult)`, or `Err` with the code of an error
  response. The stream ends after the final flag of the model, also when the flag comes without a
  response. A request that fails to submit is a stream of that one error. A model that is not
  decoupled answers with one item.
- **`futures_core::Stream`.** `InferenceStream` implements it, so `StreamExt` combinators and
  `select!` work; `next()` is also an inherent method.
- **No backpressure.** Results wait in the stream until they are read, and nothing slows the model
  down: a stream that is not read keeps every response in memory.
- **Outputs need memory of their own.** Every response gets new CPU memory for its outputs, which
  its result takes over. Outputs requested into a buffer
  (`InferenceRequest::add_requested_output_into`, see [Memory buffers](memory_buffers.md)) are
  refused: the stream fails with `INVALID_ARG` and the request is not submitted.
- **`infer_async` expects one response.** It fails when the model sends more than one response
  ("more than one response") or none ("sent no response"). Earlier versions merged the outputs of
  all responses into one result; read them with `infer_stream` instead.

### Cancelling BLS requests

- `InferenceStream::cancel` asks Triton to cancel the request. The responses sent before still
  arrive, and how the request ends is up to the model, usually with a CANCELLED error. Cancelling
  a request that is already done does nothing.
- **Dropping an `InferenceStream` cancels its request**, so a stream that is no longer read does
  not keep the model busy.
- **Dropping the future of `infer_async` cancels its request** too, unless the model has already
  answered, e.g. when the client cancelled the request it was made for.

### How it is implemented

- The response callback of the request gets every response and sends its result into the stream
  (a tokio unbounded channel). With the final flag it drops the sending side, which ends the stream.
  `infer_async` uses the same callback, keeping one response and failing on a second one, and
  sends the result through a tokio oneshot channel, which needs no runtime.
- `TRITONSERVER_InferenceRequestCancel` needs a live request, and the release callback of the
  request deletes it. `RequestCancel` keeps the pointer behind a mutex shared by the future or
  stream and the release callback. The callback clears the pointer under the mutex before it deletes
  the request, so a cancellation never touches a deleted request; cancelling after that does
  nothing.
- The input buffers of the request are freed by the release callback, as for `infer_async`, and the
  parent of a traced request is kept until the request is released, see
  [Request metadata and tracing](request_metadata_and_tracing.md).

## Reading a stream from a client

`TritonClient::infer_stream` sends one request over gRPC `ModelStreamInfer`:

```rust
let mut stream = client
    .infer_stream("mnist_stream", None, [images], ["DIGIT"], InferOptions::default())
    .await?;
while let Some(response) = stream.next().await {
    println!("{:?}", response?.outputs);
}
```

- One gRPC call carries one request. The client closes its side after sending it, so the call ends
  when Triton has finished the request; the final flag itself is not sent to the client.
- An error response is an `Error::Inference` item, a gRPC error an `Error::Grpc` item.
- The timeout of `InferOptions` limits the whole stream.
- Dropping the stream cancels the gRPC call, and Triton cancels the request.

## Limitations

- No error responses without the final flag.
- No backpressure: an `InferenceStream` keeps every response that is not read yet.
- No outputs into buffers in a BLS stream.
- Decoupled models are reachable only over gRPC `ModelStreamInfer` and BLS; `triton-ng-client`
  sends one request per gRPC call.
- No response cache for decoupled models (a limitation of Triton).
- A cancellation may be seen one check later (a data race in core).
- A streaming batch keeps its instance busy until all its streams end.
- Sequence batching of decoupled models is tested only with iterative sequences (`mnist_steps`).

## Example

[`example/custom-backend`](../example/custom-backend/src/lib.rs) serves `mnist_stream`, a decoupled
model of the same backend: a request carries several images (`[-1, 1, 28, 28]`), and the model
streams the digit of every image, each from a BLS call to `mnist_onnx`. With the
`hold_until_cancelled` parameter it waits for the client to cancel the stream after the digits, and
with `unfinished` it drops its sender without the final flag.

The `mnist` model reads `mnist_stream` over BLS (its `stream_model` config parameter):
`stream_images` streams that many copies of its image and answers with the number of digits,
`via_infer_async` reads the same stream with `infer_async`, and `cancel_stream` (`cancel` or `drop`)
cancels the stream after its first result.
