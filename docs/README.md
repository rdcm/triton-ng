# Execution model and thread safety

**Instances and batches.** Triton runs `instance_group.count` instances of a model, and every
instance has its own multi-threaded runtime owned by triton-ng. `model_instance_execute` is an
`async fn`: triton-ng runs the future of every batch on the runtime of its instance and returns
to Triton at once, so an instance works on several batches at the same time. `.await` on a BLS
call frees the thread for the other batches of the instance; no thread waits for a model.

| Model parameter | Default | What it sets |
|---|---|---|
| `worker_threads` | 1 | Threads of the runtime of every instance: how many batches of one instance run backend code at the same moment |
| `max_batches_in_flight` | unlimited | Batches one instance works on at once. When they are all in progress, the instance takes the next batch only after one of them has finished, and Triton keeps the requests in its queue |

```
instance_group [ { count: 2, kind: KIND_CPU } ]

parameters: { key: "worker_threads"        value: { string_value: "4" } }
parameters: { key: "max_batches_in_flight" value: { string_value: "8" } }
```

So `count` sets how many independent runtimes there are, and `worker_threads` how parallel each
of them is.

**Keep the batch.** Dynamic batching is what makes Triton fast: `requests` is the batch the
scheduler collected, and the backend should process it as a whole, sending it to the downstream
models as batched BLS calls. Never split it into tasks per request: every downstream call would
run with batch size one.

**Limit on batches in progress.** Without `max_batches_in_flight`, requests leave Triton's queue as
soon as they arrive, so batches are formed only by `max_queue_delay_microseconds`, and queue
policies (`max_queue_size`, timeouts, priorities) see an empty queue. Bound the load before Triton
(e.g. by the consumers of a message queue), or set `max_batches_in_flight` to keep the requests in
Triton's queue, where they are batched and its policies apply.

**State.** `Backend::model_initialize` stores the state of the model (tokenizers, vocabularies,
parsed parameters) with `Model::set_state`. It is shared by all instances and reaches every batch
through `Context::state::<T>()`, together with the `Server` for BLS calls. `Model` is not available in
`model_instance_execute`: read everything the batches need from it in `model_initialize`.

**Requests are owned by the backend.** `model_instance_execute` gets `Vec<Request>`. A request can
be answered from any thread of the runtime.

**Exactly one final response.** `Response::new`, `Request::respond_error` and
`Request::respond_cancelled` consume the request, so it cannot be answered twice. A request dropped
without a response, including when the future returns an error or panics, gets a generic error
response. Every request is released to Triton when it is dropped. The error returned by
`model_instance_execute` is only logged: answer a request with `respond_error` to give the client
the reason.

**BLS is a future.** `Server::infer_async` submits the request to Triton right away and returns an
`InferenceFuture`. Triton wakes it from its own threads. Dropping the future does not cancel the
request: Triton finishes it and the result is discarded.

**What can move to another thread:**

| Type | `Send` | `Sync` | Notes |
|---|---|---|---|
| `Request`, `Response` | yes | no | one request is used by one thread at a time |
| `InferenceRequest`, `InferenceFuture` | yes | no | |
| `Server`, `Context` | yes | yes | `Clone` |
| `Model` | no | no | used only in the init and finalize hooks |

**Lifetimes.** A model is not finalized while any of its requests is alive, and the server is not
destroyed while any model is loaded, so a batch never outlives the server it calls.

**Cancellation.** `Request::is_cancelled` tells whether the client gave up on the request; answer it
with `Request::respond_cancelled`. Triton reports a client that drops a gRPC call as a cancellation.

**Shutdown.** Before `Backend::model_instance_finalize`, triton-ng waits for every batch of the
instance and stops its runtime.

[`example/custom-backend`](../example/custom-backend/src/lib.rs) forwards batches to another model
over BLS (models with `target_model`) and holds requests until they are cancelled (`cancellable`).
