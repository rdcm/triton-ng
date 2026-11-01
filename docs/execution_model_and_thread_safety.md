# Execution model and thread safety

**Instances and batches.** Triton runs `instance_group.count` instances of a model, and every
instance has its own multi-threaded runtime owned by triton-ng. `model_instance_execute` is an
`async fn`: triton-ng runs the future of every batch on the runtime of its instance and returns
to Triton at once, so an instance works on several batches at the same time. `.await` on a BLS
call frees the thread for the other batches of the instance; no thread waits for a model.

The model parameters `worker_threads` and `max_batches_in_flight` set the threads of every
runtime and how many batches an instance works on at once, see
[Model parameters](model_parameters.md).

**Keep the batch.** Dynamic batching is what makes Triton fast: `requests` is the batch the
scheduler collected, and the backend should process it as a whole, sending it to the downstream
models as batched BLS calls. Never split it into tasks per request: every downstream call would
run with batch size one.

**State.** `Backend::model_initialize` stores the state of the model (tokenizers, vocabularies,
parsed parameters) with `Model::set_state`. It is shared by all instances and reaches every batch
through `Context::state::<T>()`, together with the `Server` for BLS calls and Triton's
`MemoryManager`. `Model` is not available in `model_instance_execute`: read everything the batches
need from it in `model_initialize`.

`Backend::model_instance_initialize` can store state of one instance with
`ModelInstance::set_state`, for what must not be shared between instances, like caches of GPU
buffers of the instance's device. Batches of the instance get it through
`Context::instance_state::<T>()`. The runtime of the instance starts before
`model_instance_initialize` and is stopped again when it fails; the instance state is dropped after
`model_instance_finalize`, once every batch of the instance has finished.

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

**Handles and threads.** Every type that wraps a Triton object either owns it, and then its
`Drop` releases it, or borrows it from Triton, and then it has no `Drop` and its lifetime is
bounded by what Triton guarantees. Backend code only sees these types, never raw pointers.

| Type | Triton object | On drop | `Send` | `Sync` | Notes |
|---|---|---|---|---|---|
| `Request` | owns the request | sends an error response if none was sent, releases the request | yes | no | one request is used by one thread at a time |
| `Response` | owns the response | deletes it unless sent; the request gets an error response | yes | no | |
| `Input<'_>` | borrows from its `Request` | nothing | no | no | cannot outlive the request |
| `Output<'_>` | borrows from its `Response` | nothing | no | no | cannot be used after the response is sent |
| `InferenceRequest` | owns the BLS request and its input buffers | deletes it if it was never submitted | yes | no | Triton deletes it after a submit |
| `InferenceFuture`, `InferenceResult` | none | | yes | yes | the result owns its output buffers |
| `TritonError` | owns the error object, if any | deletes it | yes | yes | |
| `RamBuffer`, `PinnedRamBuffer`, `GpuBuffer`, `UninitGpuBuffer` | own memory from the memory manager | free it | yes | yes | must not outlive the server, see [Memory buffers](memory_buffers.md) |
| `Server`, `MemoryManager` | borrow the server and its memory manager | nothing | yes | yes | `Clone`; valid while any model is loaded |
| `Context` | none | | yes | yes | `Clone` |
| `Model`, `ModelInstance`, `BackendHandle` | borrow the model, instance, backend | nothing | no | no | used only in the init and finalize hooks |

A unit test in the crate (`thread_safety_of_every_handle_is_as_documented`) checks every row:
it does not compile when a type is `Send` or `Sync` where this table says it is not, or the other
way around.

**Lifetimes.** A model is not finalized while any of its requests is alive, and the server is not
destroyed while any model is loaded, so a batch never outlives the server it calls. Buffers from
the memory manager must not outlive the server either: keep them in the state of the model or of
the instance.

**Cancellation.** `Request::is_cancelled` tells whether the client gave up on the request; answer it
with `Request::respond_cancelled`. Triton reports a client that drops a gRPC call as a cancellation.

**Shutdown.** Before `Backend::model_instance_finalize`, triton-ng waits for every batch of the
instance and stops its runtime.

[`example/custom-backend`](../example/custom-backend/src/lib.rs) sends every image of a batch to
`mnist_onnx` over BLS and answers with the recognized digit. A request with the
`hold_until_cancelled` parameter is held until the client cancels it. The `mnist` model works on at
most four batches at once (`max_batches_in_flight`), and unloading it waits until the held
requests are answered. A request with the `memory` parameter is answered through memory manager
buffers cached in the instance state, see [Memory buffers](memory_buffers.md).
