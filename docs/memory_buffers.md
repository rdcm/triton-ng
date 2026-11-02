# Memory buffers

By default a BLS call goes through plain CPU memory: inputs come from a `Vec<u8>`, and every
requested output is written into a `Vec<u8>` that the result takes over. When one model's output
goes straight into the next model, as in a decoding loop, the round trips through the CPU take
most of the call time. Buffers from Triton's memory manager let a backend:

- send inputs from pinned memory, which Triton copies to the GPU faster than plain memory;
- keep an output on the GPU and pass it as the input of the next BLS call;
- pass an input of the request being answered on to a BLS call without copying it, wherever
  Triton keeps it;
- reuse the same memory across calls and batches instead of allocating it every time.

## Memory manager

Every batch gets Triton's memory manager from `Context::memory`:

| Method | Returns | Memory | CPU access |
|---|---|---|---|
| `ram(len)` | `RamBuffer` | plain CPU memory | `&[u8]` / `&mut [u8]`, zeroed |
| `pinned(len)` | `PinnedRamBuffer` | page-locked CPU memory from Triton's pinned pool | `&[u8]` / `&mut [u8]`, zeroed |
| `gpu(device_id, len)` | `UninitGpuBuffer` | GPU memory from Triton's CUDA pool on `device_id` | none |

```rust
let mut tokens = ctx.memory().pinned(batch * 8)?;
tokens.copy_from_slice(&token_bytes);
let state = ctx.memory().gpu(ctx.device_id(), batch * 320 * 4)?;
```

The memory comes from the server's own pools, triton-ng does not use the CUDA runtime:

- **Pinned** memory comes from the pool set by `--pinned-memory-pool-byte-size` (256 MB by
  default). When the pool has no free block of `len` bytes, or is disabled (`0`), `pinned` fails:
  there is no fallback to plain memory.
- **GPU** memory comes from the pool set by `--cuda-memory-pool-byte-size` (64 MB per GPU by
  default). The pool has a fixed size and is shared with Triton itself. When it has no free
  block, `gpu` fails with an INTERNAL error: Triton does not report it as `UNAVAILABLE`, so a
  full pool cannot be told apart from other failures by the error code. Size the pool for
  everything the backend keeps on the GPU, caches included.
- `pinned` and `gpu` also fail on a Triton built without GPU support.

Every buffer frees its memory on drop. Zero-length buffers do not reach Triton at all.

## GPU buffers

A GPU buffer cannot be read or written from the CPU, and triton-ng has no way to copy into it.
Its content comes only from a model, so it has two types:

- `UninitGpuBuffer`: allocated, content undefined. It can only receive the output of a BLS call.
- `GpuBuffer`: a model has written its output into it. It can be passed as an input of the next
  BLS call, or turned back into an `UninitGpuBuffer` with `into_uninit` to receive another
  output.

```
memory.gpu(device, len) ──▶ UninitGpuBuffer ──add_requested_output_into──▶ BLS call
                                  ▲                                            │
                                  │ into_uninit                       take_gpu │
                                  │                                            ▼
                               GpuBuffer ◀───────────────────────────────── GpuBuffer
                                  │
                                  └──add_input_with_buffer──▶ next BLS call
```

Neither type gives access to the memory, so GPU data cannot be read on the CPU by mistake, and
memory with undefined content cannot become a model input.

## Inputs

`InferenceRequest::add_input_with_buffer` (or `add_input` + `append_input_buffer`) takes any
`InputBuffer`:

| Input buffer | Triton reads it as |
|---|---|
| `Vec<u8>` | CPU memory |
| `RamBuffer` | CPU memory |
| `PinnedRamBuffer` | pinned CPU memory |
| `GpuBuffer` | GPU memory of its device |
| `SharedInput` | the memory the request keeps the input in, CPU or GPU |
| `Arc<T>` of any of them | the memory of `T` |

```rust
inference.add_input_with_buffer("x", DataType::Int64, &[batch, 1], tokens)?;
inference.add_input_with_buffer("hi", DataType::Fp32, &[1, batch, 320], Arc::clone(&state))?;
```

triton-ng does not copy the input. Triton copies it to the memory the model needs, if it is
somewhere else.

The request owns the buffer until Triton releases the request, which can happen after the
response has already arrived. Pass an `Arc` to keep using the buffer, for example in a cache:
the `Arc` is shared until the release, and becomes unique again only after it.

## Inputs of the request

`Input::share` turns an input of the request being answered into a `SharedInput`, a BLS input
that points into the memory the request keeps the tensor in. A backend in the middle of a GPU
chain (a step of an ensemble, or a model called by another backend) passes a GPU tensor on this
way, and a CPU one without the copy `Input::bytes` would make.

```rust
let features = request.get_input("features")?.share()?;
let shape = features.shape().to_vec();
inference.add_input_with_buffer("features", features.datatype(), &shape, features)?;
```

The data of a request input lives until the request gets its final response and is released,
whichever is later. Triton's header promises the release, but the gRPC and HTTP frontends free
the data of a client request once its final response is sent (r25.09: the gRPC handler reuses
the protobuf the inputs point into; HTTP frees the request body), and CUDA shared memory regions
are held only by the response; an ensemble frees a step input when the step request is deleted.
So while a `SharedInput` of a request is alive:

- the request is not released;
- its final response is held back: `Response::send`, `respond_*`, `ResponseSender::finish` and
  the error response of a dropped request return at once, and the response is sent when the
  last shared input of the request is dropped. A failure to send it then is only logged, and
  the client gets an error response instead. The responses of a stream that are not final go
  out at once.

A BLS request owns its inputs until Triton releases it, which can be after its result has
arrived, so the client may get the response a little after the backend sent it. A shared input
kept for longer, in a cache, holds the response back that long. An input cannot be shared once
the request has its final response.

`SharedInput` cannot read the memory: like `GpuBuffer` it is only an input. Read the input on
the CPU with `Input::bytes` instead.

## Outputs

An output requested with `add_requested_output` is written into CPU memory that the result
takes over without copying it, and is read with `bytes`, `as_fp32_vec` and the other `as_*`
methods.

`add_requested_output_into` writes the output into a buffer of the backend instead:

```rust
inference.add_requested_output_into("ho", state)?;          // UninitGpuBuffer
inference.add_requested_output_into("logits", logits)?;     // PinnedRamBuffer
let mut result = ctx.server().infer_async(inference).await?;

let scores = result.output("logits")?.as_fp32_vec()?;       // RAM and pinned outputs are readable
let state: GpuBuffer = result.take_gpu("ho")?;              // hands the buffer back
let logits: PinnedRamBuffer = result.take_pinned("logits")?;
```

- **The size must match exactly.** The output of a request is its own rows of the batch, even
  when Triton's dynamic batcher merged it with other requests, so its size is known in advance:
  `rows × the other dimensions × element size`. When the model returns another size, the BLS
  call fails. Outputs whose size depends on the data have to use `add_requested_output`.
- **One output per buffer.** A buffer receives one output, so `Server::infer_stream` refuses
  a request with outputs in buffers: every response of a stream needs memory of its own, see
  [Streaming](streaming.md).
- **Taking the buffer back.** `take_gpu`, `take_pinned` and `take_ram` hand the buffer over.
  Taking it as the wrong type fails and leaves it in the result; after a take, `bytes` and
  `as_*` of that output fail. A buffer left in the result is freed with the result.
- **GPU outputs are not readable.** `bytes` and `as_*` fail for an output in a GPU buffer.

Writing an output into a buffer is not zero-copy: the model writes into its own memory first,
and its backend copies the output into the buffer. For onnxruntime that is a GPU-to-GPU copy
when the model runs on the GPU, and a copy through a pinned staging buffer when a CPU model
writes into a GPU buffer. It still avoids the round trip through the CPU.

## Caching buffers

triton-ng has no cache of its own: what to keep, how much, and for how long depends on the
backend. Keep the cache in the state of the instance (`ModelInstance::set_state`,
`Context::instance_state`), so every instance caches buffers of its own device:

```rust
#[derive(Default)]
struct Buffers {
    gpu: Mutex<Vec<GpuBuffer>>,
}

impl Buffers {
    fn gpu(&self, memory: &MemoryManager, device_id: i32, len: usize) -> Result<UninitGpuBuffer, Error> {
        let mut cached = self.gpu.lock().map_err(|_| "buffer cache is poisoned")?;
        match cached.iter().position(|b| b.len() == len && b.device_id() == device_id) {
            Some(index) => Ok(cached.swap_remove(index).into_uninit()),
            None => memory.gpu(device_id, len),
        }
    }

    fn put_gpu(&self, buffer: GpuBuffer) -> Result<(), Error> {
        self.gpu.lock().map_err(|_| "buffer cache is poisoned")?.push(buffer);
        Ok(())
    }
}

fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
    instance.set_state(Buffers::default())
}
```

```rust
let buffers = ctx.instance_state::<Buffers>()?;
inference.add_requested_output_into("ho", buffers.gpu(ctx.memory(), ctx.device_id(), len)?)?;
let mut result = ctx.server().infer_async(inference).await?;
buffers.put_gpu(result.take_gpu("ho")?)?;
```

What a real cache needs on top of this:

- **A limit in bytes.** The pools are fixed and shared with Triton; buffers of every batch size
  pile up in the cache without one. Drop the least recently used buffers above the limit.
- **Inputs are released later than outputs.** An output buffer is back in the backend's hands
  when `take_*` returns. An input buffer is held by the request until Triton releases it, which
  can be after the response. Keep inputs as `Arc` and hand one out again only when
  `Arc::strong_count` is 1.

## Lifetimes and threads

Buffers must not outlive the server. Keep them in the state of the model or of the instance:
both are dropped when the model or the instance is finalized, before the server stops.

`MemoryManager` and every buffer type are `Send + Sync`. The memory manager can be used from any
thread: Triton's pools are thread-safe.

## Example

[`example/custom-backend`](../example/custom-backend/src/lib.rs) answers a request with the
parameter `memory` set to `ram`, `pinned` or `gpu` from a buffer of that kind: the image goes in
a RAM or pinned buffer, the logits are written into a buffer cached in the instance state, and
the response reports how many buffers the instance has allocated. `gpu_input` writes the logits
into a GPU buffer and passes it, through an `Arc`, as the input of two concurrent requests to
`identity_gpu` (Triton's `identity` backend on a GPU instance); the backend drops its own `Arc`
before the requests finish and checks that both copies equal the logits. `gpu_forward` passes the
GPU logits to `identity_forward`, a model of the example backend that shares its GPU input with a
request to `identity_gpu` and answers with what comes back; the caller checks that the input was
in GPU memory and equals the logits. `wrong_size` writes the logits into a buffer of the wrong
size and gets an error.

A request with the parameter `shared_input` set to a number of milliseconds passes its image on
to `mnist_onnx` as a `SharedInput`, checks the logits against those of a copy, and keeps the
shared image that long after answering: the client gets the response only then.
