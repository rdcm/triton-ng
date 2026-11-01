# Model parameters of triton-ng

A model config can carry a `parameters` map. Triton does not interpret it: it hands the map to
the backend, and every backend reads its own keys (onnxruntime reads `intra_op_thread_count`, a
custom backend reads whatever `Model::config_parameter` asks for). triton-ng reads two keys from
it to set up the runtime of every model instance. Both are optional.

| Key | Default | What it sets |
|---|---|---|
| `worker_threads` | `1` | Threads of the runtime of every instance: how many batches of one instance run backend code at the same moment |
| `max_batches_in_flight` | unlimited (`0`) | Batches one instance works on at once |

Values are strings, as everywhere in `parameters`. A value that is not a number, or
`worker_threads` set to `0`, fails the instance and the model does not load.

## Example

```
name: "asr"
backend: "asr"
max_batch_size: 64

dynamic_batching {
  max_queue_delay_microseconds: 2000
}

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]

parameters: {
  key: "worker_threads"
  value: { string_value: "4" }
}

parameters: {
  key: "max_batches_in_flight"
  value: { string_value: "8" }
}
```

This model gets two instances. Each of them has its own runtime with four threads and works on
at most eight batches at once, so the model works on up to sixteen batches, eight running backend
code at the same moment. A model without both keys gets one thread per instance and no limit.

## How it works

Triton calls `model_instance_execute` of an instance with a batch and waits for the call to
return before it gives the instance the next batch. triton-ng starts the future of the batch on
the runtime of the instance and returns to Triton at once:

```
Triton            instance runtime (worker_threads threads)
  |  execute(batch 1)
  |-------------->  batch 1 starts, awaits BLS
  |  returns
  |  execute(batch 2)
  |-------------->  batch 2 starts while batch 1 waits
  |  returns
  ...
```

`.await` on a BLS call gives the thread to the other batches of the instance, so `worker_threads`
does not bound how many batches wait for models, only how many run backend code (tokenizing,
decoding, post-processing) at the same moment. Raise it when that code takes noticeable CPU time.

Without `max_batches_in_flight`, execute always returns at once: the instance takes every batch
Triton forms, and requests leave Triton's queue as soon as they arrive. Batches are then formed
only by `max_queue_delay_microseconds`, and queue policies (`max_queue_size`, timeouts, priorities)
see an empty queue. With `max_batches_in_flight: N`, the execute of the N+1-th batch waits until
one of the N batches has finished. Until then Triton gives the instance nothing, the requests stay
in Triton's queue, where they are batched together and its policies apply:

```
max_batches_in_flight: 2

Triton            instance runtime
  |  execute(batch 1) -> starts, returns
  |  execute(batch 2) -> starts, returns
  |  execute(batch 3) -> waits for a free slot     new requests queue up in Triton
  |                     batch 1 finishes
  |                  -> batch 3 starts, returns
```

Leave the limit out when the load is already bounded before Triton, e.g. by the consumers of a
message queue. Set it when Triton is the only place that sees the load.
