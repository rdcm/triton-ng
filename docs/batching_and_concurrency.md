# Batching and concurrency

Why a batch runs to its end on the Triton thread of its instance, how Triton forms batches, and
what to tune when a model needs more concurrency.

## How Triton forms a batch

Everything below is from the core of the Triton release triton-ng targets (`deps/triton-core`,
r25.09). A model with `dynamic_batching` has one *batcher thread* and one *execution thread per
instance* (`backend_model_instance.cc`, `TritonBackendThread::BackendThread`). The execution
thread takes a batch (a `Payload`) from the batch queue of the model and calls
`TRITONBACKEND_ModelInstanceExecute` on it; it takes the next batch only after that call returns.

The batcher thread moves requests from the request queue into a batch only when there is a *slot*
for it (`dynamic_batch_scheduler.cc`, `WaitForPayloadSlotAvailable`; `rate_limiter.cc`,
`PayloadSlotAvailable`):

- for a model with only `max_queue_delay_microseconds` set (no priorities, timeouts or
  `max_queue_size`), when fewer than `2 × instances` batches are queued;
- otherwise, when an execution thread is waiting for a batch.

Until then requests accumulate in the request queue. When there is a slot, the batcher takes up
to `max_batch_size` requests; if fewer are queued it waits until the oldest of them is
`max_queue_delay_microseconds` old, then sends what it has (`GetDynamicBatch`).

So the size of a batch is **the number of requests that arrived while every execution thread of
the model was busy**. `max_queue_delay_microseconds` bounds how long a request waits for company
when a thread is free; it does not fill batches. A backend that returns from execute before its
batch is done frees the thread at once, and from then on every batch is whatever arrived in
`max_queue_delay_microseconds`: at a steady load that is one or two requests. This is what
onnxruntime, tensorrt and python_backend never do: their execute returns when the batch is done,
and Triton's back-pressure forms the next batch meanwhile.

## What triton-ng does

`model_instance_execute` is an `async fn`, and triton-ng runs its future to the end on the Triton
thread that called execute (`block_on` in `executor.rs`: the thread parks while the future waits
for BLS calls, the response callback of the BLS request unparks it). Execute returns when the
batch is done, exactly like a synchronous backend, so batches are formed by Triton's back-pressure.
triton-ng starts no threads and has no runtime: there is nothing to configure.

The future still wins over a synchronous backend inside the batch: several BLS calls awaited
together (`futures::future::join_all`, `tokio::join!`) run in Triton at the same time, so a backend
can send the sub-batches of a batch, or the stages of different requests, downstream at once
while the thread sleeps. What does not work inside the future: timers, I/O and spawning of
tokio or other runtimes, because nothing drives them. A blocking call blocks the instance.

triton-ng 0.1 and 0.2 ran every batch on a tokio runtime per instance and returned from execute
at once. The model parameters `worker_threads` and `max_batches_in_flight` of those versions are
gone; a model that still sets them fails to load with an error that says so.

## Tuning: `instance_group` count

The number of batches a model works on at once is the number of its instances. More instances
overlap the CPU work and the BLS waits of several batches, which raises throughput as long as the
load keeps every instance's batches full; past that point batches get smaller and the downstream
models run with smaller inputs. Measure with the average batch size
(`nv_inference_count / nv_inference_exec_count`) and the queue time of the model, which triton-ng
reports to Triton (see [Metrics](metrics.md)), and pick the smallest count that keeps the GPU busy.

For a model whose work per batch is dominated by GPU time of one downstream model there is
nothing to gain from more instances; for a model that alternates GPU-heavy and CPU- or
latency-bound stages (an encoder followed by a step-by-step decoder loop), two to four instances
usually win.

## Measurements

A speech recognition backend (`asr`: an encoder followed by a step-by-step decoder, all TensorRT
models behind BLS; two instances, `max_batch_size: 8`, `max_queue_delay_microseconds: 5000`,
sub-batches of 4 to the encoder, then a BLS call per decoding step), RTX 5080, Triton 25.09. The client sends VAD utterances of a six-minute
recording (150 distinct, 2.4 s on average), `concurrency` of them at a time; three runs each,
requests per minute.

| Variant | concurrency 16 | concurrency 64 |
|---|---|---|
| synchronous `model_instance_execute` (triton-ng 717b560) | 14.3–14.6k | 17.4–17.6k |
| 0.2.0, runtime per instance, unlimited batches in flight | 14.3–14.6k | 14.4–14.9k |
| 0.2.0, `max_batches_in_flight: 1` | 14.5–14.6k | 17.3–17.6k |
| 0.2.0, `max_batches_in_flight: 2` | 13.8–14.3k | 18.7k |
| 0.2.0, `worker_threads: 4` | 14.1–14.4k | 14.2–14.9k |
| this version: batch on the Triton thread | 14.5–14.9k | 17.6–17.7k |

At concurrency 16 every variant is the same: two instances with eight requests each are never
all busy, batches are small either way. At concurrency 64 the 0.2.0 default and `worker_threads`
lose 17 %. Two things happen at once: the encoder gets 9 % more calls for the same utterances
(smaller sub-batches), and the unlimited batches in flight flood the downstream models, whose
queue time goes from 84 µs to 5.6 ms and whose output copies take 7.8 ms instead of 1.6 ms per
call (TensorRT runs one execution thread per GPU, and it waits for the output buffers of all
the concurrent responses). Limiting the batches in flight restores the batches, and two per
instance (four batches on two instances) beats one by 6 % because the decoder loop of one batch
overlaps the encoder of another: that is what more instances give, below.

The same with `instance_group` count instead of a runtime parameter, this version:

| Variant | concurrency 16 | concurrency 64 |
|---|---|---|
| 2 instances | 14.5–14.9k | 17.6–17.7k |
| 4 instances | 13.9–14.2k | 18.2–18.6k |
| 8 instances | 13.8–13.9k | 14.7–15.0k |
| synchronous, 4 instances | 13.4–13.5k | 18.6k |

Four instances win 5 % at a load that keeps them busy and lose 5 % at a load that does not;
eight lose 15 % at any load: the batches are too small again. The count is a property of the
load, and the metrics triton-ng reports (batch size, queue time) are what to pick it by.

With requests like the production ones, VAD segments merged into chunks of 15–22 s (18 distinct,
1000 requests), the encoder dominates and the difference grows:

| Variant | concurrency 16 | concurrency 64 |
|---|---|---|
| synchronous, 2 instances | 2.8–2.9k | 3.2–3.3k |
| 0.2.0, unlimited batches in flight | 2.6–2.7k | 2.3–2.4k (−28 %) |
| 0.2.0, `max_batches_in_flight: 2` | 2.6–2.7k | 3.5–3.7k |
| this version, 2 instances | 2.8–3.0k | 3.3–3.4k |
| this version, 4 instances | 2.6–2.7k | 3.6–3.8k (+14 %) |
| synchronous, 4 instances | 2.6–2.7k | 3.5–3.8k |

Under the 0.2.0 default the encoder's output copies take 28 ms per call instead of 3 ms and its
queue 19 ms instead of 0.1 ms: the batches in flight are limited by nothing but the client.

## Concurrency inside a batch

The `asr` backend cuts its batch into sub-batches of four and sent them downstream one after
another. Sending them all at once (`futures::future::join_all` over the sub-batch futures, the
batch future still runs on its Triton thread) keeps the batches Triton formed and overlaps the
decoder loop of one sub-batch with the encoder of another:

| Variant | utterances, concurrency 16 | utterances, 64 | chunks, 16 | chunks, 64 |
|---|---|---|---|---|
| synchronous, 2 instances | 14.3–14.6k | 17.4–17.6k | 2.8–2.9k | 3.2–3.3k |
| this version, 2 instances, sub-batches one by one | 14.6–14.9k | 17.3–17.5k | 2.8–2.9k | 3.3k |
| this version, 2 instances, sub-batches joined | 15.7–16.6k | 19.2–20.3k | 3.0–3.1k | 3.8k |
| this version, 4 instances, sub-batches joined | 14.3–15.7k | 18.7–19.1k | 2.6–2.8k | 3.6–3.7k |

The joined sub-batches win at every load, and more instances add nothing to them. That is what
the future is for: concurrency inside the batch, where it costs no batching, instead of
concurrency between batches, which Triton already provides with instances.
