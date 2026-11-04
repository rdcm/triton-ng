# Model parameters of triton-ng

A model config can carry a `parameters` map. Triton does not interpret it: it hands the map to
the backend, and every backend reads its own keys (onnxruntime reads `intra_op_thread_count`, a
custom backend reads whatever `Model::config_parameter` asks for).

triton-ng itself reads none of them: how many batches a model works on at once is set by
`instance_group`, see [Batching and concurrency](batching_and_concurrency.md).

## Reading a parameter

```
parameters: {
  key: "batch_size"
  value: { string_value: "4" }
}
```

```rust
fn model_initialize(model: &Model) -> Result<(), Error> {
    let batch_size = model.config_parameter_or("batch_size", 4usize)?;
    let vocab_file = model.config_parameter("vocab_file")?.ok_or("vocab_file is not set")?;
    model.set_state(State { batch_size, vocab: read_vocab(&model.path(&vocab_file)?)? })?;
    Ok(())
}
```

Values are strings, as everywhere in `parameters`; `config_parameter_or` parses them into the
type of the default and fails when a value does not parse.
