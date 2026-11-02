# Model management and configuration

How a backend reads its own configuration and takes part in loading models.

## Backend config

`tritonserver --backend-config=<backend>,<key>=<value>` passes settings to one backend for all of
its models. `Backend::config` returns them as a `BackendConfig` in `initialize` and `finalize`:

```rust
// tritonserver --backend-config=asr,pool-size=8 --backend-config=asr,cache-dir=/cache
fn initialize(backend: &Backend) -> Result<(), Error> {
    let config = backend.config()?;
    let pool_size: usize = config.parameter_or("pool-size", 4)?;
    let cache = config.required_parameter("cache-dir")?;
    Ok(())
}
```

Values are strings: `parameter` returns the raw value, `parameter_or` parses it and falls back to
a default, `required_parameter` fails when the key is missing, `parameters` lists every pair.

Next to the keys from the command line Triton adds its own keys for every backend:
`backend-directory`, `min-compute-capability`, `auto-complete-config` and
`default-max-batch-size`. A key given without a backend name reaches every backend; a key given
for this backend wins over it.

`BackendConfig` is a copy: Triton keeps the config it was started with, and triton-ng reads it
without taking it. Keep what the models need in a `static` (`OnceLock`) set in `initialize`, see
the example backend.
