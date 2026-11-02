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

## Backend attributes

`CustomBackend::attributes` tells Triton how to deploy the models of the backend:

```rust
fn attributes(_backend: &Backend) -> Result<BackendAttributes, Error> {
    Ok(BackendAttributes::new()
        .preferred_instance_group(InstanceGroupKind::Gpu, 1, &[])?
        .preferred_instance_group(InstanceGroupKind::Cpu, 2, &[])?
        .parallel_model_instance_loading(true))
}
```

- **Preferred instance groups** apply to models whose config has no `instance_group`. The first
  group is the most preferred; Triton takes the first one it can serve (a GPU group needs a GPU
  among its devices). A group of a config that names its kind but leaves out the count or the GPUs
  gets them from the first preferred group of that kind. A count of `0` and empty devices leave
  them to Triton. Devices are allowed only for GPU and AUTO groups, and counts and devices must fit
  the `int32` of the model config: triton-ng checks this when the attributes are built.
- **Parallel instance loading** lets Triton initialize the instances of a model at the same time.
  `model_instance_initialize` and `model_instance_finalize` then run on several threads at once:
  what they share (the model state, statics) must be safe for that. triton-ng's own part, starting
  and stopping the runtime of an instance, touches only that instance.

Triton calls the hook right after `initialize` and may call it again later. It ignores an error
from the hook and keeps the attributes it had, so triton-ng writes the error to the log. Without
the hook the backend sets nothing and Triton uses its defaults.

The example backend prefers two CPU instances: `mnist_stream` has no `instance_group` and gets
them, `mnist` names its group in its config and keeps it.
