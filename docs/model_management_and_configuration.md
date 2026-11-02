# Model management and configuration

How a backend reads its own configuration, completes the configs of its models, and loads and
unloads models.

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

## Completing the model config

`CustomBackend::auto_complete_config` fills in what the `config.pbtxt` of a model leaves out, like
`auto_complete_config` of python_backend. A model then needs little more than its backend:

```
name: "mnist_auto"
backend: "custom_backend"
```

```rust
fn auto_complete_config(_model: &Model, config: &mut AutoCompleteConfig) -> Result<(), Error> {
    if config.has_input("Input3") {
        return Ok(());
    }
    config.set_max_batch_size(8)?;
    config.set_dynamic_batching()?;
    config.add_input("Input3", DataType::Fp32, &[1, 28, 28])?;
    config.add_output("DIGIT", DataType::Int32, &[1])?;
    Ok(())
}
```

- **What can be set.** Triton takes back only the max batch size, inputs and outputs, the scheduler
  when the config names none, and `decoupled` (`set_decoupled`); `AutoCompleteConfig` offers
  nothing else. `add_optional_input` adds an input requests may leave out.
- **The config wins.** What the config already sets is only confirmed: an input or output of the
  config gets the data type and dims it leaves out, and fails the load when they differ; a smaller
  max batch size than the configured one, dynamic batching over a sequence or ensemble scheduler
  and the other `decoupled` fail too. The hook is shared by every model of the backend, so check
  with `has_input` and `has_output` what a model already has.
- **When.** triton-ng calls the hook right before `model_initialize`, which already sees the
  completed config (`Model::config_json`, `Model::is_decoupled`). Triton validates it after
  `model_initialize` and fails the load when it is incomplete. A hook that changes nothing does
  not touch the config. Models cannot be loaded or unloaded from the hook.
- **Off switch.** `tritonserver --disable-auto-complete-config` turns completion off for every
  backend: the hook is not called and the config must be complete.

The example backend completes `mnist_auto`, which has only its backend and parameters; the other
example models have complete configs and the hook leaves them alone.

## Loading and unloading models

A backend loads and unloads models of the server like python_backend's `pb_utils.load_model` and
`pb_utils.unload_model`:

```rust
// From the model repository, or reloaded when its files changed.
server.load_model("punctuation", None, &[])?;

// With another config.
server.load_model("punctuation", Some(r#"{"backend": "onnxruntime", "max_batch_size": 8}"#), &[])?;

// From files instead of the repository: they need a config.
let onnx = std::fs::read("/weights/punctuation_v2.onnx")?;
server.load_model(
    "punctuation_v2",
    Some(r#"{"backend": "onnxruntime"}"#),
    &[("file:1/model.onnx", &onnx)],
)?;

server.unload_model("punctuation_v1", false)?;
server.unload_model("asr_ensemble", true)?; // with its composing models
```

- **Explicit model control only.** Triton must run with `--model-control-mode=explicit`, and the
  server must have started: a model loaded at startup (`--load-model`) cannot load others from its
  hooks, Triton answers `Unavailable`. Load models from `model_instance_execute`, or from the hooks
  of models loaded later.
- **Loading blocks.** `load_model` returns when the model is ready or has failed, which takes
  seconds for a big model. In `model_instance_execute` it blocks a thread of the runtime of the
  instance, and the other batches on it wait. Call it through `tokio::task::spawn_blocking` to keep
  them going.
- **Unloading does not wait.** `unload_model` returns at once; Triton unloads the model when its
  requests in flight are answered. A model can unload itself from `model_instance_execute`.
- **Files.** Every file is named `file:<path>`, with `<path>` relative to the model directory
  (`file:1/model.onnx`). Triton writes them to a
  temporary directory during the call. triton-ng checks the names before the call: no absolute
  paths, no `..`, no repeated names, and a config whenever there are files.
- **Not in finalize hooks.** `finalize`, `model_finalize` and `model_instance_finalize` run inside a
  load or unload of Triton; both calls fail there with `Unsupported`, as in python_backend.

A model that fails to load keeps a failed state in Triton. With strict readiness (the default)
the whole server is then not ready (`/v2/health/ready`), and unloading the model does not clear
it. A name that is not in the repository and comes without files fails before that and leaves
no state. Check what you load before you load it.

The example backend loads a copy of `mnist_onnx` from its file as `mnist_from_files` on the request
parameter `load_model`, and unloads a model on `unload_model`.
