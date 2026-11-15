# Minimal Rust backend

Read `INPUT`, call another model over BLS, and return `OUTPUT`.

Run from the repository root:

```bash
make build
make download-model
make docker-env-up
```

Send FP32 `INPUT` with shape `[1, 10]` to `minimal_backend` on `localhost:8001` using the client
snippet in the root README. The response contains the same values in `OUTPUT`.

`models/minimal_backend` configures the Rust backend. Its BLS target,
`models/minimal_backend_target`, uses Triton's standard `identity` backend: it copies input to
output without running an ML model. That target requires names `INPUT0`/`OUTPUT0`; the Rust
example maps them to its public `INPUT`/`OUTPUT`.

For dynamic batching, see [the minimal batching strategy](../batch-strategy).
The root `minimal_batching` model combines this backend with that strategy. Both models run
on CPU; the shared root service also loads GPU fixtures (see Development in the
root README for requirements).
