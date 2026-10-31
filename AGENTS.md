# Working on triton-ng

## Use the Makefile only

Build, lint, format, test, and run the project only through `make` targets. Do not run `cargo`,
`docker`, `git submodule`, or other tools directly, even for a one-off check. Every step must
be reproducible by hand with `make`; a step that exists only as an ad-hoc command breaks that.

If no target does what you need, stop and ask instead of working around it. A new target is
added to the `Makefile` only after it is agreed.

| Target | What it does |
|---|---|
| `make format` | `cargo sort` and `cargo fmt` for the whole workspace |
| `make lint` | `cargo clippy` |
| `make tests` | `cargo nextest` for the whole workspace |
| `make build` | release build |
| `make download-model` | downloads the mnist models used by the tests |
| `make docker-env-up` / `docker-env-down` / `docker-env-clean` | local Triton with the example backend |
| `make docker-logs` | logs of the local Triton |
| `make update-submodules` | updates `deps/` to the branches in `.gitmodules` and copies the Triton headers into `triton-ng-sys/include` |
| `make gen-grpc-client` | regenerates the gRPC client in `triton-ng-client/src/generated` from `proto/` |
| `make publish` | publishes the crates to crates.io in dependency order; `DRY_RUN=1` only packages and verifies them |
| `make setup-protoc` | installs `protoc` |

## Tests

Unit tests live next to the code they test, in a `#[cfg(test)] mod tests` at the end of the
source file. Tests that need a running Triton go to the crate's `tests/` directory and run
against `make docker-env-up`.

Library unit tests are linked without `libtritonserver`, so they must not reach Triton
symbols, including `TritonError`'s `Drop`. Test the logic on plain Rust types and convert to
`TritonError` outside the tested part.

## Public API documentation

Every public item (types, traits, functions, methods, macros, modules) has a doc comment that
says what it does, who owns what, and when it fails. Everything a backend author calls also
has an `# Examples` section with a usage example. Keep examples in sync with the code when
the API changes.

## Submodules

The commits recorded for `deps/` must be the code of the branches named in `.gitmodules`.
The branch must match the Triton image the project targets (`r25.09` for
`nvcr.io/nvidia/tritonserver:25.09`, see `docker-compose.yml`).

Update submodules only with `make update-submodules` and commit the new `deps/` commits
together with any `.gitmodules` change. `.gitmodules` alone proves nothing: it can name one
branch while the recorded commits come from another.
