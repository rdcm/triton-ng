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

## Defensive programming

Write all code as defensively as possible. A crash in a backend takes down the whole Triton
server with every model in it, so triton-ng never trusts its inputs, Triton or the backend
author:

- Encode ownership and lifetimes in types: consuming methods, RAII guards, lifetimes that tie a
  borrowed handle to its owner. Backend code must not be able to free, release or answer
  anything twice, or use it after that.
- Check everything Triton returns: null pointers, counts, sizes against shapes, datatypes, enum
  values it may add later. Validate every value the backend passes before handing it to Triton
  (sizes, names, NaN and infinities, duplicates), and fail with an error instead of relying on
  Triton to reject it.
- Verify the ownership of every C API object against the core source of the branch in
  `.gitmodules`, not the headers, and say in a comment who owns what and why it is safe.
- No `unwrap`, `expect`, `panic!` or indexing that can fail; every fallible step returns an error.
  Panics are caught at the FFI boundary, never relied on.
- When a failure cannot be reported (in `Drop`, in a Triton callback), log it and leave Triton
  in a consistent state; prefer leaking an object to freeing it when ownership is ambiguous.

## Naming

A Rust type that wraps a Triton object is named after it without the prefix:
`TRITONSERVER_<Name>` or `TRITONBACKEND_<Name>` is `<Name>` (`TRITONSERVER_Metric` is
`Metric`, `TRITONBACKEND_Backend` is `Backend`). No `Handle`, `Raw` or similar suffixes, and
one type per Triton object.

Exceptions, where the name is taken by a more important public type:

- `TRITONSERVER_Error` is `TritonError`: `Error` is the boxed error alias every backend uses.
- `TRITONSERVER_Parameter` used for metric labels is `MetricParameter`: `Parameter` is the
  public enum of parameter values.
- `RequestRelease` holds the `TRITONBACKEND_Request` of a `Request` a second time: it is not a
  wrapper but the shared right to release the request, kept by traced BLS children after the
  `Request` is dropped.

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
