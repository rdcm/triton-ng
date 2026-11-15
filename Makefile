# Install the protobuf compiler using dnf and print its version.
setup-protoc:
	sudo dnf install protobuf-compiler
	protoc --version

# Start local Triton. Run build and download-model first; wait for model readiness before testing.
docker-env-up:
	docker compose up -d

# Stop and remove the local Triton service.
docker-env-down:
	docker compose down

# Stop local Triton and remove its volumes.
docker-env-clean:
	docker compose down -v

# Show local Triton container logs.
docker-logs:
	docker logs triton

# Sort workspace manifests and format all Rust code.
format:
	cargo sort --workspace
	cargo fmt --all

# Run Clippy.
lint:
	cargo clippy

# Build workspace documentation; treat documentation warnings as errors.
doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# Run workspace unit and integration tests. Integration tests require local Triton.
tests:
	@set -eu; \
	TRITON_NG_TEST_RUN=$$(cat /proc/sys/kernel/random/uuid); \
	export TRITON_NG_TEST_RUN; \
	trap 'rm -rf -- models/test_"$$TRITON_NG_TEST_RUN"_*' EXIT; \
	cargo nextest run --workspace --no-fail-fast

# Restore submodules to the branches in .gitmodules and refresh vendored Triton headers.
update-submodules:
	git submodule update --init --recursive
	git submodule update --remote
	cp deps/triton-core/include/triton/core/tritonbackend.h deps/triton-core/include/triton/core/tritonserver.h triton-ng-sys/include/triton/core/

# Regenerate the gRPC client from protobuf definitions; requires protoc.
gen-grpc-client:
	cargo run -p gen-grpc-client

# Publish workspace crates to crates.io. Use make publish DRY_RUN=1 to package and check only.
publish:
	cargo publish --workspace $(if $(DRY_RUN),--dry-run)

# Build the workspace in release mode, including examples, test libraries, and README and guide snippet checks.
build:
	cargo build --release

# Build, start benchmark Triton, run the benchmark suite with uv, and stop the service.
benchmarks: benchmarks-env-up
	cd benches && uv run benchmarks.py --results ../target/benchmarks
	$(MAKE) benchmarks-env-down

# Build and start benchmark Triton on ports 9000–9002; requires download-model first.
benchmarks-env-up: build
	mkdir -p $(addsuffix /1,$(wildcard benches/models/*)) target/benchmarks
	cp models/mnist_onnx/1/model.onnx benches/models/bench_onnx/1/model.onnx
	docker compose -f benches/docker-compose.yml up -d

# Stop and remove the benchmark Triton service.
benchmarks-env-down:
	docker compose -f benches/docker-compose.yml down

# Download the MNIST ONNX model and create model version directories.
download-model:
	mkdir -p $(addsuffix /1,$(wildcard models/*))
	wget https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx
	mv mnist-12.onnx models/mnist_onnx/1/model.onnx
