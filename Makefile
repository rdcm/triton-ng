setup-protoc:
	sudo dnf install protobuf-compiler
	protoc --version

docker-env-up:
	docker compose up -d

docker-env-down:
	docker compose down

docker-env-clean:
	docker compose down -v

docker-logs:
	docker logs triton

format:
	cargo sort --workspace
	cargo fmt --all

lint:
	cargo clippy

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

tests:
	cargo nextest run --workspace --no-fail-fast

update-submodules:
	git submodule update --init --recursive
	git submodule update --remote
	cp deps/triton-core/include/triton/core/tritonbackend.h deps/triton-core/include/triton/core/tritonserver.h triton-ng-sys/include/triton/core/

gen-grpc-client:
	cargo run -p gen-grpc-client

publish:
	cargo publish --workspace $(if $(DRY_RUN),--dry-run)

build:
	cargo build --release

benchmarks: benchmarks-env-up
	cd benches && uv run benchmarks.py --results ../target/benchmarks
	$(MAKE) benchmarks-env-down

benchmarks-env-up: build
	mkdir -p $(addsuffix /1,$(wildcard benches/models/*)) target/benchmarks
	cp models/mnist_onnx/1/model.onnx benches/models/bench_onnx/1/model.onnx
	docker compose -f benches/docker-compose.yml up -d

benchmarks-env-down:
	docker compose -f benches/docker-compose.yml down

download-model:
	mkdir -p models/mnist/1
	mkdir -p models/mnist_onnx/1
	mkdir -p models/mnist_stream/1
	mkdir -p models/mnist_auto/1
	mkdir -p models/mnist_steps/1
	mkdir -p models/identity_gpu/1
	mkdir -p models/identity_forward/1
	mkdir -p models/memory_usage/1
	mkdir -p models/instance_details/1
	mkdir -p models/device_blocking/1
	mkdir -p models/cpu_parallel/1
	wget https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx
	mv mnist-12.onnx models/mnist_onnx/1/model.onnx
