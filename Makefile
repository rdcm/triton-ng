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

tests:
	cargo nextest run --workspace --no-fail-fast

update-submodules:
	git submodule update --init --recursive
	git submodule update --remote

gen-grpc-client:
	cargo build --manifest-path=triton-grpc-client/Cargo.toml --release

build:
	cargo build --release

download-model:
	mkdir -p models/mnist/1
	mkdir -p models/mnist_onnx/1
	wget https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx
	mv mnist-12.onnx models/mnist_onnx/1/model.onnx
