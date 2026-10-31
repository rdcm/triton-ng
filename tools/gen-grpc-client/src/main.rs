use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let proto = root.join("proto");
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .out_dir(root.join("triton-ng-client/src/generated"))
        .compile_protos(
            &[
                proto.join("grpc_service.proto"),
                proto.join("model_config.proto"),
                proto.join("health.proto"),
            ],
            &[proto],
        )?;
    Ok(())
}
