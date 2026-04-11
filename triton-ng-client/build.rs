fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .out_dir("src/generated/")
        .compile_protos(
            &[
                "../proto/grpc_service.proto",
                "../proto/model_config.proto",
                "../proto/health.proto",
            ],
            &["../proto"],
        )?;
    Ok(())
}
