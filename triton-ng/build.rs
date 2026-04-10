fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        let cuda_path = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_ROOT"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());

        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
    }
}
