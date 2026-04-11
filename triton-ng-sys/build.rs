use anyhow::Result;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let bindings = bindgen::Builder::default()
        .clang_arg("-I../deps/triton-core/include")
        .clang_arg("-xc++")
        .header("../deps/triton-core/include/triton/core/tritonbackend.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()?;

    let out_path = PathBuf::from(env::var("OUT_DIR")?);

    bindings.write_to_file(out_path.join("bindings.rs"))?;

    Ok(())
}
