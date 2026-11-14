use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let root = manifest
        .ancestors()
        .nth(4)
        .ok_or("workspace root not found")?;
    let readme = root.join("README.md");
    println!("cargo:rerun-if-changed={}", readme.display());
    let content = std::fs::read_to_string(readme)?;
    let mut generated = String::new();
    let mut count = 0;
    for (index, block) in content.split("```rust ignore\n").skip(1).enumerate() {
        let (source, _) = block.split_once("```").ok_or("unclosed Rust fence")?;
        generated.push_str(&format!(
            "mod snippet_{index} {{\nuse triton_ng::*;\n{source}\n}}\n"
        ));
        count += 1;
    }
    if count == 0 {
        return Err("README contains no checked Rust snippets".into());
    }
    let output = PathBuf::from(std::env::var("OUT_DIR")?).join("readme.rs");
    std::fs::write(output, generated)?;
    Ok(())
}
