use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let root = manifest
        .ancestors()
        .nth(2)
        .ok_or("workspace root not found")?;
    let docs = root.join("docs");
    println!("cargo:rerun-if-changed={}", docs.display());
    let mut paths = vec![root.join("README.md")];
    for entry in std::fs::read_dir(docs)? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "md") {
            paths.push(path);
        }
    }
    paths.sort();
    let mut generated = String::new();
    let mut count = 0;
    for (page, path) in paths.iter().enumerate() {
        println!("cargo:rerun-if-changed={}", path.display());
        let content = std::fs::read_to_string(path)?;
        if content.matches("```").count() % 2 != 0 {
            return Err(format!("unclosed code fence in {}", path.display()).into());
        }
        for (index, block) in content.split("```").skip(1).step_by(2).enumerate() {
            let (language, source) = block.split_once('\n').ok_or("unclosed code fence")?;
            if language
                .split(|c: char| c == ',' || c.is_whitespace())
                .next()
                != Some("rust")
            {
                continue;
            }
            generated.push_str(&format!(
                "// {}: block {index}\nmod page_{page}_snippet_{index} {{\nuse triton_ng::*;\nuse std::sync::{{Arc, Mutex}};\nuse std::time::Instant;\n{source}\n}}\n",
                path.display()
            ));
            count += 1;
        }
    }
    if count == 0 {
        return Err("documentation contains no checked Rust snippets".into());
    }
    let output = PathBuf::from(std::env::var("OUT_DIR")?).join("readme.rs");
    std::fs::write(output, generated)?;
    Ok(())
}
