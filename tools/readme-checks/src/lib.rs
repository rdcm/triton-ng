//! Compile-only checks for the exact Rust snippets in the root README and guides.
#![allow(dead_code, unused_imports, unused_variables)]
include!(concat!(env!("OUT_DIR"), "/readme.rs"));
