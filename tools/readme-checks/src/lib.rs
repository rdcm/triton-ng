//! Compile-only checks for the exact Rust snippets in the root README.
#![allow(dead_code, unused_imports)]
include!(concat!(env!("OUT_DIR"), "/readme.rs"));
