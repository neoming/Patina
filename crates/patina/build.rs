//! Build script: compiles the patina MLIR dialect C-ABI library
//! (`mlir/` CMake subproject) into OUT_DIR and links `libpatina_mlir_c`.
//!
//! The LLVM/MLIR dev tree location defaults to the deb extraction at
//! `$HOME/.local/llvm-21/root/usr/lib/llvm-21` and can be overridden with the
//! `PATINA_LLVM_PREFIX` environment variable.
//!
//! Configure runs once (when no CMakeCache.txt exists); each build runs an
//! incremental `cmake --build`, and cargo reruns this script when any file
//! under `mlir/` (excluding `mlir/build/`) changes.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mlir_src = manifest
        .join("../../mlir")
        .canonicalize()
        .expect("mlir/ subproject not found");

    println!("cargo:rerun-if-env-changed=PATINA_LLVM_PREFIX");
    watch_dir(&mlir_src);

    let llvm = env::var("PATINA_LLVM_PREFIX").map(PathBuf::from).unwrap_or_else(|_| {
        let home = env::var("HOME").expect("HOME not set");
        PathBuf::from(home).join(".local/llvm-21/root/usr/lib/llvm-21")
    });
    assert!(
        llvm.join("lib/cmake/mlir/MLIRConfig.cmake").exists(),
        "MLIRConfig.cmake not found under {} (set PATINA_LLVM_PREFIX)",
        llvm.display()
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build = out_dir.join("mlir-build");

    if !build.join("CMakeCache.txt").exists() {
        run(
            Command::new("cmake")
                .arg("-S")
                .arg(&mlir_src)
                .arg("-B")
                .arg(&build)
                .arg("-DCMAKE_C_COMPILER=gcc-13")
                .arg("-DCMAKE_CXX_COMPILER=g++-13")
                .arg("-DCMAKE_BUILD_TYPE=Release")
                .arg(format!("-DMLIR_DIR={}/lib/cmake/mlir", llvm.display()))
                .arg(format!("-DLLVM_DIR={}/lib/cmake/llvm", llvm.display()))
                // mlir-tblgen links libLLVM/libMLIR at build time.
                .env("LD_LIBRARY_PATH", llvm.join("lib")),
            "cmake configure",
        );
    }
    run(
        Command::new("cmake")
            .arg("--build")
            .arg(&build)
            .arg("--target")
            .arg("patina_mlir_c")
            .arg("-j")
            .env("LD_LIBRARY_PATH", llvm.join("lib")),
        "cmake build patina_mlir_c",
    );

    let lib_dir = build.join("lib");
    assert!(
        lib_dir.join("libpatina_mlir_c.so").exists(),
        "libpatina_mlir_c.so was not produced"
    );
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=patina_mlir_c");
    // Build-tree rpath so test binaries find the .so without LD_LIBRARY_PATH.
    // (The .so itself carries an rpath to the LLVM lib dir via
    // CMAKE_BUILD_RPATH in mlir/CMakeLists.txt.)
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}

fn watch_dir(dir: &Path) {
    for entry in std::fs::read_dir(dir).expect("cannot read mlir/") {
        let path = entry.unwrap().path();
        if path.file_name().unwrap() == "build" {
            continue; // our own out-of-tree build dir
        }
        if path.is_dir() {
            watch_dir(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {what}: {e}"));
    assert!(status.success(), "{what} failed with {status}");
}
