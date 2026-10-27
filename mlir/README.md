# mlir/ — Patina MLIR dialect build

CMake subproject that builds the `patina` MLIR dialect from the TableGen
sources in `include/patina/` plus the `patina-opt` driver.

## Prerequisites

- LLVM/MLIR 21 development tree with `mlir-tblgen`, `libMLIR.so`, and the
  CMake packages (`MLIRConfig.cmake`). On this machine the Ubuntu
  `llvm-21-dev` / `libmlir-21-dev` 21.1.8 debs are extracted under
  `~/.local/llvm-21/root/usr/lib/llvm-21` (referred to as `$L` below).
- gcc-13 / g++-13 (the MLIR headers are built with it; do not use the system
  default gcc if it is newer).
- cmake + ninja/make.

Note: the Ubuntu `llvm-21-dev` deb ships `libLTO.so -> libLTO.so.21.1` but no
package contains the real `libLTO.so.21.1`, which makes `LLVMExports.cmake`
fail its imported-file check. Rebuild it once from the shipped archive:

```bash
L=~/.local/llvm-21/root/usr/lib/llvm-21
g++-13 -shared -fPIC -o $L/lib/libLTO.so.21.1 \
  -Wl,--whole-archive $L/lib/libLLVMLTO.a -Wl,--no-whole-archive \
  -L$L/lib -lLLVM -Wl,-soname,libLTO.so.21.1
```

## Configure and build

```bash
L=~/.local/llvm-21/root/usr/lib/llvm-21
cmake -S mlir -B mlir/build \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DMLIR_DIR=$L/lib/cmake/mlir -DLLVM_DIR=$L/lib/cmake/llvm
LD_LIBRARY_PATH=$L/lib cmake --build mlir/build -j
```

`LD_LIBRARY_PATH=$L/lib` is needed at build time (mlir-tblgen links
libLLVM/libMLIR) and at run time for `patina-opt`.

## Verify

```bash
L=~/.local/llvm-21/root/usr/lib/llvm-21
export LD_LIBRARY_PATH=$L/lib

# golden kernels must verify clean and round-trip (v1 single-tile, v2
# pipelined mainloop)
mlir/build/bin/patina-opt crates/patina/tests/golden/v1_gemm.mlir
mlir/build/bin/patina-opt crates/patina/tests/golden/v2_gemm_pipelined.mlir

# verifier contract: 5 expected-error cases (M1/M2) + 3 pipelined-mainloop
# cases (M3), plus 2 valid kernels
mlir/build/bin/patina-opt --verify-diagnostics --split-input-file \
  dialect/tests/verifier.mlir

# lower the golden kernels to PTX (goes to stdout, followed by an empty
# `module {}` from the driver)
mlir/build/bin/patina-opt -lower-to-ptx crates/patina/tests/golden/v1_gemm.mlir
mlir/build/bin/patina-opt -lower-to-ptx crates/patina/tests/golden/v2_gemm_pipelined.mlir
```

The same lowering is reachable from Rust via `patina::ffi::lower_to_ptx`;
the crate's build script builds this CMake project into the cargo OUT_DIR
and links `libpatina_mlir_c` (see `crates/patina/build.rs`).

## Layout

- `include/patina/*.td` — dialect, enum/attr, type, and op definitions
  (moved here from `dialect/` in M2a).
- `include/patina/PatinaDialect.h`, `PatinaOps.h` — umbrella headers; the op
  header also declares the name-resolution helpers used by verifiers.
- `lib/` — dialect registration (attr/type parse dispatch), custom op
  parse/print, the five structural verifiers, custom type parse/print, and
  `LowerToPTX.cpp` (the v1 `patina.kernel` → PTX emitter).
- `lib/capi/` — `libpatina_mlir_c.so`: the C ABI (`include/patina-c/`).
- `tools/patina-opt/` — MlirOptMain driver with the patina dialect registered.

`mlir/build/` is git-ignored.
