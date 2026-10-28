# Patina test suite (TDD specification)

These tests define the contract for M1 (`rust-dsl -> patina dialect`),
M2 (`dialect verify + -> PTX lowering`), and M3 (pipelined mainloop).
All of them compile and run; the suite going green *is* the milestone.

## Layout

| File | Layer | Runs when | What it pins down |
|---|---|---|---|
| `crates/patina/tests/layout.rs` | Rust, host-side | `cargo test` (M1a: layout algebra) | TMA 4D-view derivation, smem descriptor fields, idesc packing, **K-advance = 16 units (regression for the `da += 8` bug)**, TMEM lane addressing, tile byte counts |
| `crates/patina/tests/golden_v1_gemm.rs` | Rust → dialect | `cargo test` (M1b: trace + printer) | the full public DSL API surface + text-exact dialect output |
| `crates/patina/tests/golden/v1_gemm.mlir` | dialect | compared by the test above; later `patina-opt %s` | the golden dialect syntax for the v1 warp-specialized GEMM |
| `dialect/tests/verifier.mlir` | dialect | `patina-opt --verify-diagnostics --split-input-file` (M1c, M3) | every structural rule: role legality, expect_tx byte match, TMEM alloc/dealloc pairing, phase ordering, role/block warp budget; M3: wait_stage role rule + barrier-kind spelling, expect_tx_stage byte match, yield arity |
| `crates/patina/tests/ffi_verify.rs` | dialect via C ABI | `cargo test` (M2b) | traced dialect verifies through `patina::ffi::verify`; broken text returns diagnostics |
| `crates/patina/tests/lower_ptx.rs` + `crates/patina/tests/golden/v1_gemm.ptx` | dialect → PTX | `cargo test` (M2c) | trace → emit → `patina::ffi::lower_to_ptx` matches the golden PTX; instruction inventory pinned (2 TMA copies, 2 UMMA steps with scaleC 0→1, 4 TMEM loads, barrier choreography); golden compiles under `ptxas -arch=sm_100a` |
| `crates/patina/tests/golden_v2_gemm_pipelined.rs` + `crates/patina/tests/golden/v2_gemm_pipelined.mlir` | Rust → dialect | `cargo test` (M3: pipeline trace + printer) | the M3 DSL surface (Pipeline/StageBuf/LoopIx/Scalar/Val<u32>, `for_each` runtime loop, runtime mbarrier phases) + text-exact pipelined dialect output |
| `crates/patina/tests/lower_ptx_v2.rs` + `crates/patina/tests/golden/v2_gemm_pipelined.ptx` | dialect → PTX | `cargo test` (M3-lowering) | pipelined lowering: `scf.for` → counted PTX loop with runtime phase flip, staged TMA/UMMA/barrier addressing (`shl.b32 %tmp, %s, 3` / `, %s, 13`), dynamic `scale_c`; staged tiles live in dynamic shared memory (64 KiB ring > 48 KiB static limit); golden compiles under `ptxas -arch=sm_100a` |
| `harness/gemm_check.cpp` | PTX → hardware | on a GB200 only (`./harness/build.sh`, see `docs/HARDWARE_TESTING.md`) | end-to-end numerical check: driver-API load of the golden PTX, `C = A·Bᵀ` vs a double-precision host reference; exits 0 PASS / 1 FAIL / 3 non-sm_100a |
| `crates/patina/tests/hw_run.rs` + `crates/patina/src/run.rs` | DSL → hardware | on a GB200 only (`cargo test --features cuda --test hw_run`) | the same loop in Rust: `patina::run` declares the `cuda.h` C ABI via FFI and links `libcuda.so`, tracing→lowering→launching→checking in one `cargo test`; skips vacuously off-Blackwell |
| `crates/patina/tests/run_smoke.rs` | driver plumbing | any GPU (`cargo test --features cuda --test run_smoke`) | runs a trivial `vec_add` PTX through `patina::run::Cuda` (load→launch→copy→check) on the *current* device, so the driver-API path is proven locally before GB200; also asserts tcgen05 kernels are gated on sm_100a |


## Running (once the code exists)

All of the below (plus the harness build) is automated by `./scripts/preflight.sh`.
Hardware (GB200) execution is covered in `docs/HARDWARE_TESTING.md`.

```bash
# M1a: layout algebra only — first tests that should pass
cargo test --test layout

# M1b: trace the v1 kernel and diff against the golden dialect file
cargo test --test golden_v1_gemm
BLESS=1 cargo test --test golden_v1_gemm   # rewrite golden after intentional changes

# M1c/M2a: dialect verifier contract (build patina-opt first, see mlir/README.md)
mlir/build/bin/patina-opt --verify-diagnostics --split-input-file dialect/tests/verifier.mlir
mlir/build/bin/patina-opt crates/patina/tests/golden/v1_gemm.mlir   # golden verifies clean

# M2b: C ABI + Rust FFI (build.rs compiles the mlir/ subproject into OUT_DIR)
cargo test --test ffi_verify

# M2c: dialect -> PTX lowering
cargo test --test lower_ptx
BLESS=1 cargo test --test lower_ptx    # rewrite the golden PTX after intentional changes
mlir/build/bin/patina-opt -lower-to-ptx crates/patina/tests/golden/v1_gemm.mlir
~/.local/cuda-12.8/bin/ptxas -arch=sm_100a crates/patina/tests/golden/v1_gemm.ptx

# M3: pipelined mainloop (v2 trace + dialect)
cargo test --test golden_v2_gemm_pipelined
mlir/build/bin/patina-opt crates/patina/tests/golden/v2_gemm_pipelined.mlir

# M3-lowering: pipelined dialect -> PTX
cargo test --test lower_ptx_v2
mlir/build/bin/patina-opt -lower-to-ptx crates/patina/tests/golden/v2_gemm_pipelined.mlir
~/.local/cuda-12.8/bin/ptxas -arch=sm_100a crates/patina/tests/golden/v2_gemm_pipelined.ptx
```

## Rules for changing tests

- The golden `.mlir` is changed only via `BLESS=1`, and the diff must be in
  the commit message. The v2 goldens (`v2_gemm_pipelined.mlir` / `.ptx`) are
  hand-edited instead: `BLESS=1` rewrites them RAW and wipes the comments
  that document the loop/barrier choreography.
- A verifier diagnostic text change requires updating both
  `mlir/include/patina/PatinaOps.td` (and its implementation in
  `mlir/lib/PatinaOps.cpp`) and `dialect/tests/verifier.mlir` in the same
  commit.
- `crates/patina/tests/layout.rs` asserts **numbers**, not implementations — if a number
  here changes, the corresponding line in `tcgen05_gemm.cu` should change too;
  treat any mismatch as a bug somewhere, never silently update both.

## Later (not M1)

- `trybuild` compile-fail tests for role-token misuse (`umma` in a Load role,
  TMEM borrow escaping an Epi closure) — currently documented as comments in
  `golden_v1_gemm.rs`.
- FileCheck-based lowering tests once M2 (`dialect -> PTX`) starts.
