# Patina test suite (TDD specification)

These tests are written **before** the implementation. They define the
contract that M1 (`rust-dsl -> patina dialect`) must satisfy. Nothing here
compiles or runs yet — the tests going green *is* the milestone.

## Layout

| File | Layer | Runs when | What it pins down |
|---|---|---|---|
| `crates/patina/tests/layout.rs` | Rust, host-side | `cargo test` (M1a: layout algebra) | TMA 4D-view derivation, smem descriptor fields, idesc packing, **K-advance = 16 units (regression for the `da += 8` bug)**, TMEM lane addressing, tile byte counts |
| `crates/patina/tests/golden_v1_gemm.rs` | Rust → dialect | `cargo test` (M1b: trace + printer) | the full public DSL API surface + text-exact dialect output |
| `crates/patina/tests/golden/v1_gemm.mlir` | dialect | compared by the test above; later `patina-opt %s` | the golden dialect syntax for the v1 warp-specialized GEMM |
| `dialect/tests/verifier.mlir` | dialect | `patina-opt --verify-diagnostics --split-input-file` (M1c) | every structural rule: role legality, expect_tx byte match, TMEM alloc/dealloc pairing, phase ordering, role/block warp budget |

## Running (once the code exists)

```bash
# M1a: layout algebra only — first tests that should pass
cargo test --test layout

# M1b: trace the v1 kernel and diff against the golden dialect file
cargo test --test golden_v1_gemm
BLESS=1 cargo test --test golden_v1_gemm   # rewrite golden after intentional changes

# M1c: dialect verifier contract
patina-opt --verify-diagnostics --split-input-file dialect/tests/verifier.mlir
```

## Rules for changing tests

- The golden `.mlir` is changed only via `BLESS=1`, and the diff must be in
  the commit message.
- A verifier diagnostic text change requires updating both
  `dialect/PatinaOps.td` and `dialect/tests/verifier.mlir` in the same commit.
- `crates/patina/tests/layout.rs` asserts **numbers**, not implementations — if a number
  here changes, the corresponding line in `tcgen05_gemm.cu` should change too;
  treat any mismatch as a bug somewhere, never silently update both.

## Later (not M1)

- `trybuild` compile-fail tests for role-token misuse (`umma` in a Load role,
  TMEM borrow escaping an Epi closure) — currently documented as comments in
  `golden_v1_gemm.rs`.
- FileCheck-based lowering tests once M2 (`dialect -> PTX`) starts.
