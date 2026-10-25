# Patina dialect (MLIR / TableGen)

This directory contains the TableGen definition of the `patina` MLIR dialect
(M1c): `PatinaDialect.td`, `PatinaTypes.td`, `PatinaAttrs.td`,
`PatinaOps.td`, plus the verifier contract `tests/verifier.mlir`.

## Build status: deferred

**These files are written but not built.** Building them requires an
LLVM/MLIR development tree (headers, `MLIRIR`, `MLIRParser` libraries) with
`mlir-tblgen` to generate the C++ from the `.td` files, plus a C++17
toolchain to compile the dialect registration and the verifier hooks, and a
`patina-opt` driver (parser + `--verify-diagnostics`) to run:

```bash
patina-opt --verify-diagnostics --split-input-file tests/verifier.mlir
```

No MLIR toolchain is installed on this machine, so the M1c build is deferred
per the milestone plan. The ops/types/attrs defined here cover exactly what
appears in the golden file `crates/patina/tests/golden/v1_gemm.mlir`; the
verifier doc strings in `PatinaOps.td` carry the exact diagnostics that
`tests/verifier.mlir` expects.

## Change rules (from docs/TESTING.md)

- A verifier diagnostic text change requires updating both
  `PatinaOps.td` and `tests/verifier.mlir` in the same commit.
- The golden `.mlir` is changed only via `BLESS=1 cargo test`, with the diff
  in the commit message.

## Notes for the implementer

- `!patina.tensor` shape printing (`128x32`) and bare space keywords
  (`smem`, `gmem`) need custom parser/printer C++; the `assemblyFormat`
  lines in the `.td` files are structural sketches.
- `#patina.umma_k_major<swizzle_none>` prints only the swizzle; the space
  parameter defaults to `smem` in the custom parser.
- The Rust front-end aliases `#patina.umma_k_major<swizzle_none>` as `#lay`;
  the dialect parser must accept the alias (standard MLIR attribute aliasing).
