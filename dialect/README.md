# Patina dialect (MLIR / TableGen)

The `patina` MLIR dialect. As of M2a the TableGen sources live in
`mlir/include/patina/` (`PatinaDialect.td`, `PatinaTypes.td`,
`PatinaAttrs.td`, `PatinaOps.td`) and the dialect builds via the CMake
subproject in `mlir/` — see `mlir/README.md` for build instructions.

This directory keeps the verifier contract `tests/verifier.mlir`, exercised
with:

```bash
patina-opt --verify-diagnostics --split-input-file dialect/tests/verifier.mlir
```

(With `--split-input-file` every split parses independently: attribute
aliases do not carry across splits, so each split that uses `#lay` defines it
locally, and no comment may contain the literal split marker.)

## Change rules (from docs/TESTING.md)

- A verifier diagnostic text change requires updating both
  `mlir/include/patina/PatinaOps.td` (implementation lives in
  `mlir/lib/PatinaOps.cpp`) and `dialect/tests/verifier.mlir` in the same
  commit.
- The golden `.mlir` is changed only via `BLESS=1 cargo test`, with the diff
  in the commit message.

## Notes

- `!patina.tensor` shape printing (`128x32`) and bare space keywords
  (`smem`, `gmem`) are custom parser/printer C++ in `mlir/lib/PatinaTypes.cpp`.
- The Rust front-end aliases `#patina.umma_k_major<swizzle_none>` as `#lay`
  (standard MLIR attribute aliasing).
