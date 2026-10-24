# Patina — Design Proposal

**A Rust DSL for warp-specialized Blackwell (tcgen05) GEMM kernels, lowered through its own MLIR dialect.**

*Status: draft · 2026-10-24*

---

## 1. Summary

Patina is a Rust-embedded DSL for writing NVIDIA Blackwell (SM100) tensor-core
kernels — starting with single-CTA GEMM — with warp specialization as a
first-class programming model. Kernel code is ordinary Rust that *traces* into
a typed IR; the IR is emitted as **Patina dialect** MLIR, verified by a
dialect-aware tool (`patina-opt`), and (in later milestones) lowered through
MLIR's NVGPU/NVVM path to PTX and SASS.

The name: a patina is a thin oxide layer over metal — a Rust pun, NVIDIA green,
and the design philosophy in one word: *a thin typed layer over tcgen05/PTX,
not an optimizing compiler.*

## 2. Motivation

The reference kernel (`~/WorkSpace/tcgen05_gemm/tcgen05_gemm.cu`) is a minimal
single-CTA tcgen05 GEMM. Of its ~200 lines, roughly 30 express the algorithm;
the rest is encoding:

- **Descriptor bit-packing** — UMMA shared-memory descriptors (`>>4` fields,
  version bits) and instruction descriptors (format/major/M/N bit fields).
- **Layout arithmetic by hand** — the canonical swizzle-none K-major smem
  layout (`kc*128 + (r/8)*256 + (r%8)*16`) and the 4D TMA tensor-map view that
  reproduces it.
- **Implicit execution scopes** — `if (tid == 0)` for MMA issue, `tid < 32`
  for TMEM alloc, all 128 threads for `tcgen05.ld`. Violations compile fine
  and corrupt silently.
- **Untracked synchronization state** — mbarrier phase parities (0 vs 1),
  `expect_tx` byte counts, proxy fences, the 4-step TMEM dealloc sequence.

This is not hypothetical: the original reference contained a latent bug —
`da += 8` advanced the second MMA to K-slice 8..23 instead of 16..31 (fixed
to `+= 16` in the TMA rewrite). Hand descriptor arithmetic is exactly the
class of error a DSL should make unrepresentable.

## 3. Goals / Non-goals

**Goals**

- G1. Warp-specialized producer/consumer kernels (Load / MMA / Epilogue roles)
  as the primary model.
- G2. Type-checked execution scopes: an op that is illegal for a role does
  not compile.
- G3. Layouts, descriptors, TMA tensor maps, TMEM addressing, and barrier
  phases *derived*, never hand-encoded.
- G4. One abstraction level above PTX; `ptxas` remains the optimizer.
- G5. Everything inspectable: textual IR at every stage, golden-file tests.

**Non-goals (v1)**

- Autotuning, scheduling search, or a general GPU programming language.
- Runtime k-loops / multi-stage pipelines (needs IR loops — milestone 3).
- Clusters / `cta_group::2` (designed for, not implemented).
- Swizzled smem layouts (32B/64B/128B) beyond `SWIZZLE_NONE`.

## 4. Architecture

```
Rust trace (host)                verification split
  layout algebra ─┐
  scope tokens    │   Rust types: ergonomic, points at user code
  RAII handles    ▼
  Typed IR ─────────────────────────────────────────┐
    │                                               │
    ▼ emit                                          │ dialect verifiers:
  Patina dialect (.mlir) ──► patina-opt             │ structural, testable
    │                        (parse + verify)       │ without Rust
    ▼ lower (milestone 2+)                          │
  NVGPU/NVVM dialects ──► LLVM/NVPTX ──► PTX ──► ptxas ──► SASS
```

Two-stage staging model (like CuTe DSL's meta/object stages): shapes, layouts,
and trace-time `for` loops are plain Rust; anything wrapped in
`Val<T>` / `Ptr<T, Space>` / `Tensor<...>` becomes an IR node.

### Project layout

```
patina/
├── Cargo.toml                  # workspace
├── crates/
│   └── patina/                 # front-end: layout, trace IR, dialect printer
│       └── src/{layout, ir, ops, emit/mlir}.rs
├── dialect/                    # MLIR dialect definition (TableGen + C++)
│   ├── PatinaOps.td
│   ├── PatinaTypes.td
│   └── PatinaAttrs.td
├── tools/patina-opt/           # parser + verifier driver
├── examples/v1_gemm.rs         # the reference kernel in Patina
└── tests/                      # TDD spec, written before the implementation
    ├── TESTING.md              #   how the suite runs, change rules
    ├── layout.rs               #   host-side encoding math (first to go green)
    ├── golden_v1_gemm.rs       #   DSL API surface + golden comparison
    └── golden/v1_gemm.mlir     #   checked-in golden dialect output
(dialect/tests/verifier.mlir    # patina-opt verifier contract, +/- cases)
```

## 5. Language design

### 5.1 Typed tensors

`Tensor<dtype, shape, space, layout>` — layout is part of the type, so
UMMA-illegal states are unrepresentable and `smem_descriptor()` exists only
where legal. Layout algebra follows CuTe: `Layout = Shape ∘ Stride` with
composition; Blackwell swizzles are more layout functions.

### 5.2 Warp specialization as first-class structure

Roles are types with const-generic warp counts; a kernel is one structured
`switch (warp_id)`:

```rust
struct GemmRoles;
impl Roles for GemmRoles {
    type Load = Role<1>;   // warp 0        -> LoadThread token
    type Mma  = Role<1>;   // warp 1        -> MmaThread token
    type Epi  = Role<4>;   // warps 2..6    -> EpiWarp token
}
```

Capability is carried by the token:

| Role token | Warps | Has | Lacks (compile error) |
|---|---|---|---|
| `LoadThread` | elected thread, warp 0 | `tma_load`, `expect_tx` | `umma`, `tmem_ld`, `tmem_alloc` |
| `MmaThread` | elected thread, warp 1 | `umma_*`, `commit` | `tma_load`, `tmem_ld` |
| `EpiWarp` | full warps | `tmem_ld/st`, reg math, `st.global` | `umma`, `tma_load` |

### 5.3 Synchronization as typed objects

- `MBarrier<N>`: arrival count in the type; `wait(phase(k))` — parity is
  tracked, never a raw literal.
- `expect_tx` byte counts are derived from the tensors the copies cover;
  mismatch = error.
- Pipelines (`Pipeline<STAGES>`) wrap full/empty barrier rings with stage
  values carrying ring index + phase (milestone 3).

### 5.4 RAII resources

`blk.tmem_alloc::<f32, 128>()` allocates TMEM (warp-collective), emits
`relinquish_alloc_permit`, and on drop emits
`tcgen05.fence::before/after_thread_sync` + `bar.sync` + `dealloc`.
Leaks and missing fences are borrow errors at trace time.

### 5.5 Derived encodings

- `make_desc` / `make_idesc` become pure functions of layout + dtype types.
- K-advance comes from `layout.k_stride() >> 4`, never a literal.
- TMEM lane-field addressing (`warp_rank << 21`) comes from `warp.ld(&acc)`.
- Host-side 4D TMA tensor-map views are derived from the same layout value.

### 5.6 Fences

v1: explicit `fence_async_proxy()` / `fence_mbarrier_init()` calls, with the
IR verifier *warning* on missing generic→async proxy orderings.
Later: automatic insertion from memory-effect tracking.

### 5.7 Escape hatch

A raw-PTX IR node with declared operand metadata, so hand asm neither blocks
verification nor is needed once an op exists.

## 6. The Patina dialect (milestone-1 target)

Deliberately one level above PTX: no descriptors, no parities, no lane math —
but every op names its PTX realization in one sentence. Versioned from day one
(`patina.version` attr) so lowering-pass churn fails loudly.

**Types**

```mlir
!patina.tensor<f16, 128x32, smem, #patina.umma_k_major<swizzle_none>>
!patina.tmem<f32, 128>            !patina.mbarrier<1>
!patina.tma_map<f16, #layout>     !patina.frag<f32, 128>
!patina.ptr<f32, gmem>
```

**Ops**

| Op | Carries | Lowers to |
|---|---|---|
| `patina.kernel` | roles, block, arch, version | `.entry` + params |
| `patina.smem_alloc` | tensor type | static smem, `cvta.to.shared` |
| `patina.mbarrier.create` | arrival count | `mbarrier.init` + `fence.mbarrier_init` |
| `patina.tmem.alloc` / `.dealloc` | columns | `tcgen05.alloc`/`relinquish`, fence pair/`dealloc` |
| `patina.warp_specialize` + `patina.role` regions | role→warp range | `switch(warp_id)` |
| `patina.expect_tx` / `patina.tma_load` | bytes, coords, barrier | `arrive.expect_tx`, `cp.async.bulk.tensor.4d` |
| `patina.wait` | barrier + phase attr | `try_wait.parity` loop |
| `patina.umma.begin/step/advance_k/commit` | acc, operands, accum flag | descriptor packing, `tcgen05.mma`, `commit` |
| `patina.tmem_ld` / `patina.store` | shapes | `tcgen05.ld.32x32b.x32` + `wait::ld`, `st.global.v4` |

**Dialect verifiers** (in `patina-opt`): `umma.*` only inside an `mma` role
region; `expect_tx` bytes == Σ bytes of `tma_load`s on that barrier; every
`tmem.alloc` reaches exactly one `dealloc`; barrier phases monotone per role;
producer/consumer arrival counts balanced.

### The v1 kernel in the dialect (golden file)

```mlir
patina.kernel @tcgen05_gemm_v1
    version = 0.1, arch = sm_100a, block = 192
    roles = [#patina.role<load, 1>, #patina.role<mma, 1>, #patina.role<epi, 4>]
  (%tma_a: !patina.tma_map<f16, #lay>, %tma_b: !patina.tma_map<f16, #lay>,
   %c: !patina.ptr<f32, gmem>) {

  %sa = patina.smem_alloc : !patina.tensor<f16, 128x32, smem, #lay>
  %sb = patina.smem_alloc : !patina.tensor<f16, 128x32, smem, #lay>
  %load_done = patina.mbarrier.create : !patina.mbarrier<1>
  %mma_done  = patina.mbarrier.create : !patina.mbarrier<1>
  %acc = patina.tmem.alloc cols = 128 : !patina.tmem<f32, 128>

  patina.warp_specialize {
    patina.role load {
      patina.expect_tx %load_done, 16384
      patina.tma_load %tma_a -> %sa, coords [0,0,0,0], complete %load_done
      patina.tma_load %tma_b -> %sb, coords [0,0,0,0], complete %load_done
    }
    patina.role mma {
      patina.wait %load_done, phase = 0
      %m = patina.umma.begin %acc, %sa, %sb
      patina.umma.step %m {accum = false}
      patina.umma.advance_k %m, 16
      patina.umma.step %m {accum = true}
      patina.umma.commit %m -> %mma_done
    }
    patina.role epi {
      patina.wait %mma_done, phase = 0
      %frag = patina.tmem_ld %acc : !patina.frag<f32, 128>
      patina.store %c, %frag
    }
  }
  patina.tmem.dealloc %acc
}
```

## 7. Reference mapping (Rust v1 → dialect → current .cu)

The Rust front-end expressing the same kernel (condensed):

```rust
kb.block(|blk| {
    let s_a = blk.smem::<f16, _, UmmaKMajor<NoSwizzle>>(shape![128, 32]);
    let s_b = blk.smem::<f16, _, UmmaKMajor<NoSwizzle>>(shape![128, 32]);
    let load_done = blk.mbarrier::<1>();
    let mma_done  = blk.mbarrier::<1>();
    let acc = blk.tmem_alloc::<f32, 128>()?;

    blk.warp_specialize::<GemmRoles>(|ws| {
        ws.role::<Load>(|t: LoadThread| {
            t.expect_tx(&load_done, s_a.bytes() + s_b.bytes());
            t.tma_load(&tma_a, &s_a, coord![0,0,0,0], &load_done);
            t.tma_load(&tma_b, &s_b, coord![0,0,0,0], &load_done);
        });
        ws.role::<Mma>(|t: MmaThread| {
            load_done.wait(phase(0));
            let mut m = t.umma_f16(&acc, &s_a, &s_b);
            m.step(Accum::No);  m.advance_k::<16>();  m.step(Accum::Yes);
            m.commit(&mma_done);
        });
        ws.role::<Epi>(|w: EpiWarp| {
            mma_done.wait(phase(0));
            let frag: Frag<f32, 128> = w.tmem_ld(&acc);
            w.store_v4(c + w.lane_row::<128>(), &frag);
        });
    });
});
```

Acceptance: the traced output is text-identical to `tests/golden/v1_gemm.mlir`,
and `patina-opt` verifies it. (Milestone 2 adds: lowering produces PTX
semantically equivalent to `tcgen05_gemm/tcgen05_gemm.sm_100a.ptx`.)

## 8. Host companion

TMA makes the DSL two-sided. Host API:

```rust
let tile = UmmaKMajor::<NoSwizzle>::new(shape![128, 32], f16::TY);
let tma_a = TmaMap::encode(&tile, a.ptr())?;   // 4D view derived; wraps
                                               // cuTensorMapEncodeTiled
module.launch(grid![1], block![192], &stream, (tma_a, tma_b, c.ptr()))?;
```

The same `layout` value drives host TMA encode and device descriptor packing —
one source of truth.

## 9. Verification strategy

1. **Rust trace-time**: type errors for role/scope/lifetime violations.
2. **Dialect**: `patina-opt --verify-diagnostics` on golden files (no Rust
   needed to test the dialect).
3. **Golden diff**: traced dialect vs checked-in `.mlir`.
4. **End-to-end** (milestone 2+): lowered PTX vs reference `.ptx`; SASS diff
   (`cuobjdump`) against `tcgen05_gemm.sm_100a.sass`; numerical correctness on
   hardware when available.

## 10. Milestones

- **M1 — Rust DSL → Patina dialect.** Layout algebra, trace IR, dialect
  printer; TableGen dialect + `patina-opt` verifiers; v1 warp-specialized
  single-tile GEMM as golden test. *No lowering passes.*
- **M2 — Dialect → PTX.** Lowering passes via NVGPU/NVVM (or direct PTX
  printer as fallback); parity with the reference kernel's PTX/SASS.
- **M3 — Real mainloop.** IR loop nodes with loop-carried stage/phase state;
  `Pipeline<STAGES>`; `setmaxnreg` role rebalancing; double-buffered TMEM
  accumulators (epilogue overlap).
- **M4 — Hardware breadth.** Swizzled layouts, `cta_group::2` + clusters
  (role model already anticipates leader-CTA roles), TMA multicast.
- **M5 — Tooling polish.** `melior` in-process MLIR bindings, error
  diagnostics mapped back to Rust source spans, kernel cache.

## 11. Open questions

1. **UMMA op shape**: `begin/step/advance_k/commit` chain vs one flat
   `patina.umma.mma` op per K16 step with an explicit descriptor-carried
   value. Sets the dialect's abstraction altitude — decide first in M1.
2. **Phase modeling across loops** (M3): phase attributes suffice for M1's
   single-use barriers; ring pipelines may need phase *values*.
3. **Role→warp assignment**: auto-pack (v1 choice, printed in IR dump) vs
   user-pinned; interacts with `setmaxnreg` warpgroup granularity in M3.
4. **crates.io name**: `patina` appears unclaimed (404 on the registry API);
   reserve early if the project goes public.

## 12. Naming

**Patina** — *a thin typed layer over tcgen05.* Dialect namespace `patina.`,
tool `patina-opt`, crate `patina`. (Runner-up from the search: **Karat** —
GEMM/K-dimension pun, also unclaimed.)
