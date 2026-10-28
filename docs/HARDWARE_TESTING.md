# Running Patina on a GB200 (Blackwell / sm_100a)

Patina's tcgen05 GEMM kernels cannot execute on pre-Blackwell GPUs (the
reference kernel and both golden PTX files are `sm_100a`-only). This runbook
covers the one thing that only a GB200 can do: **execute the generated PTX and
check it numerically**.

Everything else — Rust front-end, dialect verification, PTX lowering, and
`ptxas` assembly — runs offline via `./scripts/preflight.sh`.

## 0. Before you go (on any machine)

```bash
./scripts/preflight.sh
```

This must print `PREFLIGHT PASSED`. It runs `cargo test`, the `patina-opt`
verifier contract, the CLI lowering, `ptxas -arch=sm_100a` on both golden PTX
files, compiles the hardware harness, and runs a local driver-API smoke test
(`tests/run_smoke.rs`, a trivial non-tcgen05 kernel) to prove the
load→launch→copy→check plumbing works on whatever GPU is present. If it
fails, fix it *before* you touch the GB200.

## 1. GB200 server prerequisites

| Requirement | Minimum | Why |
|---|---|---|
| GPU | GB200 / B200 (sm_100a) | tcgen05 is Blackwell-only |
| Driver | >= 570 (Blackwell branch) | JITs PTX 8.7 → tcgen05 SASS |
| `cuda.h` + `libcuda` | any CUDA 12.x toolkit | the harness is host-only driver API |
| g++ | g++-13 | harness build (plain C++17) |

Check the prerequisites on the server:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
#   expect: compute_cap = 10.0  and  driver_version >= 570.xx
```

> The harness is compiled with the **driver API only** (`cuda.h` + `libcuda`),
> not nvcc, so it does not need the full CUDA toolkit or a matching gcc/nvcc
> pair. The PTX was already assembled offline by `ptxas 12.8`; the driver JITs
> it at `cuModuleLoadDataEx`.
>
> **No NVRTC is involved.** Patina's `LowerToPTX` pass prints PTX *source
> text* directly (it does not go through LLVM/NVPTX, and it does not emit CUDA
> C++). NVRTC compiles CUDA C++ source; there is nothing for it to do here.

## 2. What to copy to the server

The harness needs only these (no LLVM/MLIR toolchain, no cargo, no Rust):

```
harness/gemm_check.cpp
harness/build.sh
crates/patina/tests/golden/v1_gemm.ptx
crates/patina/tests/golden/v2_gemm_pipelined.ptx
```

Or, simplest: clone the repo and run `./scripts/preflight.sh` there to confirm
the environment, then build the harness.

## 3. Build the harness on the server

```bash
./harness/build.sh
#   (override toolchain if needed:  CXX=g++-13 CUDA_HOME=/usr/local/cuda ./harness/build.sh)
```

## 4. Run the numerical checks

The harness loads PTX with the driver API, fills `A[128,K]`, `B[128,K]` (f16)
deterministically, builds the TMA tensor maps, launches one CTA
(`block = 192`), and checks `C = A·Bᵀ` against a double-precision host
reference.

```bash
# v1 single-tile kernel (K = 32)
./harness/gemm_check crates/patina/tests/golden/v1_gemm.ptx tcgen05_gemm_v1 32

# v2 pipelined kernel, one K-tile  (must match v1 bit-for-bit)
./harness/gemm_check crates/patina/tests/golden/v2_gemm_pipelined.ptx tcgen05_gemm_v2 32

# v2 pipelined kernel, four K-tiles (exercises the stage-wrap phase flip
# and the dynamic scaleC on the first K16 of each tile)
./harness/gemm_check crates/patina/tests/golden/v2_gemm_pipelined.ptx tcgen05_gemm_v2 128
```

Expected output for a passing run:

```
device: NVIDIA GB200 ... (sm_100a)
kernel: tcgen05_gemm_v2  K=128  k_tiles=4  block=192  dyn_smem=65536
max abs err: 1.23e-06   max rel err: 2.34e-07
mismatches:  0 / 16384
sum(C):      ...
PASS
```

The exit code is `0` on PASS, `1` on a wrong result, `2` on a usage/launch
error, and `3` if the device is not `sm_100a`.

## 5. Alternative: run the whole loop from Rust (`cargo test`)

Patina can also drive the CUDA driver API directly from Rust — no C++ and no
nvcc — through `patina::run` (behind `feature = "cuda"`). It declares the
`cuda.h` C ABI via `extern "C"` and links `libcuda.so`, so the full loop
(trace DSL → emit dialect → lower to PTX → run → check) is one command:

```bash
cargo test --features cuda --test hw_run -- --nocapture
```

This does the same three runs as section 4, but traces/lowers the PTX on the
spot instead of using the checked-in golden files. It needs the full Rust +
LLVM/MLIR toolchain on the server (the crate's `build.rs` compiles the `mlir/`
subproject), so it is the heavier option; the C++ harness in section 4 is the
minimal-toolchain fallback. On a non-Blackwell GPU it prints `skip:` and
passes vacuously.

## 6. Interpreting the results

- **All three runs PASS** — the DSL→dialect→PTX pipeline is numerically
  correct end-to-end. `v2 K=32` matching `v1 K=32` also cross-validates the
  pipelined lowering against the single-tile lowering.
- **`v1` passes, `v2 K=128` fails** — the multi-tile pipeline (stage-wrap
  phase flip, staged TMA coordinates, dynamic scaleC) is where to look.
- **`v1` fails** — the shared machinery (TMA tensor map, descriptors, TMEM
  epilogue) is wrong; the `max abs/rel err` magnitude tells you whether it is
  a gross layout error (O(K)) or a small numerical issue.
- **Hang / timeout** — barrier choreography is the prime suspect
  (`expect_tx` byte counts vs. `tma_load` bytes, `wait` phase parity, or the
  producer/consumer phase convention). Run under `compute-sanitizer`:

  ```bash
  compute-sanitizer --tool memcheck ./harness/gemm_check .../v2_gemm_pipelined.ptx tcgen05_gemm_v2 32
  ```

## 7. Cross-check against the reference kernel (optional)

The C++ reference `~/WorkSpace/tcgen05_gemm/tcgen05_gemm.cu` is the same GEMM
written by hand. If you want to confirm Patina's PTX against it, build the
reference's cubin on the server (`~/WorkSpace/tcgen05_gemm/build.sh`) and diff
the instruction inventories — or adapt its `tcgen05_gemm_launch` host stub to
run both on identical inputs. The harness above already checks against a CPU
reference, which is the stronger correctness statement.

## 8. Static equivalence verification (done offline, already green)

Before spending GB200 time, the generated PTX was diffed instruction-by-
instruction against two independent tcgen05 references, and it matches:

* **vs. `~/WorkSpace/tcgen05_gemm/tcgen05_gemm.sm_100a.ptx`** (hand-written
  reference): identical instruction descriptors and constants —
  `idesc = 136314896`, smem descriptor `(addr>>4) | 70437464178688`
  (`LBO=128>>4`, `SBO=256>>4`, `version=1<<46`), K-advance `+16` descriptor
  units (= 256 B = 2 core matrices), `tcgen05.mma.cta_group::1.kind::f16`
  with the `setp.ne` scaleC predicate, TMA `cp.async.bulk.tensor.4d` with
  `expect_tx 16384`, `tcgen05.ld.32x32b.x32.b32`, TMEM lane base
  `warp<<21`, and the full alloc/relinquish/commit/dealloc/fence sequence.
* **vs. CUTLASS `include/cute/`** (sm100 UMMA): the `SmemDescriptor` and
  `InstrDescriptor` bit layouts, the `make_umma_desc` K-major SWIZZLE_NONE
  fields (`LBO=128`, `SBO=256`, `version=1`), and `DP_b = 1<<21` for TMEM
  lanes all match Patina's encodings. The v2 pipeline phase convention
  (producer phase 1 / consumer phase 0) matches `make_producer_start_state`,
  and the dynamic scaleC (`i != 0` overwrite/accumulate) matches CUTLASS's
  `ScaleOut::Zero` → `ScaleOut::One`.

Only structural differences are intentional: Patina uses 6 warps
(load/mma/epi×4, block 192) vs. the reference's 4, and two barriers
(`load_done`/`mma_done`) vs. the reference's single two-phase barrier.
