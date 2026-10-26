//! Golden test: the Rust DSL trace of the v2 pipelined-mainloop GEMM must
//! produce tests/golden/v2_gemm_pipelined.mlir verbatim.
//!
//! C[128,128] f32 += A[128,K] * B[128,K]^T with K = k_tiles * 32, a 4-stage
//! smem ring pipeline, roles Load(1)/Mma(1)/Epi(4) as v1.
//!
//! STATUS: TDD specification — this file defines the M3 public API of the
//! `patina` crate (pipelines, runtime loops, runtime phases). It does not
//! compile until M3-rust lands; that is the point.
//! Run (then):  cargo test --test golden_v2_gemm_pipelined
//! Update golden after intentional dialect changes:  BLESS=1 cargo test ...
//!
//! Dialect contract exercised here (see the golden file's header comment):
//!   * `Pipeline<S>` owns full[S] + empty[S] barrier rings; phases are
//!     runtime values carried alongside the stage index (`PipeStage`).
//!   * The producer waits empty[s] with initial phase 1, publishes full[s];
//!     the consumer waits full[s] with initial phase 0, releases empty[s];
//!     the phase flips when the stage wraps mod S (`st.next(&pipe)`).
//!   * `t.for_each(&k_tiles, |i, st| ...)` is a RUNTIME loop (zero-based):
//!     it emits one `patina.for` with loop-carried (stage, phase) — unlike a
//!     plain Rust `for`, which unrolls at trace time.

use patina::prelude::*;

// ---------------------------------------------------------------------------
// Role set: Load = 1 warp, Mma = 1 warp, Epi = 4 warps (total 6 warps = 192)
// ---------------------------------------------------------------------------
struct GemmRoles;
impl Roles for GemmRoles {
    type Load = Role<1>;
    type Mma  = Role<1>;
    type Epi  = Role<4>;
}

// ---------------------------------------------------------------------------
// The kernel, exactly as it should read in the DSL
// ---------------------------------------------------------------------------
fn tcgen05_gemm_v2(
    kb: &mut KernelCtx,
    tma_a: TmaMap<f16, UmmaKMajor<NoSwizzle>>,
    tma_b: TmaMap<f16, UmmaKMajor<NoSwizzle>>,
    c: Ptr<f32, Gmem>,
    k_tiles: Val<u32>,
) {
    kb.block(|blk| {
        // 4-stage ring: full[4] + empty[4] barriers, owned by the pipeline.
        let pipe = blk.pipeline::<4>();
        // Staged smem: S contiguous tiles each; stage slices come from
        // pipe.stage_buf(...) inside the loop.
        let s_a = pipe.buf::<f16, UmmaKMajor<NoSwizzle>>(shape![128, 32]);
        let s_b = pipe.buf::<f16, UmmaKMajor<NoSwizzle>>(shape![128, 32]);

        let acc_done = blk.mbarrier::<1>();
        let acc = blk.tmem_alloc::<f32, 128>()?;

        blk.warp_specialize::<GemmRoles>(|ws| {
            ws.role::<Load>(|t: LoadThread| {
                // producer: wait empty[s] (initial phase 1), publish full[s]
                t.for_each(&k_tiles, |i, st| {
                    t.producer_acquire(&pipe, &st);
                    t.publish(&pipe, &st, s_a.bytes() + s_b.bytes());
                    let sa_s = pipe.stage_buf(&s_a, &st);
                    let sb_s = pipe.stage_buf(&s_b, &st);
                    let i4 = i.muli::<4>();
                    t.tma_load_stage(&tma_a, &sa_s, coord![0, 0, i4, 0], &pipe, &st);
                    t.tma_load_stage(&tma_b, &sb_s, coord![0, 0, i4, 0], &pipe, &st);
                    st.next(&pipe)
                });
            });

            ws.role::<Mma>(|t: MmaThread| {
                // consumer: wait full[s] (initial phase 0), release empty[s]
                t.for_each(&k_tiles, |i, st| {
                    t.consumer_acquire(&pipe, &st);
                    let sa_s = pipe.stage_buf(&s_a, &st);
                    let sb_s = pipe.stage_buf(&s_b, &st);
                    let cont = i.nez();
                    let mut m = t.umma_f16(&acc, &sa_s, &sb_s);
                    m.step_dyn(&cont); // scaleC = (i != 0): C += on every tile but the first
                    m.advance_k::<16>();
                    m.step(Accum::Yes);
                    m.commit_stage(&pipe, &st);
                    st.next(&pipe)
                });
                // drain: one more commit so the epilogue can wait the MMA group
                t.commit_tail(&acc_done);
            });

            ws.role::<Epi>(|w: EpiWarp| {
                acc_done.wait(phase(0));
                let frag: Frag<f32, 128> = w.tmem_ld(&acc);
                w.store_v4(c + w.lane_row::<128>(), &frag);
            });
        });
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// Golden comparison
// ---------------------------------------------------------------------------
#[test]
fn v2_gemm_matches_golden_dialect() {
    let mlir = Kernel::trace(tcgen05_gemm_v2, Arch::Sm100a)
        .roles::<GemmRoles>()
        .block(192)
        .emit_dialect(); // -> String of Patina-dialect MLIR

    patina::testing::expect_golden(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/v2_gemm_pipelined.mlir"),
        &mlir,
    ); // on mismatch: diff + fail; BLESS=1 rewrites the golden file
}

// ---------------------------------------------------------------------------
// Compile-fail expectations (documented here, enforced via trybuild later):
//   * `st.next(&pipe)` not returned from the for_each closure -> type error
//   * `t.producer_acquire` inside the Mma role     -> no method on MmaThread
//   * `pipe.stage_buf` on an unstaged smem tensor  -> no method / type error
//   * `m.step_dyn(...)` on a Load/Epi token        -> no umma capability
// ---------------------------------------------------------------------------
