//! Golden test: the Rust DSL trace of the v1 warp-specialized GEMM must
//! produce tests/golden/v1_gemm.mlir verbatim.
//!
//! STATUS: TDD specification — this file defines the intended public API of
//! the `patina` crate. It does not compile until M1a lands; that is the point.
//! Run (then):  cargo test --test golden_v1_gemm
//! Update golden after intentional dialect changes:  BLESS=1 cargo test ...

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
fn tcgen05_gemm_v1(
    kb: &mut KernelCtx,
    tma_a: TmaMap<f16, UmmaKMajor<NoSwizzle>>,
    tma_b: TmaMap<f16, UmmaKMajor<NoSwizzle>>,
    c: Ptr<f32, Gmem>,
) {
    kb.block(|blk| {
        let s_a = blk.smem::<f16, _, UmmaKMajor<NoSwizzle>>(shape![128, 32]);
        let s_b = blk.smem::<f16, _, UmmaKMajor<NoSwizzle>>(shape![128, 32]);

        let load_done = blk.mbarrier::<1>();
        let mma_done  = blk.mbarrier::<1>();
        let acc = blk.tmem_alloc::<f32, 128>()?;

        blk.warp_specialize::<GemmRoles>(|ws| {
            ws.role::<Load>(|t: LoadThread| {
                t.expect_tx(&load_done, s_a.bytes() + s_b.bytes());
                t.tma_load(&tma_a, &s_a, coord![0, 0, 0, 0], &load_done);
                t.tma_load(&tma_b, &s_b, coord![0, 0, 0, 0], &load_done);
            });

            ws.role::<Mma>(|t: MmaThread| {
                load_done.wait(phase(0));
                let mut m = t.umma_f16(&acc, &s_a, &s_b);
                for k in 0..2 {
                    // trace-time loop: unrolls into the two umma.step ops;
                    // K advances only BETWEEN steps
                    if k > 0 { m.advance_k::<16>(); }
                    m.step(if k == 0 { Accum::No } else { Accum::Yes });
                }
                m.commit(&mma_done);
            });

            ws.role::<Epi>(|w: EpiWarp| {
                mma_done.wait(phase(0));
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
fn v1_gemm_matches_golden_dialect() {
    let mlir = Kernel::trace(tcgen05_gemm_v1, Arch::Sm100a)
        .roles::<GemmRoles>()
        .block(192)
        .emit_dialect(); // -> String of Patina-dialect MLIR

    patina::testing::expect_golden(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/v1_gemm.mlir"),
        &mlir,
    ); // on mismatch: diff + fail; BLESS=1 rewrites the golden file
}

// ---------------------------------------------------------------------------
// Compile-fail expectations (documented here, enforced via trybuild later):
//   * `t.umma_f16(...)` inside a Load role        -> no method on LoadThread
//   * `w.tma_load(...)`  inside an Epi role       -> no method on EpiWarp
//   * dropping `acc` while an Epi closure holds &acc -> borrow error
//   * block(128) with GemmRoles (needs 192)       -> trace-time panic
// ---------------------------------------------------------------------------
