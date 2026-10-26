//! FFI test: traced Rust DSL output must verify clean through the C ABI,
//! and broken dialect text must come back as an `Err` with the diagnostic.

use patina::ffi;
use patina::prelude::*;

struct GemmRoles;
impl Roles for GemmRoles {
    type Load = Role<1>;
    type Mma  = Role<1>;
    type Epi  = Role<4>;
}

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

fn trace_v1() -> String {
    Kernel::trace(tcgen05_gemm_v1, Arch::Sm100a)
        .roles::<GemmRoles>()
        .block(192)
        .emit_dialect()
}

#[test]
fn traced_v1_kernel_verifies_via_ffi() {
    let mlir = trace_v1();
    ffi::verify(&mlir).expect("traced v1 kernel must verify clean via FFI");
}

#[test]
fn invalid_dialect_returns_diagnostic() {
    // Wrong expect_tx byte count (half of the real 16384).
    let mlir = trace_v1().replace("expect_tx %load_done, 16384", "expect_tx %load_done, 8192");
    assert!(mlir.contains("expect_tx %load_done, 8192"));
    let err = ffi::verify(&mlir).expect_err("wrong expect_tx bytes must fail verification");
    assert!(
        err.contains("expect_tx bytes (8192) != sum of tma_load bytes"),
        "unexpected diagnostic: {err}"
    );
}

#[test]
fn unparseable_text_returns_diagnostic() {
    let err = ffi::verify("this is not mlir at all").expect_err("garbage must fail to parse");
    assert!(!err.is_empty(), "diagnostic must not be empty");
}

#[test]
fn lower_to_ptx_stub_reports_unimplemented() {
    // M2b: the C ABI exposes lower_to_ptx as a stub; M2c implements it.
    let err = ffi::lower_to_ptx(&trace_v1()).expect_err("stub must return Err");
    assert!(err.contains("unimplemented"), "unexpected diagnostic: {err}");
}
