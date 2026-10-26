//! End-to-end lowering test: trace the v1 kernel in the DSL, emit the
//! dialect, lower through the C ABI, and compare against the golden PTX
//! (tests/golden/v1_gemm.ptx). The golden is also validated externally with
//! `ptxas -arch=sm_100a`.
//!
//! Update the golden after intentional lowering changes: BLESS=1 cargo test.

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
fn v1_gemm_lowers_to_golden_ptx() {
    let ptx = ffi::lower_to_ptx(&trace_v1()).expect("v1 lowering must succeed");
    patina::testing::expect_golden(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/v1_gemm.ptx"),
        &ptx,
    );
}

/// Strip `//` comments so inventory counts see only instructions.
fn code_only(ptx: &str) -> String {
    ptx.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

#[test]
fn lowered_ptx_instruction_inventory() {
    let ptx = ffi::lower_to_ptx(&trace_v1()).unwrap();
    assert!(ptx.contains(".target sm_100a"));
    assert!(ptx.contains(".entry tcgen05_gemm_v1"));
    let code = code_only(&ptx);

    assert_eq!(count(&code, "cp.async.bulk.tensor.4d"), 2, "two TMA copies");
    assert_eq!(count(&code, "mbarrier.arrive.expect_tx"), 1);
    assert_eq!(count(&code, "tcgen05.mma"), 2, "two K16 UMMA steps");
    assert_eq!(count(&code, "tcgen05.commit"), 1);
    assert_eq!(count(&code, "tcgen05.ld.sync.aligned.32x32b.x32.b32"), 4);
    assert_eq!(count(&code, "tcgen05.wait::ld"), 4);
    assert_eq!(count(&code, "tcgen05.alloc"), 1);
    assert_eq!(count(&code, "tcgen05.relinquish_alloc_permit"), 1);
    assert_eq!(count(&code, "tcgen05.dealloc"), 1);
    assert_eq!(count(&code, "tcgen05.fence::before_thread_sync"), 1);
    assert_eq!(count(&code, "tcgen05.fence::after_thread_sync"), 1);
    assert_eq!(count(&code, "mbarrier.try_wait.parity"), 2);
    assert_eq!(count(&code, "mbarrier.init"), 2);
    assert_eq!(count(&code, "bar.sync"), 2);
    assert_eq!(count(&code, "st.global.v4.f32"), 32, "128 f32 per thread");

    // scaleC 0 -> 1: first K16 step zeroes the accumulator input.
    let first = code.find("mov.b32 %scale_c, 0;").expect("scale_c = 0 step");
    let second = code.find("mov.b32 %scale_c, 1;").expect("scale_c = 1 step");
    assert!(first < second, "scaleC must step 0 then 1");
    assert_eq!(count(&code, "setp.ne.b32 SC, %scale_c, 0"), 2);

    // K advance between the two steps: +16 descriptor units on both A and B.
    assert_eq!(count(&code, "add.s64 %da, %da, 16;"), 1);
    assert_eq!(count(&code, "add.s64 %db, %db, 16;"), 1);
}

#[test]
fn non_kernel_module_is_an_error() {
    let err = ffi::lower_to_ptx("module { }").expect_err("no kernel to lower");
    assert!(err.contains("no patina.kernel"), "unexpected: {err}");
}
