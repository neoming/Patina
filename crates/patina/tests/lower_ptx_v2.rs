//! End-to-end lowering test (M3): trace the v2 pipelined-mainloop kernel in
//! the DSL, emit the dialect, lower through the C ABI, and compare against
//! the golden PTX (tests/golden/v2_gemm_pipelined.ptx). The golden is also
//! validated externally with `ptxas -arch=sm_100a`.
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

fn tcgen05_gemm_v2(
    kb: &mut KernelCtx,
    tma_a: TmaMap<f16, UmmaKMajor<NoSwizzle>>,
    tma_b: TmaMap<f16, UmmaKMajor<NoSwizzle>>,
    c: Ptr<f32, Gmem>,
    k_tiles: Val<u32>,
) {
    kb.block(|blk| {
        let pipe = blk.pipeline::<4>();
        let s_a = pipe.buf::<f16, UmmaKMajor<NoSwizzle>, _>(shape![128, 32]);
        let s_b = pipe.buf::<f16, UmmaKMajor<NoSwizzle>, _>(shape![128, 32]);

        let acc_done = blk.mbarrier_named::<1>("acc_done");
        let acc = blk.tmem_alloc::<f32, 128>()?;

        blk.warp_specialize::<GemmRoles>(|ws| {
            ws.role::<Load>(|t: LoadThread| {
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
                t.for_each(&k_tiles, |i, st| {
                    t.consumer_acquire(&pipe, &st);
                    let sa_s = pipe.stage_buf(&s_a, &st);
                    let sb_s = pipe.stage_buf(&s_b, &st);
                    let cont = i.nez();
                    let mut m = t.umma_f16(&acc, &sa_s, &sb_s);
                    m.step_dyn(&cont);
                    m.advance_k::<16>();
                    m.step(Accum::Yes);
                    m.commit_stage(&pipe, &st);
                    st.next(&pipe)
                });
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

fn trace_v2() -> String {
    Kernel::trace(tcgen05_gemm_v2, Arch::Sm100a)
        .roles::<GemmRoles>()
        .block(192)
        .emit_dialect()
}

#[test]
fn v2_gemm_lowers_to_golden_ptx() {
    let ptx = ffi::lower_to_ptx(&trace_v2()).expect("v2 lowering must succeed");
    patina::testing::expect_golden(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/v2_gemm_pipelined.ptx"),
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
fn lowered_v2_ptx_instruction_inventory() {
    let ptx = ffi::lower_to_ptx(&trace_v2()).unwrap();
    assert!(ptx.contains(".target sm_100a"));
    assert!(ptx.contains(".entry tcgen05_gemm_v2"));
    let code = code_only(&ptx);

    // Dynamic shared memory: 4 stages x 2 tiles x 8 KiB exceeds the 48 KiB
    // static .shared limit, so the ring lives in dyn_smem.
    assert!(code.contains(".extern .shared .b8 dyn_smem[];"));
    assert!(code.contains(".shared .align 8 .b64 full_bar[4];"));
    assert!(code.contains(".shared .align 8 .b64 empty_bar[4];"));

    // Runtime loop structure: two LOOP regions with condition + backedge.
    assert!(code.contains("LOOP_0:"), "load-role loop label");
    assert!(code.contains("LOOP_1:"), "mma-role loop label");
    assert_eq!(count(&code, "setp.ge.u32 %p1, %i, %kt;"), 2, "loop conditions");
    assert!(code.contains("bra LOOP_0;"), "load backedge");
    assert!(code.contains("bra LOOP_1;"), "mma backedge");
    assert!(code.contains("ld.param.u32 %kt, [k_tiles];"));

    // Stage-indexed addressing: barriers (8 B stride) and smem (8192 B).
    assert_eq!(count(&code, "shl.b32 %tmp, %s, 3;"), 6, "stage barrier addrs");
    assert_eq!(count(&code, "shl.b32 %tmp, %s, 13;"), 4, "stage smem addrs");
    assert!(code.contains("add.s32 %smem_addr, %smem_addr, 32768;"), "sb ring offset");

    // Phase as a register operand, flipped on wrap.
    assert_eq!(count(&code, "[%bar_s], %ph;"), 2, "runtime-phase waits");
    assert_eq!(count(&code, "setp.eq.u32 %p3, %s, 4;"), 2, "wrap checks");
    assert_eq!(count(&code, "@%p3 xor.b32 %ph, %ph, 1;"), 2, "phase flips");

    // Producer starts at phase 1, consumer at 0.
    assert!(code.contains("mov.b32 %ph, 1;"));
    assert!(code.contains("mov.b32 %ph, 0;"));

    // Dynamic scaleC: nez(i) feeds the first K16 step of each tile.
    assert!(code.contains("setp.ne.u32 %p4, %i, 0;"));
    assert!(code.contains("selp.b32 %cont, 1, 0, %p4;"));
    assert!(code.contains("mov.b32 %scale_c, %cont;"));
    assert!(code.contains("mov.b32 %scale_c, 1;"), "second step accumulates");
    assert!(!code.contains("mov.b32 %scale_c, 0;"), "no static first step");

    // Runtime TMA coordinate (4 * i) in both copies.
    assert!(code.contains("shl.b32 %i4, %i, 2;"));
    assert_eq!(
        count(&code, "{%zero, %zero, %i4, %zero}"),
        2,
        "runtime K coord in both TMA copies"
    );

    // Instruction counts.
    assert_eq!(count(&code, "cp.async.bulk.tensor.4d"), 2, "two TMA copies");
    assert_eq!(count(&code, "mbarrier.arrive.expect_tx"), 1);
    assert_eq!(count(&code, "tcgen05.mma"), 2, "two K16 UMMA steps");
    assert_eq!(count(&code, "tcgen05.commit"), 2, "staged commit + drain");
    assert_eq!(count(&code, "tcgen05.ld.sync.aligned.32x32b.x32.b32"), 4);
    assert_eq!(count(&code, "mbarrier.init"), 9, "full[4] + empty[4] + acc_done");
    assert_eq!(count(&code, "mbarrier.try_wait.parity"), 3, "load/mma/epi waits");
    assert_eq!(count(&code, "tcgen05.alloc"), 1);
    assert_eq!(count(&code, "tcgen05.dealloc"), 1);
    assert_eq!(count(&code, "bar.sync"), 2);
    assert_eq!(count(&code, "st.global.v4.f32"), 32, "128 f32 per thread");

    // K advance between the two steps: +16 descriptor units on both A and B.
    assert_eq!(count(&code, "add.s64 %da, %da, 16;"), 1);
    assert_eq!(count(&code, "add.s64 %db, %db, 16;"), 1);
}

/// A pipeline kernel whose `patina.for` uses renamed index/iter_args (not the
/// front-end's %i/%s/%ph). Regression: the lowering once initialized the
/// parsed names but ran the loop skeleton on hardcoded %i/%s/%ph registers,
/// emitting undeclared-register PTX while exiting 0.
#[test]
fn renamed_loop_vars_lower_to_consistent_ptx() {
    let mlir = r#"
#lay = #patina.umma_k_major<swizzle_none>

patina.kernel @renamed_loop
    version = 0.1, arch = #patina.arch<sm_100a>, block = 96,
    roles = [#patina.role<load, 1>, #patina.role<mma, 1>, #patina.role<epi, 1>]
  (%tma_a: !patina.tma_map<f16, #lay>,
   %c: !patina.ptr<f32, gmem>,
   %k_tiles: !patina.u32) {
  %pipe = patina.pipeline.create stages = 4 : !patina.pipeline<4>
  %sa = patina.smem_alloc stages = 4 : !patina.tensor<f16, 128x32, smem, #lay>
  %acc_done = patina.mbarrier.create arrivals = 1 : !patina.mbarrier<1>
  %acc = patina.tmem.alloc cols = 128 : !patina.tmem<f32, 128>
  patina.warp_specialize {
    patina.role load {
      patina.for %j = 0 to %k_tiles iter_args(%x = 0, %p = 1) {
        patina.wait_stage %pipe, empty, %x, %p
        patina.expect_tx_stage %pipe, full, %x, 8192
        %sa_s = patina.stage_buf %sa, %x : !patina.tensor<f16, 128x32, smem, #lay>
        patina.tma_load %tma_a -> %sa_s, coords [0, 0, 0, 0], complete %pipe, full, %x
        %x1, %p1 = patina.stage_next %x, %p of %pipe
        patina.yield %x1, %p1
      }
    }
    patina.role mma {
      patina.for %j = 0 to %k_tiles iter_args(%x = 0, %p = 0) {
        patina.wait_stage %pipe, full, %x, %p
        %sa_s = patina.stage_buf %sa, %x : !patina.tensor<f16, 128x32, smem, #lay>
        %m = patina.umma.begin %acc, %sa_s, %sa_s
        patina.umma.step %m {accum = true}
        patina.umma.commit %m -> %pipe, empty, %x
        %x1, %p1 = patina.stage_next %x, %p of %pipe
        patina.yield %x1, %p1
      }
      patina.umma.commit %m -> %acc_done
    }
    patina.role epi {
      patina.wait %acc_done, phase = 0
      %frag = patina.tmem_ld %acc : !patina.frag<f32, 128>
      patina.store %c, %frag
    }
  }
  patina.tmem.dealloc %acc
}
"#;
    let ptx = ffi::lower_to_ptx(mlir).expect("renamed-loop kernel must lower");
    // The renamed names must be declared and used consistently; the
    // hardcoded %i/%s/%ph skeleton must be gone from the loop bodies.
    assert!(ptx.contains(".reg .b32 %j, %x, %p;"), "missing decl:\n{ptx}");
    assert!(ptx.contains("setp.ge.u32 %p1, %j, %kt;"), "loop cond:\n{ptx}");
    assert!(
        ptx.contains("mbarrier.try_wait.parity.shared::cta.b64 P, [%bar_s], %p;"),
        "runtime-phase wait:\n{ptx}"
    );
    assert!(ptx.contains("shl.b32 %tmp, %x, 3;"), "stage addressing:\n{ptx}");
    assert!(ptx.contains("setp.eq.u32 %p3, %x, 4;"), "ring wrap check:\n{ptx}");
    assert!(
        !ptx.contains("setp.ge.u32 %p1, %i, %kt;"),
        "loop condition must not use the hardcoded %i:\n{ptx}"
    );
}
