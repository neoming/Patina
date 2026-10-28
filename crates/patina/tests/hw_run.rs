//! Hardware correctness test: trace → emit dialect → lower to PTX → **run on
//! the GPU** → compare against a host reference. This is the in-Rust
//! counterpart of `harness/gemm_check.cpp`; it only compiles/links with
//! `--features cuda` and only runs on a Blackwell (sm_100a) device.
//!
//! ```bash
//! # on a GB200:
//! cargo test --features cuda --test hw_run -- --nocapture
//! ```
//!
//! On a non-Blackwell GPU (or a machine with no CUDA driver) each case prints
//! `skip: ...` and passes vacuously, so the suite stays green everywhere else.

#![cfg(feature = "cuda")]

use patina::ffi;
use patina::prelude::*;

struct GemmRoles;
impl Roles for GemmRoles {
    type Load = Role<1>;
    type Mma = Role<1>;
    type Epi = Role<4>;
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
        let mma_done = blk.mbarrier::<1>();
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
                    if k > 0 {
                        m.advance_k::<16>();
                    }
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

fn trace_v1() -> String {
    Kernel::trace(tcgen05_gemm_v1, Arch::Sm100a)
        .roles::<GemmRoles>()
        .block(192)
        .emit_dialect()
}

fn trace_v2() -> String {
    Kernel::trace(tcgen05_gemm_v2, Arch::Sm100a)
        .roles::<GemmRoles>()
        .block(192)
        .emit_dialect()
}

// Data-path helpers live in their own module WITHOUT the prelude glob import,
// because `patina::prelude::*` shadows the primitive `f32`/`f16` with the
// DSL's dtype markers. Here `f32`/`f64` are the Rust primitives.
mod data {
    /// Deterministic inputs matching `harness/gemm_check.cpp::fill_inputs`.
    pub fn inputs(k: u64) -> (Vec<f32>, Vec<f32>) {
        let n = (128 * k) as usize;
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        for m in 0..128u32 {
            for kk in 0..k {
                let idx = m as usize * k as usize + kk as usize;
                let (mf, kf) = (m as f32, kk as f32);
                a[idx] = 0.9 * (0.1 * (mf + 1.0) + 0.7 * kf).sin() * (0.05 * (kf + 1.0)).cos();
                b[idx] = 0.8 * (0.13 * (mf + 1.0) + 0.11 * kf).cos()
                    * (0.03 * (mf + kf + 1.0)).sin();
            }
        }
        (a, b)
    }

    pub fn run_and_check(name: &str, k: u64, ptx: &str) -> bool {
        let (a, b) = inputs(k);
        let c = match patina::run::run_gemm(ptx, name, k, &a, &b) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{name} K={k}: run failed: {e}");
                return false;
            }
        };
        let reference = patina::run::reference_gemm(&a, &b, k);
        let (max_abs, max_rel, mismatches) = patina::run::compare(&c, &reference);
        println!("{name} K={k}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} mismatches={mismatches}");
        mismatches == 0
    }
}

#[test]
fn v1_and_v2_are_numerically_correct_on_hardware() {
    match patina::run::compute_capability() {
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
        Ok((maj, min)) if maj < 10 => {
            eprintln!("skip: sm_{maj}{min} (tcgen05 needs sm_100a)");
            return;
        }
        Ok((maj, min)) => println!("device: sm_{maj}{min}"),
    }

    let ptx1 = ffi::lower_to_ptx(&trace_v1()).expect("v1 lowering");
    let ptx2 = ffi::lower_to_ptx(&trace_v2()).expect("v2 lowering");

    assert!(data::run_and_check("tcgen05_gemm_v1", 32, &ptx1));
    // v2 with one K-tile must match v1 (cross-validates the pipelined
    // lowering against the single-tile lowering).
    assert!(data::run_and_check("tcgen05_gemm_v2", 32, &ptx2));
    // Four K-tiles exercise the stage-wrap phase flip and the dynamic scaleC.
    assert!(data::run_and_check("tcgen05_gemm_v2", 128, &ptx2));
}
