// tests/golden/v2_gemm_pipelined.mlir
//
// Golden dialect output for the v2 pipelined-mainloop GEMM:
// C[128,128] f32 += A[128, K] * B[128, K]^T with K = k_tiles * 32, a 4-stage
// smem ring pipeline, single CTA, roles Load(1)/Mma(1)/Epi(4) as v1.
//
// New surface vs v1 (M3):
//   !patina.u32                       scalar runtime values (k_tiles, i, s, ph)
//   !patina.pipeline<S>               S-stage ring: full[S] + empty[S] barriers
//   patina.pipeline.create            CTA-scope pipeline creation
//   patina.smem_alloc stages = S      staged buffer (S contiguous tiles)
//   patina.stage_buf %buf, %s         the stage-s slice of a staged buffer
//   patina.for / patina.yield         runtime loop, loop-carried (s, ph)
//   patina.wait_stage                 phase as a VALUE (ring pipelines make
//                                     type-level phases impossible)
//   patina.expect_tx_stage            expect_tx on full[s]
//   patina.stage_next                 s+1 mod S, phase flip on wrap
//   patina.muli / patina.nez          scalar arithmetic for coords / scaleC
//   patina.umma.step ..., accum %c    dynamic scaleC (i != 0)
//   tma_load / umma.commit            staged barrier form: (pipe, kind, s)
//
// Phase convention (CUTLASS): the producer waits empty[s] with initial phase
// 1 (passes immediately on fresh barriers), the consumer waits full[s] with
// initial phase 0; the phase flips when the stage wraps mod S.
//
// RUN: patina-opt %s > /dev/null   // must verify clean
// The Rust side compares traced output textually against this file
// (tests/golden_v2_gemm_pipelined.rs); update with BLESS=1 cargo test.

#lay = #patina.umma_k_major<swizzle_none>

patina.kernel @tcgen05_gemm_v2
    version = 0.1, arch = #patina.arch<sm_100a>, block = 192,
    roles = [#patina.role<load, 1>, #patina.role<mma, 1>, #patina.role<epi, 4>]
  (%tma_a:   !patina.tma_map<f16, #lay>,
   %tma_b:   !patina.tma_map<f16, #lay>,
   %c:       !patina.ptr<f32, gmem>,
   %k_tiles: !patina.u32) {

  %pipe = patina.pipeline.create stages = 4 : !patina.pipeline<4>

  %sa = patina.smem_alloc stages = 4 : !patina.tensor<f16, 128x32, smem, #lay>
  %sb = patina.smem_alloc stages = 4 : !patina.tensor<f16, 128x32, smem, #lay>

  %acc_done = patina.mbarrier.create arrivals = 1 : !patina.mbarrier<1>

  %acc = patina.tmem.alloc cols = 128 : !patina.tmem<f32, 128>

  patina.warp_specialize {

    patina.role load {
      // producer: wait empty[s] (initial phase 1), publish full[s]
      patina.for %i = 0 to %k_tiles iter_args(%s = 0, %ph = 1) {
        patina.wait_stage %pipe, empty, %s, %ph
        patina.expect_tx_stage %pipe, full, %s, 16384
        %sa_s = patina.stage_buf %sa, %s : !patina.tensor<f16, 128x32, smem, #lay>
        %sb_s = patina.stage_buf %sb, %s : !patina.tensor<f16, 128x32, smem, #lay>
        %i4 = patina.muli %i, 4
        patina.tma_load %tma_a -> %sa_s, coords [0, 0, %i4, 0], complete %pipe, full, %s
        patina.tma_load %tma_b -> %sb_s, coords [0, 0, %i4, 0], complete %pipe, full, %s
        %s1, %ph1 = patina.stage_next %s, %ph of %pipe
        patina.yield %s1, %ph1
      }
    }

    patina.role mma {
      // consumer: wait full[s] (initial phase 0), release empty[s]
      patina.for %i = 0 to %k_tiles iter_args(%s = 0, %ph = 0) {
        patina.wait_stage %pipe, full, %s, %ph
        %sa_s = patina.stage_buf %sa, %s : !patina.tensor<f16, 128x32, smem, #lay>
        %sb_s = patina.stage_buf %sb, %s : !patina.tensor<f16, 128x32, smem, #lay>
        %cont = patina.nez %i
        %m = patina.umma.begin %acc, %sa_s, %sb_s
        patina.umma.step %m, accum %cont
        patina.umma.advance_k %m, 16
        patina.umma.step %m {accum = true}
        patina.umma.commit %m -> %pipe, empty, %s
        %s1, %ph1 = patina.stage_next %s, %ph of %pipe
        patina.yield %s1, %ph1
      }
      // drain: one more commit on acc_done so the epilogue can wait the MMA
      // group (%m names the in-loop chain; tcgen05.commit takes no operand)
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
