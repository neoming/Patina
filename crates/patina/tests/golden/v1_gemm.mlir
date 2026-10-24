// tests/golden/v1_gemm.mlir
//
// Golden dialect output for the v1 warp-specialized single-tile GEMM.
// Reference: ~/WorkSpace/tcgen05_gemm/tcgen05_gemm.cu (TMA path).
//
// RUN (once patina-opt exists): patina-opt %s > /dev/null   // must verify clean
// The Rust side compares traced output textually against this file
// (tests/golden_v1_gemm.rs); update with BLESS=1 after intentional changes.

#lay = #patina.umma_k_major<swizzle_none>

patina.kernel @tcgen05_gemm_v1
    version = 0.1, arch = #patina.arch<sm_100a>, block = 192,
    roles = [#patina.role<load, 1>, #patina.role<mma, 1>, #patina.role<epi, 4>]
  (%tma_a: !patina.tma_map<f16, #lay>,
   %tma_b: !patina.tma_map<f16, #lay>,
   %c:     !patina.ptr<f32, gmem>) {

  %sa = patina.smem_alloc : !patina.tensor<f16, 128x32, smem, #lay>
  %sb = patina.smem_alloc : !patina.tensor<f16, 128x32, smem, #lay>

  %load_done = patina.mbarrier.create arrivals = 1 : !patina.mbarrier<1>
  %mma_done  = patina.mbarrier.create arrivals = 1 : !patina.mbarrier<1>

  // warp-collective on warp 0, before dispatch; dealloc after the role join
  %acc = patina.tmem.alloc cols = 128 : !patina.tmem<f32, 128>

  patina.warp_specialize {

    patina.role load {
      // byte count must equal the sum of both tiles (8192 + 8192)
      patina.expect_tx %load_done, 16384
      patina.tma_load %tma_a -> %sa, coords [0, 0, 0, 0], complete %load_done
      patina.tma_load %tma_b -> %sb, coords [0, 0, 0, 0], complete %load_done
    }

    patina.role mma {
      patina.wait %load_done, phase = 0
      // idesc (f16 in, f32 acc, M=128, N=128, K-major) is packed by lowering
      // from the operand types; descriptors (LBO=128, SBO=256) from #lay
      %m = patina.umma.begin %acc, %sa, %sb
      patina.umma.step %m {accum = false}   // K16 #0, scaleC = 0
      patina.umma.advance_k %m, 16          // 16 halves = 2 core matrices
      patina.umma.step %m {accum = true}    // K16 #1, scaleC = 1
      patina.umma.commit %m -> %mma_done
    }

    patina.role epi {
      patina.wait %mma_done, phase = 0
      // 4 x tcgen05.ld.32x32b.x32; TMEM lane field derived from warp rank
      %frag = patina.tmem_ld %acc : !patina.frag<f32, 128>
      patina.store %c, %frag                // st.global.v4, coalesced
    }
  }

  patina.tmem.dealloc %acc
}
