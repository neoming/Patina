// dialect/tests/verifier.mlir
//
// Verifier contract for patina-opt. Each case is separated by a `//` dash
// line and runs independently:
//   RUN: patina-opt --verify-diagnostics --split-input-file %s
//
// The first module must verify clean; every following module must fail with
// the annotated diagnostic. Keep messages in sync with dialect/PatinaOps.td.

#lay = #patina.umma_k_major<swizzle_none>

// ===========================================================================
// valid: minimal well-formed kernel (single-role is legal; warp_specialize
// with one role is degenerate but sound)
// ===========================================================================
patina.kernel @ok
    version = 0.1, arch = #patina.arch<sm_100a>, block = 32,
    roles = [#patina.role<mma, 1>]
  (%c: !patina.ptr<f32, gmem>) {
  %acc = patina.tmem.alloc cols = 32 : !patina.tmem<f32, 32>
  patina.warp_specialize {
    patina.role mma { }
  }
  patina.tmem.dealloc %acc
}

// -----
// invalid: umma op inside a load role region
// ===========================================================================
#lay = #patina.umma_k_major<swizzle_none>

patina.kernel @bad_role
    version = 0.1, arch = #patina.arch<sm_100a>, block = 64,
    roles = [#patina.role<load, 1>, #patina.role<mma, 1>]
  (%sa: !patina.tensor<f16, 128x32, smem, #lay>,
   %sb: !patina.tensor<f16, 128x32, smem, #lay>) {
  %acc = patina.tmem.alloc cols = 128 : !patina.tmem<f32, 128>
  patina.warp_specialize {
    patina.role load {
      // expected-error @+1 {{'patina.umma.begin' is only legal inside an 'mma' role region}}
      %m = patina.umma.begin %acc, %sa, %sb
    }
    patina.role mma { }
  }
  patina.tmem.dealloc %acc
}

// -----
// invalid: expect_tx byte count does not match the copies on that barrier
// ===========================================================================
#lay = #patina.umma_k_major<swizzle_none>

patina.kernel @bad_tx
    version = 0.1, arch = #patina.arch<sm_100a>, block = 32,
    roles = [#patina.role<load, 1>]
  (%tma_a: !patina.tma_map<f16, #lay>) {
  %sa = patina.smem_alloc : !patina.tensor<f16, 128x32, smem, #lay>
  %bar = patina.mbarrier.create arrivals = 1 : !patina.mbarrier<1>
  patina.warp_specialize {
    patina.role load {
      // expected-error @+1 {{expect_tx bytes (4096) != sum of tma_load bytes on this barrier (8192)}}
      patina.expect_tx %bar, 4096
      patina.tma_load %tma_a -> %sa, coords [0, 0, 0, 0], complete %bar
    }
  }
}

// -----
// invalid: tmem.alloc with no reachable dealloc
// ===========================================================================
patina.kernel @tmem_leak
    version = 0.1, arch = #patina.arch<sm_100a>, block = 32,
    roles = [#patina.role<mma, 1>]
  () {
  // expected-error @+1 {{tmem.alloc result must be consumed by exactly one patina.tmem.dealloc}}
  %acc = patina.tmem.alloc cols = 32 : !patina.tmem<f32, 32>
  patina.warp_specialize {
    patina.role mma { }
  }
}

// -----
// invalid: barrier phases must be waited in order per role
// ===========================================================================
patina.kernel @phase_skip
    version = 0.1, arch = #patina.arch<sm_100a>, block = 32,
    roles = [#patina.role<mma, 1>]
  () {
  %bar = patina.mbarrier.create arrivals = 1 : !patina.mbarrier<1>
  %acc = patina.tmem.alloc cols = 32 : !patina.tmem<f32, 32>
  patina.warp_specialize {
    patina.role mma {
      // expected-error @+1 {{phase 1 waited before phase 0 on this barrier}}
      patina.wait %bar, phase = 1
    }
  }
  patina.tmem.dealloc %acc
}

// -----
// invalid: role warp counts exceed the block
// ===========================================================================
// expected-error @+1 {{roles require 8 warps (256 threads) but block = 128}}
patina.kernel @warp_overflow
    version = 0.1, arch = #patina.arch<sm_100a>, block = 128,
    roles = [#patina.role<load, 1>, #patina.role<mma, 1>, #patina.role<epi, 6>]
  () {
  %acc = patina.tmem.alloc cols = 32 : !patina.tmem<f32, 32>
  patina.warp_specialize {
    patina.role load { }
    patina.role mma  { }
    patina.role epi  { }
  }
  patina.tmem.dealloc %acc
}

// -----
// valid: compact pipelined mainloop (same shape as the v2 golden kernel)
// ===========================================================================
#lay = #patina.umma_k_major<swizzle_none>

patina.kernel @pipe_ok
    version = 0.1, arch = #patina.arch<sm_100a>, block = 64,
    roles = [#patina.role<load, 1>, #patina.role<mma, 1>]
  (%tma_a: !patina.tma_map<f16, #lay>,
   %k_tiles: !patina.u32) {
  %pipe = patina.pipeline.create stages = 4 : !patina.pipeline<4>
  %sa = patina.smem_alloc stages = 4 : !patina.tensor<f16, 128x32, smem, #lay>
  %acc_done = patina.mbarrier.create arrivals = 1 : !patina.mbarrier<1>
  %acc = patina.tmem.alloc cols = 128 : !patina.tmem<f32, 128>
  patina.warp_specialize {
    patina.role load {
      patina.for %i = 0 to %k_tiles iter_args(%s = 0, %ph = 1) {
        patina.wait_stage %pipe, empty, %s, %ph
        patina.expect_tx_stage %pipe, full, %s, 8192
        %sa_s = patina.stage_buf %sa, %s : !patina.tensor<f16, 128x32, smem, #lay>
        patina.tma_load %tma_a -> %sa_s, coords [0, 0, 0, 0], complete %pipe, full, %s
        %s1, %ph1 = patina.stage_next %s, %ph of %pipe
        patina.yield %s1, %ph1
      }
    }
    patina.role mma {
      patina.for %i = 0 to %k_tiles iter_args(%s = 0, %ph = 0) {
        patina.wait_stage %pipe, full, %s, %ph
        %sa_s = patina.stage_buf %sa, %s : !patina.tensor<f16, 128x32, smem, #lay>
        %m = patina.umma.begin %acc, %sa_s, %sa_s
        patina.umma.step %m {accum = true}
        patina.umma.commit %m -> %pipe, empty, %s
        %s1, %ph1 = patina.stage_next %s, %ph of %pipe
        patina.yield %s1, %ph1
      }
      patina.umma.commit %m -> %acc_done
    }
  }
  patina.tmem.dealloc %acc
}

// -----
// invalid: producer-side wait (empty) inside the mma role
// ===========================================================================
patina.kernel @bad_wait_role
    version = 0.1, arch = #patina.arch<sm_100a>, block = 64,
    roles = [#patina.role<load, 1>, #patina.role<mma, 1>]
  (%k_tiles: !patina.u32) {
  %pipe = patina.pipeline.create stages = 4 : !patina.pipeline<4>
  patina.warp_specialize {
    patina.role load { }
    patina.role mma {
      patina.for %i = 0 to %k_tiles iter_args(%s = 0, %ph = 0) {
        // expected-error @+1 {{'patina.wait_stage' on 'empty' barriers is only legal in a 'load' role region}}
        patina.wait_stage %pipe, empty, %s, %ph
        patina.yield %s, %ph
      }
    }
  }
}

// -----
// invalid: expect_tx_stage byte count does not match the staged copies
// ===========================================================================
#lay = #patina.umma_k_major<swizzle_none>

patina.kernel @bad_tx_stage
    version = 0.1, arch = #patina.arch<sm_100a>, block = 64,
    roles = [#patina.role<load, 1>, #patina.role<mma, 1>]
  (%tma_a: !patina.tma_map<f16, #lay>,
   %k_tiles: !patina.u32) {
  %pipe = patina.pipeline.create stages = 4 : !patina.pipeline<4>
  %sa = patina.smem_alloc stages = 4 : !patina.tensor<f16, 128x32, smem, #lay>
  patina.warp_specialize {
    patina.role load {
      patina.for %i = 0 to %k_tiles iter_args(%s = 0, %ph = 1) {
        patina.wait_stage %pipe, empty, %s, %ph
        // expected-error @+1 {{expect_tx_stage bytes (4096) != sum of tma_load bytes on this stage barrier (8192)}}
        patina.expect_tx_stage %pipe, full, %s, 4096
        %sa_s = patina.stage_buf %sa, %s : !patina.tensor<f16, 128x32, smem, #lay>
        patina.tma_load %tma_a -> %sa_s, coords [0, 0, 0, 0], complete %pipe, full, %s
        patina.yield %s, %ph
      }
    }
    patina.role mma { }
  }
}

// -----
// invalid: patina.yield must yield exactly the iter_args of the enclosing for
// ===========================================================================
patina.kernel @bad_yield
    version = 0.1, arch = #patina.arch<sm_100a>, block = 32,
    roles = [#patina.role<mma, 1>]
  (%k_tiles: !patina.u32) {
  patina.warp_specialize {
    patina.role mma {
      patina.for %i = 0 to %k_tiles iter_args(%s = 0, %ph = 0) {
        // expected-error @+1 {{patina.yield operand count (1) does not match patina.for iter_args (2)}}
        patina.yield %s
      }
    }
  }
}
