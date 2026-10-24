// dialect/tests/verifier.mlir
//
// Verifier contract for patina-opt. Each case is separated by `// -----`
// and runs independently:
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
