//===- PatinaDialect.td - Patina dialect definition ---------*- tablegen -*-===//
//
// Patina: a thin typed layer over tcgen05 (Blackwell warp-specialized GEMM).
// One level above PTX: no descriptors, no parities, no lane math — every op
// names its PTX realization in its doc. See PROPOSAL.md section 6.
//
// BUILD REQUIREMENT: an LLVM/MLIR dev tree with mlir-tblgen. The M1c build is
// deferred; see dialect/README.md.
//
//===----------------------------------------------------------------------===//

#ifndef PATINA_DIALECT_TD
#define PATINA_DIALECT_TD

include "mlir/IR/DialectBase.td"

def Patina_Dialect : Dialect {
  let name = "patina";
  let cppNamespace = "::patina";

  let summary = "Warp-specialized Blackwell (tcgen05) kernel dialect";
  let description = [{
    The Patina dialect models single-CTA (v1) warp-specialized tensor-core
    kernels: a `patina.kernel` holds CTA-scope allocations and one
    `patina.warp_specialize` dispatch with one `patina.role` region per role
    (load / mma / epi). Kernels are versioned from day one via the `version`
    attribute so lowering-pass churn fails loudly.

    Structural rules enforced by the dialect verifiers (contract:
    dialect/tests/verifier.mlir — keep diagnostic strings in sync):

      1. `patina.umma.*` ops are only legal inside an `mma` role region.
      2. `patina.expect_tx` bytes == sum of the bytes of every
         `patina.tma_load` completing on the same barrier.
      3. Every `patina.tmem.alloc` result is consumed by exactly one
         `patina.tmem.dealloc`.
      4. Barrier phases are waited in order per role region.
      5. Sum of role warp counts x 32 == the kernel's `block` threads.
  }];
}

#endif // PATINA_DIALECT_TD
