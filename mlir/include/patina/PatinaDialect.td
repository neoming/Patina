//===- PatinaDialect.td - Patina dialect definition ---------*- tablegen -*-===//
//
// Patina: a thin typed layer over tcgen05 (Blackwell warp-specialized GEMM).
// One level above PTX: no descriptors, no parities, no lane math — every op
// names its PTX realization in its doc. See PROPOSAL.md section 6.
//
// Verifier diagnostics must stay in sync with dialect/tests/verifier.mlir
// (docs/TESTING.md change rules).
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

    Value references use symbolic names (the golden syntax annotates types
    only at definitions), so ops carry `StringAttr` references and a captured
    `name` attribute per result; verifiers resolve names within the kernel.

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

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;

  let extraClassDeclaration = [{
    mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                   mlir::Type type) const override;
    void printAttribute(mlir::Attribute attr,
                        mlir::DialectAsmPrinter &printer) const override;
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;
  }];
}

#endif // PATINA_DIALECT_TD
