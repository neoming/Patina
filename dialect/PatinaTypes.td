//===- PatinaTypes.td - Patina dialect types ----------------*- tablegen -*-===//
//
// Types of the Patina dialect. Requires an LLVM/MLIR dev tree (mlir-tblgen);
// M1c build deferred. Element dtypes print as MLIR builtin float types
// (`f16`, `f32`); shapes print in `128x32` form, which needs a custom
// parser/printer hook (assemblyFormat sketches the structure only).
//
//===----------------------------------------------------------------------===//

#ifndef PATINA_TYPES_TD
#define PATINA_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "PatinaDialect.td"
include "PatinaAttrs.td"

class Patina_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Patina_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// A typed tensor; layout is part of the type so UMMA-illegal states are
// unrepresentable. `!patina.tensor<f16, 128x32, smem, #lay>`
def Patina_TensorType : Patina_Type<"Tensor", "tensor"> {
  let summary = "typed tensor: dtype, shape, memory space, layout";
  let description = [{
    Layout is part of the type: `smem_descriptor` derivation exists only
    where legal, and the layout attr drives both the host TMA tensor-map
    encode and the device UMMA descriptor packing.
  }];
  let parameters = (ins
    "mlir::Type" : $dtype,
    "llvm::ArrayRef<int64_t>" : $shape,
    "Patina_Space" : $space,
    "mlir::Attribute" : $layout);
  // Custom parser prints $shape as `128x32` and $space as a bare keyword.
}

// TMEM accumulator allocation: `!patina.tmem<f32, 128>` (dtype, columns).
def Patina_TmemType : Patina_Type<"Tmem", "tmem"> {
  let summary = "TMEM accumulator: element dtype and column count";
  let parameters = (ins "mlir::Type" : $dtype, "uint32_t" : $cols);
}

// MBarrier with arrival count in the type: `!patina.mbarrier<1>`.
def Patina_MBarrierType : Patina_Type<"MBarrier", "mbarrier"> {
  let summary = "mbarrier with a static arrival count";
  let parameters = (ins "uint32_t" : $arrivals);
}

// Host-encoded TMA tensor map: `!patina.tma_map<f16, #lay>`.
def Patina_TmaMapType : Patina_Type<"TmaMap", "tma_map"> {
  let summary = "TMA tensor map (CUtensorMap) for a layout";
  let parameters = (ins "mlir::Type" : $dtype, "mlir::Attribute" : $layout);
}

// Register fragment loaded back from TMEM: `!patina.frag<f32, 128>`.
def Patina_FragType : Patina_Type<"Frag", "frag"> {
  let summary = "per-thread register fragment of a TMEM accumulator";
  let parameters = (ins "mlir::Type" : $dtype, "uint32_t" : $elems);
}

// Typed pointer: `!patina.ptr<f32, gmem>`.
def Patina_PtrType : Patina_Type<"Ptr", "ptr"> {
  let summary = "typed pointer into a memory space";
  let parameters = (ins "mlir::Type" : $dtype, "Patina_Space" : $space);
}

#endif // PATINA_TYPES_TD
