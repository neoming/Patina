//===- PatinaTypes.td - Patina dialect types ----------------*- tablegen -*-===//
//
// Types of the Patina dialect. All use custom parse/print (implemented in
// lib/PatinaTypes.cpp): shapes print as `128x32`, spaces as bare keywords.
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
  let hasCustomAssemblyFormat = 1;
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
    "mlir::Type":$dtype,
    ArrayRefParameter<"int64_t">:$shape,
    EnumParameter<Patina_Space>:$space,
    "mlir::Attribute":$layout);
  // Custom parse/print in PatinaTypes.cpp.
}

// TMEM accumulator allocation: `!patina.tmem<f32, 128>` (dtype, columns).
def Patina_TmemType : Patina_Type<"Tmem", "tmem"> {
  let summary = "TMEM accumulator: element dtype and column count";
  let parameters = (ins "mlir::Type":$dtype, "uint32_t":$cols);
}

// MBarrier with arrival count in the type: `!patina.mbarrier<1>`.
def Patina_MBarrierType : Patina_Type<"MBarrier", "mbarrier"> {
  let summary = "mbarrier with a static arrival count";
  let parameters = (ins "uint32_t":$arrivals);
}

// Host-encoded TMA tensor map: `!patina.tma_map<f16, #lay>`.
def Patina_TmaMapType : Patina_Type<"TmaMap", "tma_map"> {
  let summary = "TMA tensor map (CUtensorMap) for a layout";
  let parameters = (ins "mlir::Type":$dtype, "mlir::Attribute":$layout);
}

// Register fragment loaded back from TMEM: `!patina.frag<f32, 128>`.
def Patina_FragType : Patina_Type<"Frag", "frag"> {
  let summary = "per-thread register fragment of a TMEM accumulator";
  let parameters = (ins "mlir::Type":$dtype, "uint32_t":$elems);
}

// Typed pointer: `!patina.ptr<f32, gmem>`.
def Patina_PtrType : Patina_Type<"Ptr", "ptr"> {
  let summary = "typed pointer into a memory space";
  let parameters = (ins "mlir::Type":$dtype, EnumParameter<Patina_Space>:$space);
}

// In-flight UMMA chain value (never printed with an explicit type).
def Patina_UmmaType : Patina_Type<"Umma", "umma"> {
  let summary = "in-flight UMMA begin/step/advance_k/commit chain";
}

// Runtime 32-bit unsigned scalar (k_tiles, loop index, stage, phase):
// `!patina.u32`. Ring pipelines carry phases as VALUES of this type
// (PROPOSAL open question 2, resolved in M3).
def Patina_U32Type : Patina_Type<"U32", "u32"> {
  let summary = "runtime 32-bit unsigned scalar value";
}

// S-stage smem ring pipeline (full[S] + empty[S] mbarrier arrays):
// `!patina.pipeline<4>`.
def Patina_PipelineType : Patina_Type<"Pipeline", "pipeline"> {
  let summary = "S-stage ring pipeline: full[S] + empty[S] mbarrier arrays";
  let parameters = (ins "uint32_t":$stages);
}

#endif // PATINA_TYPES_TD
