//===- Passes.h - patina lowering passes ------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PATINA_PASSES_H
#define PATINA_PASSES_H

#include <optional>
#include <string>

#include "mlir/IR/BuiltinOps.h"

namespace patina {

/// Lower the single patina.kernel in `module` to PTX source text (the v1
/// warp-specialized form: load/mma single-warp roles, one TMEM accumulator,
/// f16 in / f32 acc, UmmaKMajor swizzle_none smem). Returns nullopt and
/// fills `err` for anything the v1 lowering does not cover.
std::optional<std::string> lowerModuleToPTX(mlir::ModuleOp module,
                                            std::string &err);

/// Register the patina passes (currently: -lower-to-ptx).
void registerPatinaPasses();

} // namespace patina

#endif // PATINA_PASSES_H
