//===- PatinaOps.h - Patina dialect ops -------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PATINA_PATINAOPS_H
#define PATINA_PATINAOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "patina/PatinaDialect.h"

#define GET_OP_CLASSES
#include "patina/PatinaOps.h.inc"

namespace patina {
/// Find the single-result patina op inside `kernel` whose captured
/// `value_name` equals `name` (nullptr if none).
mlir::Operation *findDefByName(KernelOp kernel, llvm::StringRef name);
/// The type of the value called `name`: a patina op result type, or a kernel
/// entry-block argument type via the kernel's `arg_names` attribute.
mlir::Type resolveValueType(KernelOp kernel, llvm::StringRef name);
} // namespace patina

#endif // PATINA_PATINAOPS_H
