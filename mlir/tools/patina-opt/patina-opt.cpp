//===- patina-opt.cpp - Patina dialect parser/verifier driver -------------===//
//
// Usage:
//   patina-opt file.mlir                                   (parse + verify)
//   patina-opt --verify-diagnostics --split-input-file dialect/tests/verifier.mlir
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "patina/PatinaDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<patina::PatinaDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "patina-opt\n", registry));
}
