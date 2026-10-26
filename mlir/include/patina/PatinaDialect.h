//===- PatinaDialect.h - Patina dialect -------------------------*- C++ -*-===//
//
// Umbrella header: dialect, enums, attributes, types.
//
//===----------------------------------------------------------------------===//

#ifndef PATINA_PATINADIALECT_H
#define PATINA_PATINADIALECT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "patina/PatinaDialect.h.inc"

#include "patina/PatinaEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "patina/PatinaAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "patina/PatinaTypes.h.inc"

#endif // PATINA_PATINADIALECT_H
