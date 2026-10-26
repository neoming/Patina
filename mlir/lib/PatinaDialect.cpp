//===- PatinaDialect.cpp - dialect registration --------------------------===//

#include "patina/PatinaDialect.h"
#include "patina/PatinaOps.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "patina/PatinaDialect.cpp.inc"

#include "patina/PatinaEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "patina/PatinaAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "patina/PatinaTypes.cpp.inc"

using namespace patina;

mlir::Attribute
PatinaDialect::parseAttribute(mlir::DialectAsmParser &parser,
                              mlir::Type type) const {
  llvm::SMLoc loc = parser.getCurrentLocation();
  llvm::StringRef mnemonic;
  mlir::Attribute attr;
  mlir::OptionalParseResult res =
      generatedAttributeParser(parser, &mnemonic, type, attr);
  if (res.has_value() && mlir::succeeded(*res))
    return attr;
  if (!res.has_value())
    parser.emitError(loc) << "unknown patina attribute: " << mnemonic;
  return {};
}

void PatinaDialect::printAttribute(mlir::Attribute attr,
                                   mlir::DialectAsmPrinter &printer) const {
  (void)generatedAttributePrinter(attr, printer);
}

mlir::Type PatinaDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::SMLoc loc = parser.getCurrentLocation();
  llvm::StringRef mnemonic;
  mlir::Type type;
  mlir::OptionalParseResult res = generatedTypeParser(parser, &mnemonic, type);
  if (res.has_value() && mlir::succeeded(*res))
    return type;
  if (!res.has_value())
    parser.emitError(loc) << "unknown patina type: " << mnemonic;
  return {};
}

void PatinaDialect::printType(mlir::Type type,
                              mlir::DialectAsmPrinter &printer) const {
  (void)generatedTypePrinter(type, printer);
}

void PatinaDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "patina/PatinaAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "patina/PatinaTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "patina/PatinaOps.cpp.inc"
      >();
}
