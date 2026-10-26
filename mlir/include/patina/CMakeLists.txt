set(LLVM_TARGET_DEFINITIONS PatinaDialect.td)
mlir_tablegen(PatinaDialect.h.inc -gen-dialect-decls -dialect patina)
mlir_tablegen(PatinaDialect.cpp.inc -gen-dialect-defs -dialect patina)

set(LLVM_TARGET_DEFINITIONS PatinaAttrs.td)
mlir_tablegen(PatinaEnums.h.inc -gen-enum-decls)
mlir_tablegen(PatinaEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(PatinaAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(PatinaAttrs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS PatinaTypes.td)
mlir_tablegen(PatinaTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PatinaTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS PatinaOps.td)
mlir_tablegen(PatinaOps.h.inc -gen-op-decls)
mlir_tablegen(PatinaOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(PatinaIncGen)
