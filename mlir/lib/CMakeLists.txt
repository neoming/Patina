add_mlir_dialect_library(PatinaMLIR
  PatinaDialect.cpp
  PatinaTypes.cpp
  PatinaOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../include

  DEPENDS
  PatinaIncGen

  LINK_LIBS PUBLIC
  MLIR
)
