add_executable(patina-opt patina-opt.cpp)
target_link_libraries(patina-opt PRIVATE PatinaMLIR MLIRMlirOptMain MLIR)
