# C ABI shared library consumed by the Rust crate (crates/patina/build.rs).
add_library(patina_mlir_c SHARED PatinaCAPI.cpp)
target_link_libraries(patina_mlir_c PRIVATE PatinaMLIR MLIR)
target_include_directories(patina_mlir_c
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
