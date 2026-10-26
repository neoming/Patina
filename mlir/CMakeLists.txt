# Root CMake project for the Patina MLIR dialect, patina-opt, and the
# patina_mlir_c C-ABI library. Requires an LLVM/MLIR dev tree; see README.md.

cmake_minimum_required(VERSION 3.20)
project(patina-mlir LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# The deb layout ships mlir-tblgen next to the libs; MLIRConfig leaves
# MLIR_TABLEGEN_EXE as a bare name, so pin it to the install bin dir.
if(NOT EXISTS "${MLIR_TABLEGEN_EXE}")
  set(MLIR_TABLEGEN_EXE "${MLIR_INSTALL_PREFIX}/bin/mlir-tblgen")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Find libMLIR/libLLVM at runtime without LD_LIBRARY_PATH.
set(CMAKE_BUILD_RPATH "${MLIR_INSTALL_PREFIX}/lib")

add_subdirectory(include/patina)
add_subdirectory(lib)
add_subdirectory(tools/patina-opt)
