#!/usr/bin/env bash
# Build the host-side driver-API harness (no nvcc needed — the PTX is already
# compiled; the driver JITs it for the target device).
#
# Usage: ./build.sh   (or: CXX=g++-13 CUDA_HOME=/usr/local/cuda ./build.sh)
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-g++-13}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# --- locate cuda.h ----------------------------------------------------------
INCDIR=""
for d in "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include" \
         /usr/local/cuda/include /usr/include; do
  if [ -f "$d/cuda.h" ]; then INCDIR="$d"; break; fi
done
if [ -z "$INCDIR" ]; then
  echo "error: cuda.h not found (set CUDA_HOME)" >&2
  exit 1
fi

# --- locate libcuda.so (stubs are fine for linking; the driver provides the
#     real libcuda.so.1 at runtime) ------------------------------------------
LIBDIR=""
for d in "$CUDA_HOME/lib64/stubs" "$CUDA_HOME/lib64" \
         "$CUDA_HOME/targets/x86_64-linux/lib/stubs" \
         "$CUDA_HOME/targets/x86_64-linux/lib" \
         /usr/local/cuda/lib64/stubs /usr/local/cuda/lib64 \
         /usr/lib/x86_64-linux-gnu/stubs /usr/lib/x86_64-linux-gnu; do
  if [ -f "$d/libcuda.so" ] || [ -f "$d/libcuda.so.1" ]; then LIBDIR="$d"; break; fi
done
if [ -z "$LIBDIR" ]; then
  echo "error: libcuda.so not found (is the NVIDIA driver installed?)" >&2
  exit 1
fi

echo ">> compiling harness/gemm_check.cpp (cuda.h: $INCDIR, libcuda: $LIBDIR)"
"$CXX" -O2 -std=c++17 -Wall -Wextra -I"$INCDIR" gemm_check.cpp \
  -o gemm_check -L"$LIBDIR" -lcuda -lm
echo ">> wrote harness/gemm_check"
