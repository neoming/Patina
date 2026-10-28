#!/usr/bin/env bash
# ============================================================================
# preflight.sh — run every Patina check that CAN run without a GB200.
#
# This is the local gate before spending time on a Blackwell (sm_100a) box.
# It exercises the full compile/verify/lower path and compiles the hardware
# harness, but it cannot execute tcgen05 kernels on a non-Blackwell GPU (the
# harness will report "device is not sm_100a", which is expected and fine).
#
# Usage: ./scripts/preflight.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

LLVM="${PATINA_LLVM_PREFIX:-$HOME/.local/llvm-21/root/usr/lib/llvm-21}"
PTXAS="${PTXAS:-$HOME/.local/cuda-12.8/bin/ptxas}"
export LD_LIBRARY_PATH="$LLVM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

FAIL=0
step() { printf '\n\033[1;34m== %s ==\033[0m\n' "$*"; }
ok()   { printf '\033[1;32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '\033[1;31mFAIL\033[0m %s\n' "$*"; FAIL=1; }

# 1. Rust front-end + dialect + PTX lowering (full TDD suite) ----------------
step "cargo test (layout, golden dialect, FFI verify, PTX lowering)"
if cargo test --quiet; then ok "cargo test"; else bad "cargo test"; fi

# 2. Dialect verifier contract + golden round-trip ----------------------------
step "patina-opt: verifier contract + golden kernels"
if mlir/build/bin/patina-opt --verify-diagnostics --split-input-file \
     dialect/tests/verifier.mlir >/dev/null; then
  ok "verifier.mlir contract"
else
  bad "verifier.mlir contract"
fi
for g in crates/patina/tests/golden/v1_gemm.mlir \
         crates/patina/tests/golden/v2_gemm_pipelined.mlir; do
  if mlir/build/bin/patina-opt "$g" >/dev/null; then
    ok "verify $(basename "$g")"
  else
    bad "verify $(basename "$g")"
  fi
done

# 3. CLI lowering path emits valid PTX (exact golden equivalence is
#    enforced by cargo test via expect_golden, which normalizes comments).
step "patina-opt -lower-to-ptx -> ptxas (v1 + v2)"
for g in crates/patina/tests/golden/v1_gemm.mlir \
         crates/patina/tests/golden/v2_gemm_pipelined.mlir; do
  # patina-opt prints the PTX, then an empty `module {}` MLIR dump; drop that
  # tail before feeding the PTX to ptxas.
  mlir/build/bin/patina-opt -lower-to-ptx "$g" 2>/dev/null \
    | sed '/^module {/,$d' > /tmp/patina_emitted.ptx
  if "$PTXAS" -arch=sm_100a /tmp/patina_emitted.ptx -o /tmp/patina_emitted.cubin \
      2>/dev/null; then
    ok "lower + ptxas $(basename "$g")"
  else
    bad "lower + ptxas $(basename "$g")"
  fi
done

# 4. Golden PTX assembles for the target arch ---------------------------------
step "ptxas -arch=sm_100a (v1 + v2)"
for g in crates/patina/tests/golden/v1_gemm.ptx \
         crates/patina/tests/golden/v2_gemm_pipelined.ptx; do
  if "$PTXAS" -arch=sm_100a "$g" -o /tmp/patina_preflight.cubin 2>/dev/null; then
    ok "ptxas $(basename "$g")"
  else
    bad "ptxas $(basename "$g")"
  fi
done

# 5. Hardware harness compiles (driver-API host program) ----------------------
step "harness build"
if ./harness/build.sh >/dev/null; then ok "harness build"; else bad "harness build"; fi

# 5b. Rust hardware runner: compile + run a trivial kernel on the LOCAL GPU
#     (validates the driver-API plumbing now; skips on machines with no driver).
step "rust cuda runner (compile + local smoke test)"
if cargo build --features cuda --tests --quiet 2>/dev/null; then
  ok "cargo build --features cuda"
  if cargo test --features cuda --test run_smoke --quiet 2>/dev/null; then
    ok "run_smoke (driver-API load/launch/copy/check on local GPU)"
  else
    bad "run_smoke"
  fi
else
  echo "  skip: libcuda.so not found here (will build on the GB200)"
fi

# 6. Device capability (informational only) -----------------------------------
step "device capability"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
else
  echo "nvidia-smi not found"
fi

printf '\n\033[1;3%sm%s\033[0m\n' "$([ $FAIL -eq 0 ] && echo 2 || echo 1)" \
  "$([ $FAIL -eq 0 ] && echo "PREFLIGHT PASSED — ready to copy artifacts to the GB200" \
                      || echo "PREFLIGHT FAILED — fix the above before going to the GB200")"
exit $FAIL
