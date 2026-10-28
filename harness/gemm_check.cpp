// ============================================================================
// gemm_check — end-to-end numerical verification for Patina tcgen05 GEMM PTX.
//
// Loads a Patina-generated PTX kernel with the CUDA *driver* API, runs it on
// the current device with deterministic inputs, and checks
//
//     C[m,n] = sum_k A[m,k] * B[n,k]          (i.e. C = A @ B^T)
//
// against a host-side reference computed in double precision. A[128,K] and
// B[128,K] are f16 (K-major, row-major in memory); C[128,128] is f32.
//
// The two kernels share one TMA tensor-map layout (see
// crates/patina/src/layout.rs, UmmaKMajor::tma_view) parameterized by K:
//
//   gdim  = {8, 8, 4*k_tiles, 16}   (f16 elements per dim)
//   gstrd = {2*K, 16, 16*K}         (bytes, dims 1..=3)
//   box   = {8, 8, 4, 16}           (ONE K-tile == 32 f16)
//   coords = {0, 0, 4*i, 0}         (the kernel advances dim2 by 4 per tile)
//
// Host-only: compiled with g++ (no nvcc) against cuda.h + libcuda. The PTX is
// JIT-compiled by the driver, so the *driver* (not the toolkit) must support
// sm_100a (tcgen05 requires Blackwell; e.g. driver >= 570 on a GB200).
//
// Usage:
//   gemm_check <ptx_file> <kernel_name> <K>
//
//   kernel_name: tcgen05_gemm_v1   (K must be 32)
//                tcgen05_gemm_v2   (K must be a multiple of 32; k_tiles=K/32)
//
// Exit codes: 0 PASS, 1 FAIL (wrong result), 2 usage/launch error,
//             3 device is not sm_100a (tcgen05 unsupported here).
// ============================================================================

#include <cuda.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult r_ = (call);                                                      \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char* name = nullptr;                                              \
      cuGetErrorName(r_, &name);                                               \
      const char* msg = nullptr;                                               \
      cuGetErrorString(r_, &msg);                                              \
      std::fprintf(stderr, "CUDA error at %s:%d in %s: %s (%s)\n",            \
                   __FILE__, __LINE__, #call,                                  \
                   name ? name : "?", msg ? msg : "?");                        \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

static constexpr uint32_t M = 128;
static constexpr uint32_t N = 128;
static constexpr uint32_t BLOCK = 192;      // load(1) + mma(1) + epi(4) warps
static constexpr size_t V2_DYN_SMEM = 65536; // 4-stage ring, 2 x 4 x 8 KiB

// ---------------------------------------------------------------------------
// f16 <-> f32 (self-contained; no CUDA math library needed)
// ---------------------------------------------------------------------------
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign; // +-0
    } else {       // subnormal
      int e = -14;
      uint32_t m = mant;
      while (!(m & 0x400u)) {
        m <<= 1;
        --e;
      }
      m &= 0x3ffu;
      bits = sign | ((uint32_t)(e + 127) << 23) | (m << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (mant << 13); // inf/nan
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

static uint16_t f32_to_f16(float x) { // round-to-nearest-even
  uint32_t u;
  std::memcpy(&u, &x, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  uint32_t exp = (u >> 23) & 0xffu;
  uint32_t mant = u & 0x7fffffu;
  if (exp == 0xff) { // inf / nan
    return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
  }
  int e = (int)exp - 127 + 15;
  if (e >= 0x1f)
    return (uint16_t)(sign | 0x7c00u); // overflow -> inf
  if (e <= 0) {                        // subnormal / zero
    if (e < -10)
      return (uint16_t)sign; // underflow -> 0
    mant |= 0x800000u;
    int shift = 14 - e;
    uint32_t h = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (h & 1)))
      ++h;
    return (uint16_t)(sign | h);
  }
  uint32_t h = sign | ((uint32_t)e << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1)))
    ++h;
  return (uint16_t)h;
}

// ---------------------------------------------------------------------------
// TMA tensor map for the UMMA SWIZZLE_NONE K-major layout, parameterized by K.
// ---------------------------------------------------------------------------
static CUtensorMap make_tmap(const void* gptr, uint64_t K) {
  const uint64_t k_tiles = K / 32;
  const cuuint64_t gdim[4] = {8, 8, 4 * k_tiles, 16};
  const cuuint64_t gstrd[3] = {2 * K, 16, 16 * K}; // bytes, dims 1..=3
  const cuuint32_t box[4] = {8, 8, 4, 16};         // one K-tile (32 f16)
  const cuuint32_t estrd[4] = {1, 1, 1, 1};
  CUtensorMap tmap{};
  CUresult rc = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, const_cast<void*>(gptr), gdim,
      gstrd, box, estrd, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(rc, &msg);
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n",
                 msg ? msg : "?");
    std::exit(2);
  }
  return tmap;
}

// ---------------------------------------------------------------------------
// deterministic f16 inputs in [-1, 1]
// ---------------------------------------------------------------------------
static void fill_inputs(uint16_t* A, uint16_t* B, uint64_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint64_t k = 0; k < K; ++k) {
      float a = 0.9f * std::sin(0.1f * (m + 1) + 0.7f * (float)k) *
                std::cos(0.05f * ((float)k + 1));
      float b = 0.8f * std::cos(0.13f * (m + 1) + 0.11f * (float)k) *
                std::sin(0.03f * ((float)(m + k) + 1));
      A[m * K + k] = f32_to_f16(a);
      B[m * K + k] = f32_to_f16(b);
    }
  }
}

static void reference_gemm(const uint16_t* A, const uint16_t* B, uint64_t K,
                           double* C_ref) {
  std::memset(C_ref, 0, (size_t)M * N * sizeof(double));
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (uint64_t k = 0; k < K; ++k) {
        acc += (double)f16_to_f32(A[m * K + k]) *
               (double)f16_to_f32(B[n * K + k]);
      }
      C_ref[m * N + n] = acc;
    }
  }
}

static const char* arch_name(int major, int minor) {
  static char buf[32];
  std::snprintf(buf, sizeof(buf), "sm_%d%d", major, minor);
  return buf;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: %s <ptx_file> <kernel_name> <K>\n"
                 "  kernel_name: tcgen05_gemm_v1 (K=32) | tcgen05_gemm_v2 "
                 "(K multiple of 32)\n",
                 argv[0]);
    return 2;
  }
  const char* ptx_path = argv[1];
  const char* kernel = argv[2];
  const uint64_t K = std::strtoull(argv[3], nullptr, 10);

  bool v2;
  if (std::strcmp(kernel, "tcgen05_gemm_v1") == 0) {
    v2 = false;
    if (K != 32) {
      std::fprintf(stderr, "tcgen05_gemm_v1 requires K=32 (got %llu)\n",
                   (unsigned long long)K);
      return 2;
    }
  } else if (std::strcmp(kernel, "tcgen05_gemm_v2") == 0) {
    v2 = true;
    if (K % 32 != 0) {
      std::fprintf(stderr, "tcgen05_gemm_v2 requires K a multiple of 32 (got "
                           "%llu)\n",
                   (unsigned long long)K);
      return 2;
    }
  } else {
    std::fprintf(stderr, "unknown kernel '%s'\n", kernel);
    return 2;
  }

  CUDA_CHECK(cuInit(0));
  CUdevice dev;
  CUDA_CHECK(cuDeviceGet(&dev, 0));
  int major = 0, minor = 0;
  CUDA_CHECK(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CUDA_CHECK(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  char dev_name[128];
  CUDA_CHECK(cuDeviceGetName(dev_name, sizeof(dev_name), dev));

  std::printf("device: %s (%s)\n", dev_name, arch_name(major, minor));
  if (major < 10) {
    std::fprintf(stderr,
                 "device is %s, but tcgen05 needs sm_100a (Blackwell/GB200). "
                 "The PTX/pipeline is fine; this host GPU cannot run it.\n",
                 arch_name(major, minor));
    return 3;
  }

  CUcontext ctx;
  CUDA_CHECK(cuCtxCreate(&ctx, 0, dev));

  // Load + JIT the PTX for this device.
  FILE* f = std::fopen(ptx_path, "rb");
  if (!f) {
    std::perror(ptx_path);
    return 2;
  }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<char> ptx((size_t)sz + 1);
  if (sz > 0 && std::fread(ptx.data(), 1, (size_t)sz, f) != (size_t)sz) {
    std::perror("fread");
    return 2;
  }
  std::fclose(f);
  ptx[(size_t)sz] = '\0';

  CUmodule mod;
  CUDA_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
  CUfunction fn;
  CUDA_CHECK(cuModuleGetFunction(&fn, mod, kernel));

  // Device buffers: A, B f16 [128,K]; C f32 [128,128].
  CUdeviceptr d_A = 0, d_B = 0, d_C = 0;
  const size_t ab_bytes = (size_t)M * K * 2;
  const size_t c_bytes = (size_t)M * N * 4;
  CUDA_CHECK(cuMemAlloc(&d_A, ab_bytes));
  CUDA_CHECK(cuMemAlloc(&d_B, ab_bytes));
  CUDA_CHECK(cuMemAlloc(&d_C, c_bytes));
  CUDA_CHECK(cuMemsetD8(d_C, 0, c_bytes));

  std::vector<uint16_t> A((size_t)M * K), B((size_t)M * K);
  std::vector<double> C_ref((size_t)M * N);
  fill_inputs(A.data(), B.data(), K);
  reference_gemm(A.data(), B.data(), K, C_ref.data());

  CUDA_CHECK(cuMemcpyHtoD(d_A, A.data(), ab_bytes));
  CUDA_CHECK(cuMemcpyHtoD(d_B, B.data(), ab_bytes));

  alignas(64) CUtensorMap tma_a = make_tmap((void*)d_A, K);
  alignas(64) CUtensorMap tma_b = make_tmap((void*)d_B, K);
  uint32_t k_tiles = (uint32_t)(K / 32);

  void* params[4];
  params[0] = &tma_a;
  params[1] = &tma_b;
  params[2] = &d_C;
  params[3] = &k_tiles;
  // (param count is read from the PTX function metadata, not the API)

  CUstream stream = nullptr;
  CUDA_CHECK(cuStreamCreate(&stream, 0));
  CUDA_CHECK(cuLaunchKernel(fn, 1, 1, 1, BLOCK, 1, 1,
                            v2 ? V2_DYN_SMEM : 0, stream, params, nullptr));
  CUDA_CHECK(cuStreamSynchronize(stream));

  std::vector<float> C((size_t)M * N);
  CUDA_CHECK(cuMemcpyDtoH(C.data(), d_C, c_bytes));

  // Compare against the double-precision reference.
  double atol = 1e-3, rtol = 1e-3;
  double max_abs = 0.0, max_rel = 0.0;
  uint64_t mismatches = 0;
  double sum = 0.0;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      double got = C[m * N + n];
      double ref = C_ref[m * N + n];
      sum += got;
      double abs_err = std::fabs(got - ref);
      double rel_err = abs_err / std::fmax(1.0, std::fabs(ref));
      if (abs_err > max_abs)
        max_abs = abs_err;
      if (rel_err > max_rel)
        max_rel = rel_err;
      if (abs_err > atol && rel_err > rtol)
        ++mismatches;
    }
  }

  std::printf("kernel: %s  K=%llu  k_tiles=%u  block=%u  dyn_smem=%zu\n",
              kernel, (unsigned long long)K, k_tiles, BLOCK,
              v2 ? V2_DYN_SMEM : 0);
  std::printf("max abs err: %.6e   max rel err: %.6e\n", max_abs, max_rel);
  std::printf("mismatches:  %llu / %u\n", (unsigned long long)mismatches,
              M * N);
  std::printf("sum(C):      %.9e\n", sum);

  bool pass = mismatches == 0;
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  if (!pass) {
    // print a few failing samples to help debug
    for (uint32_t m = 0; m < M && m < 4; ++m) {
      for (uint32_t n = 0; n < N && n < 4; ++n) {
        double got = C[m * N + n], ref = C_ref[m * N + n];
        if (std::fabs(got - ref) > atol &&
            std::fabs(got - ref) / std::fmax(1.0, std::fabs(ref)) > rtol)
          std::printf("  C[%u,%u] got %.6f ref %.6f\n", m, n, (float)got,
                      (float)ref);
      }
    }
  }

  CUDA_CHECK(cuMemFree(d_A));
  CUDA_CHECK(cuMemFree(d_B));
  CUDA_CHECK(cuMemFree(d_C));
  CUDA_CHECK(cuModuleUnload(mod));
  CUDA_CHECK(cuCtxDestroy(ctx));
  return pass ? 0 : 1;
}
