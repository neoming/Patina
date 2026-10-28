//! Hardware execution through the CUDA **driver** API (behind `feature = "cuda"`).
//!
//! The CUDA driver API is a plain C ABI (`cuda.h` + `libcuda.so`), so Rust
//! calls it directly via `extern "C"` — no C++, no nvcc, and no NVRTC. Patina
//! emits PTX *source text* (`patina::ffi::lower_to_ptx`); the driver JITs it
//! for the current device inside [`Cuda::load`] / `cuModuleLoadDataEx`.
//!
//! Two layers:
//! * [`Cuda`] — a context handle with the generic plumbing
//!   (init / load / launch / alloc / copy), usable for any PTX kernel.
//! * [`run_gemm`] — the tcgen05 GEMM-specific wrapper (TMA tensor maps,
//!   128x128 f32 accumulator, block 192) used by `tests/hw_run.rs`.
//!
//! Enabled with `cargo test --features cuda` (links `libcuda.so`; requires an
//! NVIDIA driver at build/runtime).

use std::ffi::{c_char, c_void, CStr};

// ---------------------------------------------------------------------------
// driver-API types and constants (mirror cuda.h)
// ---------------------------------------------------------------------------
type CudaResult = i32;
type CudaDevice = i32;
type CudaContext = *mut c_void;
type CudaModule = *mut c_void;
type CudaFunction = *mut c_void;
type CudaStream = *mut c_void;
type CudaDevicePtr = u64;

const CUDA_SUCCESS: CudaResult = 0;

const CU_TENSOR_MAP_DATA_TYPE_FLOAT16: u32 = 6;
const CU_TENSOR_MAP_INTERLEAVE_NONE: u32 = 0;
const CU_TENSOR_MAP_SWIZZLE_NONE: u32 = 0;
const CU_TENSOR_MAP_L2_PROMOTION_L2_128B: u32 = 2;
const CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE: u32 = 0;

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

/// Opaque 128-byte, 64-byte-aligned tensor map (never interpreted on the host).
#[repr(C, align(64))]
pub struct CudaTensorMap([u8; 128]);

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: u32) -> CudaResult;
    fn cuDeviceGet(dev: *mut CudaDevice, ordinal: i32) -> CudaResult;
    fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: CudaDevice) -> CudaResult;
    fn cuCtxCreate(ctx: *mut CudaContext, flags: u32, dev: CudaDevice) -> CudaResult;
    fn cuCtxDestroy(ctx: CudaContext) -> CudaResult;
    fn cuModuleLoadDataEx(
        module: *mut CudaModule,
        image: *const c_void,
        num_options: u32,
        options: *mut i32,
        option_values: *mut *mut c_void,
    ) -> CudaResult;
    fn cuModuleUnload(module: CudaModule) -> CudaResult;
    fn cuModuleGetFunction(
        hfunc: *mut CudaFunction,
        module: CudaModule,
        name: *const c_char,
    ) -> CudaResult;
    #[allow(clippy::too_many_arguments)]
    fn cuLaunchKernel(
        f: CudaFunction,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shared_mem_bytes: u32,
        stream: CudaStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CudaResult;
    fn cuMemAlloc(dptr: *mut CudaDevicePtr, bytes: usize) -> CudaResult;
    fn cuMemFree(dptr: CudaDevicePtr) -> CudaResult;
    fn cuMemcpyHtoD(dst: CudaDevicePtr, src: *const c_void, bytes: usize) -> CudaResult;
    fn cuMemcpyDtoH(dst: *mut c_void, src: CudaDevicePtr, bytes: usize) -> CudaResult;
    fn cuMemsetD8(dst: CudaDevicePtr, uc: u8, n: usize) -> CudaResult;
    fn cuStreamCreate(s: *mut CudaStream, flags: u32) -> CudaResult;
    fn cuStreamDestroy(s: CudaStream) -> CudaResult;
    fn cuStreamSynchronize(s: CudaStream) -> CudaResult;
    #[allow(clippy::too_many_arguments)]
    fn cuTensorMapEncodeTiled(
        tensor_map: *mut CudaTensorMap,
        data_type: u32,
        tensor_rank: u32,
        global_address: *mut c_void,
        global_dim: *const u64,
        global_strides: *const u64,
        box_dim: *const u32,
        element_strides: *const u32,
        interleave: u32,
        swizzle: u32,
        l2_promotion: u32,
        oob_fill: u32,
    ) -> CudaResult;
    fn cuGetErrorName(err: CudaResult, name: *mut *const c_char) -> CudaResult;
    fn cuGetErrorString(err: CudaResult, msg: *mut *const c_char) -> CudaResult;
}

#[derive(Debug, Clone)]
pub struct CudaError(pub String);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CudaError {}

fn error_string(code: CudaResult) -> String {
    let mut name: *const c_char = std::ptr::null();
    let mut msg: *const c_char = std::ptr::null();
    unsafe {
        cuGetErrorName(code, &mut name);
        cuGetErrorString(code, &mut msg);
        let name = if name.is_null() {
            String::new()
        } else {
            CStr::from_ptr(name).to_string_lossy().into_owned()
        };
        let msg = if msg.is_null() {
            String::new()
        } else {
            CStr::from_ptr(msg).to_string_lossy().into_owned()
        };
        format!("{name} ({msg})")
    }
}

fn check(code: CudaResult, what: &str) -> Result<(), CudaError> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError(format!("{what}: {}", error_string(code))))
    }
}

// ---------------------------------------------------------------------------
// f16 <-> f32 (round-to-nearest-even; f16->f32 is exact)
// ---------------------------------------------------------------------------
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e: i32 = -14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp - 15 + 127) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(x: f32) -> u16 {
    let u = x.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xff) as u32;
    let mant = u & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp as i32 - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow -> 0
        }
        let mant = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = sign | ((e as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    h
}

// ---------------------------------------------------------------------------
// TMA tensor map for the UMMA SWIZZLE_NONE K-major layout, parameterized by K
// (see `crates/patina/src/layout.rs`, `UmmaKMajor::tma_view`).
// ---------------------------------------------------------------------------
fn encode_tensor_map(ptr: CudaDevicePtr, k: u64) -> Result<CudaTensorMap, CudaError> {
    let k_tiles = k / 32;
    let gdim = [8u64, 8, 4 * k_tiles, 16];
    let gstrd = [2 * k, 16, 16 * k]; // bytes, dims 1..=3
    let boxd = [8u32, 8, 4, 16]; // one K-tile (32 f16)
    let estrd = [1u32; 4];
    let mut tm = CudaTensorMap([0u8; 128]);
    check(
        unsafe {
            cuTensorMapEncodeTiled(
                &mut tm,
                CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                4,
                ptr as *mut c_void,
                gdim.as_ptr(),
                gstrd.as_ptr(),
                boxd.as_ptr(),
                estrd.as_ptr(),
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        },
        "cuTensorMapEncodeTiled",
    )?;
    Ok(tm)
}

// ---------------------------------------------------------------------------
// Generic driver plumbing
// ---------------------------------------------------------------------------

/// A CUDA context handle. `cuCtxDestroy` on drop reclaims every allocation and
/// module created under it, so callers may early-return without manual cleanup.
pub struct Cuda {
    ctx: CudaContext,
    dev: CudaDevice,
}

/// A JIT-loaded PTX module.
pub struct Module {
    inner: CudaModule,
}

impl Cuda {
    /// Initialize the driver and create a context on device 0.
    pub fn init() -> Result<Cuda, CudaError> {
        check(unsafe { cuInit(0) }, "cuInit")?;
        let mut dev: CudaDevice = 0;
        check(unsafe { cuDeviceGet(&mut dev, 0) }, "cuDeviceGet")?;
        let mut ctx: CudaContext = std::ptr::null_mut();
        check(unsafe { cuCtxCreate(&mut ctx, 0, dev) }, "cuCtxCreate")?;
        Ok(Cuda { ctx, dev })
    }

    /// `(major, minor)` compute capability of the context's device.
    pub fn compute_capability(&self) -> Result<(i32, i32), CudaError> {
        let mut major = 0i32;
        let mut minor = 0i32;
        check(
            unsafe {
                cuDeviceGetAttribute(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    self.dev,
                )
            },
            "cuDeviceGetAttribute(major)",
        )?;
        check(
            unsafe {
                cuDeviceGetAttribute(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    self.dev,
                )
            },
            "cuDeviceGetAttribute(minor)",
        )?;
        Ok((major, minor))
    }

    /// Allocate `bytes` of device global memory.
    pub fn alloc(&self, bytes: usize) -> Result<CudaDevicePtr, CudaError> {
        let mut p: CudaDevicePtr = 0;
        check(unsafe { cuMemAlloc(&mut p, bytes) }, "cuMemAlloc")?;
        Ok(p)
    }

    pub fn free(&self, ptr: CudaDevicePtr) -> Result<(), CudaError> {
        check(unsafe { cuMemFree(ptr) }, "cuMemFree")
    }

    pub fn memset(&self, ptr: CudaDevicePtr, val: u8, bytes: usize) -> Result<(), CudaError> {
        check(unsafe { cuMemsetD8(ptr, val, bytes) }, "cuMemsetD8")
    }

    pub fn memcpy_htod(
        &self,
        dst: CudaDevicePtr,
        src: *const u8,
        bytes: usize,
    ) -> Result<(), CudaError> {
        check(
            unsafe { cuMemcpyHtoD(dst, src as *const c_void, bytes) },
            "cuMemcpyHtoD",
        )
    }

    pub fn memcpy_dtoh(
        &self,
        src: CudaDevicePtr,
        dst: *mut u8,
        bytes: usize,
    ) -> Result<(), CudaError> {
        check(
            unsafe { cuMemcpyDtoH(dst as *mut c_void, src, bytes) },
            "cuMemcpyDtoH",
        )
    }

    /// Load PTX source text and JIT it for the current device.
    pub fn load(&self, ptx: &str) -> Result<Module, CudaError> {
        let ptx_c = std::ffi::CString::new(ptx).map_err(|_| CudaError("PTX has NUL".into()))?;
        let mut module: CudaModule = std::ptr::null_mut();
        check(
            unsafe {
                cuModuleLoadDataEx(
                    &mut module,
                    ptx_c.as_ptr() as *const c_void,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            "cuModuleLoadDataEx",
        )?;
        Ok(Module { inner: module })
    }

    /// Launch `f` and block until it completes.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        f: CudaFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        dyn_smem: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        let mut stream: CudaStream = std::ptr::null_mut();
        check(unsafe { cuStreamCreate(&mut stream, 0) }, "cuStreamCreate")?;
        let r = check(
            unsafe {
                cuLaunchKernel(
                    f,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    dyn_smem,
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        );
        let sync = check(unsafe { cuStreamSynchronize(stream) }, "cuStreamSynchronize");
        let destroy = check(unsafe { cuStreamDestroy(stream) }, "cuStreamDestroy");
        r?;
        sync?;
        destroy
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        unsafe { cuCtxDestroy(self.ctx) };
    }
}

impl Module {
    pub fn function(&self, name: &str) -> Result<CudaFunction, CudaError> {
        let name_c = std::ffi::CString::new(name).map_err(|_| CudaError("NUL in name".into()))?;
        let mut f: CudaFunction = std::ptr::null_mut();
        check(
            unsafe { cuModuleGetFunction(&mut f, self.inner, name_c.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(f)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { cuModuleUnload(self.inner) };
    }
}

/// `(major, minor)` compute capability of device 0 (e.g. `(10, 0)` on a GB200).
///
/// Convenience wrapper: no context is created (device attributes do not need
/// one), so this is cheap enough to use as a "can I run tcgen05?" probe.
pub fn compute_capability() -> Result<(i32, i32), CudaError> {
    check(unsafe { cuInit(0) }, "cuInit")?;
    let mut dev: CudaDevice = 0;
    check(unsafe { cuDeviceGet(&mut dev, 0) }, "cuDeviceGet")?;
    let mut major = 0i32;
    let mut minor = 0i32;
    check(
        unsafe {
            cuDeviceGetAttribute(
                &mut major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            )
        },
        "cuDeviceGetAttribute(major)",
    )?;
    check(
        unsafe {
            cuDeviceGetAttribute(
                &mut minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            )
        },
        "cuDeviceGetAttribute(minor)",
    )?;
    Ok((major, minor))
}

// ---------------------------------------------------------------------------
// tcgen05 GEMM runner
// ---------------------------------------------------------------------------

/// Run a Patina GEMM kernel and return the `C` tile (128x128 f32, row-major).
///
/// * `ptx`    — PTX source (as emitted by `patina::ffi::lower_to_ptx`).
/// * `kernel` — `"tcgen05_gemm_v1"` (K=32) or `"tcgen05_gemm_v2"` (K=k*32).
/// * `k`      — total K dimension (32 for v1; multiple of 32 for v2).
/// * `a`      — `A[128,k]` f32, row-major (`A[m][k]`).
/// * `b`      — `B[128,k]` f32, row-major (`B[n][k]`).
///
/// The inputs are rounded to f16 on the host (matching the kernel's f16
/// operand type) and copied to the device; the TMA tensor maps are built from
/// the same layout derivation as the device descriptors.
pub fn run_gemm(
    ptx: &str,
    kernel: &str,
    k: u64,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, CudaError> {
    const M: u64 = 128;
    const N: u64 = 128;
    const BLOCK: u32 = 192; // load(1) + mma(1) + epi(4) warps
    const V2_DYN_SMEM: u32 = 65536;

    let v2 = match kernel {
        "tcgen05_gemm_v1" => {
            assert_eq!(k, 32, "v1 requires K=32");
            false
        }
        "tcgen05_gemm_v2" => {
            assert_eq!(k % 32, 0, "v2 requires K a multiple of 32");
            true
        }
        other => panic!("unknown kernel {other}"),
    };
    assert_eq!(a.len() as u64, M * k, "A must be 128*k");
    assert_eq!(b.len() as u64, N * k, "B must be 128*k");

    let cuda = Cuda::init()?;
    let (major, minor) = cuda.compute_capability()?;
    if major < 10 {
        return Err(CudaError(format!(
            "device is sm_{major}{minor}; tcgen05 needs sm_100a (Blackwell/GB200)"
        )));
    }

    let module = cuda.load(ptx)?;
    let func = module.function(kernel)?;

    let ab_bytes = (M * k * 2) as usize;
    let c_bytes = (M * N * 4) as usize;
    let d_a = cuda.alloc(ab_bytes)?;
    let d_b = cuda.alloc(ab_bytes)?;
    let mut d_c = cuda.alloc(c_bytes)?;
    cuda.memset(d_c, 0, c_bytes)?;

    let a16: Vec<u16> = a.iter().map(|&x| f32_to_f16(x)).collect();
    let b16: Vec<u16> = b.iter().map(|&x| f32_to_f16(x)).collect();
    cuda.memcpy_htod(d_a, a16.as_ptr() as *const u8, ab_bytes)?;
    cuda.memcpy_htod(d_b, b16.as_ptr() as *const u8, ab_bytes)?;

    let mut tma_a = encode_tensor_map(d_a, k)?;
    let mut tma_b = encode_tensor_map(d_b, k)?;
    let mut k_tiles = (k / 32) as u32;

    let mut params: Vec<*mut c_void> = vec![
        &mut tma_a as *mut CudaTensorMap as *mut c_void,
        &mut tma_b as *mut CudaTensorMap as *mut c_void,
        &mut d_c as *mut CudaDevicePtr as *mut c_void,
    ];
    if v2 {
        params.push(&mut k_tiles as *mut u32 as *mut c_void);
    }

    cuda.launch(func, (1, 1, 1), (BLOCK, 1, 1), if v2 { V2_DYN_SMEM } else { 0 }, &mut params)?;

    let mut c = vec![0f32; (M * N) as usize];
    cuda.memcpy_dtoh(d_c, c.as_mut_ptr() as *mut u8, c_bytes)?;
    Ok(c)
}

/// Host reference: `C[m,n] = sum_k A[m,k] * B[n,k]`, double precision.
pub fn reference_gemm(a: &[f32], b: &[f32], k: u64) -> Vec<f64> {
    let mut c = vec![0f64; 128 * 128];
    for m in 0..128usize {
        for n in 0..128usize {
            let mut acc = 0f64;
            for kk in 0..k as usize {
                acc += f64::from(a[m * k as usize + kk]) * f64::from(b[n * k as usize + kk]);
            }
            c[m * 128 + n] = acc;
        }
    }
    c
}

/// Compare a kernel result against `reference_gemm`; returns
/// `(max_abs_err, max_rel_err, mismatch_count)` with the tolerance used by
/// the harness (`1e-3` absolute and relative).
pub fn compare(c: &[f32], reference: &[f64]) -> (f64, f64, u64) {
    let (atol, rtol) = (1e-3, 1e-3);
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut mismatches = 0u64;
    for (got, &want) in c.iter().zip(reference) {
        let abs_err = (f64::from(*got) - want).abs();
        let rel_err = abs_err / want.abs().max(1.0);
        max_abs = max_abs.max(abs_err);
        max_rel = max_rel.max(rel_err);
        if abs_err > atol && rel_err > rtol {
            mismatches += 1;
        }
    }
    (max_abs, max_rel, mismatches)
}
