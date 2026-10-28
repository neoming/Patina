//! Driver-API plumbing smoke test: runs a trivial (non-tcgen05) PTX kernel on
//! the *current* GPU, so the load→launch→copy→check path is proven to work on
//! any CUDA device — including the local non-Blackwell GPU — before spending
//! GB200 time. The tcgen05 kernels themselves are only exercised by
//! `tests/hw_run.rs` on sm_100a.
//!
//! ```bash
//! cargo test --features cuda --test run_smoke -- --nocapture
//! ```

#![cfg(feature = "cuda")]

use patina::run::Cuda;
use std::ffi::c_void;

/// A minimal `c[i] = a[i] + b[i]` kernel. `.target sm_89` so it runs on the
/// local Ada GPU and forward-compatibly on Blackwell (sm_100a).
const VEC_ADD: &str = r#"
.version 8.7
.target sm_89
.address_size 64

.visible .entry vec_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %tid.x;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra L_ret;
    mul.wide.u32 %rd4, %r2, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
L_ret:
    ret;
}
"#;

#[test]
fn driver_plumbing_runs_a_trivial_kernel() {
    let cuda = match Cuda::init() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    let (maj, min) = cuda.compute_capability().unwrap();
    println!("device: sm_{maj}{min}");

    let n = 256u32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();

    let module = cuda.load(VEC_ADD).unwrap();
    let func = module.function("vec_add").unwrap();

    let bytes = n as usize * 4;
    let d_a = cuda.alloc(bytes).unwrap();
    let d_b = cuda.alloc(bytes).unwrap();
    let d_c = cuda.alloc(bytes).unwrap();
    cuda.memcpy_htod(d_a, a.as_ptr() as *const u8, bytes).unwrap();
    cuda.memcpy_htod(d_b, b.as_ptr() as *const u8, bytes).unwrap();

    let mut params: Vec<*mut c_void> = vec![
        &d_a as *const u64 as *mut c_void,
        &d_b as *const u64 as *mut c_void,
        &d_c as *const u64 as *mut c_void,
        &n as *const u32 as *mut c_void,
    ];
    cuda.launch(func, (1, 1, 1), (n, 1, 1), 0, &mut params).unwrap();

    let mut c = vec![0f32; n as usize];
    cuda.memcpy_dtoh(d_c, c.as_mut_ptr() as *mut u8, bytes).unwrap();

    for i in 0..n as usize {
        let want = a[i] + b[i];
        assert!(
            (c[i] - want).abs() < 1e-6,
            "c[{i}] = {} != {}",
            c[i],
            want
        );
    }
    println!("vec_add OK: {n} elements");
}

#[test]
fn tcgen05_kernels_are_gated_on_sm_100a() {
    let (maj, min) = match patina::run::compute_capability() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no CUDA device ({e})");
            return;
        }
    };
    if maj >= 10 {
        eprintln!("skip: sm_{maj}{min} is Blackwell (covered by hw_run)");
        return;
    }
    // On a pre-Blackwell device, run_gemm must refuse *before* touching the
    // (empty) PTX, with a capability error — not launch garbage.
    let a = vec![0f32; 128 * 32];
    let b = vec![0f32; 128 * 32];
    let err = patina::run::run_gemm("", "tcgen05_gemm_v1", 32, &a, &b).unwrap_err();
    assert!(err.0.contains("sm_100a"), "unexpected error: {err}");
    println!("tcgen05 gate OK: {err}");
}
