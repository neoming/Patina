//! Safe Rust bindings to the patina MLIR dialect C ABI
//! (`libpatina_mlir_c`, built from `mlir/` by this crate's build script).
//!
//! Strings returned by the C API are heap-allocated and freed with
//! `patina_string_free` exactly once, inside these wrappers.

use std::ffi::{c_char, CStr, CString};

enum PatinaContext {}

extern "C" {
    fn patina_context_create() -> *mut PatinaContext;
    fn patina_context_destroy(ctx: *mut PatinaContext);
    fn patina_verify(ctx: *mut PatinaContext, mlir_text: *const c_char) -> *mut c_char;
    fn patina_lower_to_ptx(
        ctx: *mut PatinaContext,
        mlir_text: *const c_char,
        err_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn patina_string_free(s: *mut c_char);
}

/// Owning handle for a `PatinaContext`.
struct Context(*mut PatinaContext);

impl Context {
    fn new() -> Self {
        let ctx = unsafe { patina_context_create() };
        assert!(!ctx.is_null(), "patina_context_create returned null");
        Context(ctx)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { patina_context_destroy(self.0) }
    }
}

/// Take ownership of a C-API string, copy it out, and free it.
unsafe fn take_string(s: *mut c_char) -> String {
    assert!(!s.is_null());
    let owned = CStr::from_ptr(s).to_string_lossy().into_owned();
    patina_string_free(s);
    owned
}

fn c_text(mlir_text: &str) -> Result<CString, String> {
    CString::new(mlir_text).map_err(|_| "MLIR text contains an interior NUL".to_string())
}

/// Parse `mlir_text` as patina-dialect MLIR and run the dialect verifiers.
/// `Ok(())` when the module verifies; `Err(diagnostic)` otherwise.
pub fn verify(mlir_text: &str) -> Result<(), String> {
    let text = c_text(mlir_text)?;
    let ctx = Context::new();
    unsafe {
        let err = patina_verify(ctx.0, text.as_ptr());
        if err.is_null() {
            Ok(())
        } else {
            Err(take_string(err))
        }
    }
}

/// Lower the patina kernel in `mlir_text` to PTX source text.
/// (M2b stub: always returns `Err("patina_lower_to_ptx: unimplemented (M2c)")`.)
pub fn lower_to_ptx(mlir_text: &str) -> Result<String, String> {
    let text = c_text(mlir_text)?;
    let ctx = Context::new();
    unsafe {
        let mut err: *mut c_char = std::ptr::null_mut();
        let ptx = patina_lower_to_ptx(ctx.0, text.as_ptr(), &mut err);
        if ptx.is_null() {
            let msg = if err.is_null() {
                "patina_lower_to_ptx failed (no diagnostic)".to_string()
            } else {
                take_string(err)
            };
            Err(msg)
        } else {
            Ok(take_string(ptx))
        }
    }
}
