//===- PatinaCAPI.h - C ABI for the patina MLIR dialect -------------------===//
//
// Ownership rules:
//   - PatinaContext is owned by the caller: create with
//     patina_context_create, release with patina_context_destroy. Contexts
//     are independent; each call site may create its own.
//   - Every char* returned by this API (including via out-parameters) is
//     heap-allocated and must be released by the caller with
//     patina_string_free. NULL is never an error to free.
//
//===----------------------------------------------------------------------===//

#ifndef PATINA_C_PATINACAPI_H
#define PATINA_C_PATINACAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PatinaContext PatinaContext;

PatinaContext *patina_context_create(void);
void patina_context_destroy(PatinaContext *ctx);

/// Parse `mlir_text` (NUL-terminated patina-dialect MLIR) and run the
/// verifier. Returns NULL on success; on parse/verify failure returns a
/// heap-allocated diagnostic string (caller frees with patina_string_free).
char *patina_verify(PatinaContext *ctx, const char *mlir_text);

/// Lower the patina kernel in `mlir_text` to PTX text.
/// On success returns heap-allocated PTX source and leaves *err_out NULL.
/// On failure returns NULL and sets *err_out to a heap-allocated diagnostic
/// (when err_out itself is non-NULL). The v1 lowering covers the v1
/// warp-specialized kernel form only.
char *patina_lower_to_ptx(PatinaContext *ctx, const char *mlir_text,
                          char **err_out);

/// Free a string returned by this API. NULL-safe.
void patina_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif // PATINA_C_PATINACAPI_H
