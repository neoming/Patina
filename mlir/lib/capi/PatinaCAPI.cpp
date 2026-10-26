//===- PatinaCAPI.cpp - C ABI implementation ------------------------------===//
//
// Ownership: strings cross the boundary malloc'd; patina_string_free frees.
//
//===----------------------------------------------------------------------===//

#include "patina-c/PatinaCAPI.h"

#include "patina/Passes.h"
#include "patina/PatinaDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <string>

struct PatinaContext {
  mlir::MLIRContext ctx;
  PatinaContext() { ctx.getOrLoadDialect<patina::PatinaDialect>(); }
};

static char *dupCString(llvm::StringRef s) {
  char *out = static_cast<char *>(std::malloc(s.size() + 1));
  if (!out)
    return nullptr;
  std::memcpy(out, s.data(), s.size());
  out[s.size()] = '\0';
  return out;
}

extern "C" {

PatinaContext *patina_context_create(void) { return new PatinaContext(); }

void patina_context_destroy(PatinaContext *ctx) { delete ctx; }

char *patina_verify(PatinaContext *ctx, const char *mlir_text) {
  if (!ctx || !mlir_text)
    return dupCString("patina_verify: null argument");

  std::string diags;
  {
    llvm::raw_string_ostream os(diags);
    mlir::ScopedDiagnosticHandler handler(
        &ctx->ctx, [&](mlir::Diagnostic &d) {
          d.print(os);
          os << "\n";
          return mlir::success();
        });

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &ctx->ctx);
    if (module && mlir::succeeded(mlir::verify(module.get())))
      return nullptr;
  }
  if (diags.empty())
    diags = "patina_verify: verification failed (no diagnostic captured)\n";
  return dupCString(diags);
}

char *patina_lower_to_ptx(PatinaContext *ctx, const char *mlir_text,
                          char **err_out) {
  auto failWith = [&](llvm::StringRef msg) -> char * {
    if (err_out)
      *err_out = dupCString(msg);
    return nullptr;
  };
  if (!ctx || !mlir_text)
    return failWith("patina_lower_to_ptx: null argument");

  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    std::string diags;
    llvm::raw_string_ostream os(diags);
    mlir::ScopedDiagnosticHandler handler(
        &ctx->ctx, [&](mlir::Diagnostic &d) {
          d.print(os);
          os << "\n";
          return mlir::success();
        });
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &ctx->ctx);
    if (!module || mlir::failed(mlir::verify(module.get())))
      return failWith(os.str());
  }

  std::string err;
  std::optional<std::string> ptx =
      patina::lowerModuleToPTX(module.get(), err);
  if (!ptx)
    return failWith(err);
  return dupCString(*ptx);
}

void patina_string_free(char *s) { std::free(s); }

} // extern "C"
