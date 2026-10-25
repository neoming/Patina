//===- PatinaOps.td - Patina dialect ops --------------------*- tablegen -*-===//
//
// Ops of the Patina dialect, one level above PTX (PROPOSAL §6). Every op
// names its PTX realization in its description. Ops marked `hasVerifier`
// declare C++ verifier hooks implemented in patina-opt; the exact diagnostic
// strings are the contract in dialect/tests/verifier.mlir — change them only
// together with that file (docs/TESTING.md rules).
//
// Requires an LLVM/MLIR dev tree (mlir-tblgen); M1c build deferred.
//
//===----------------------------------------------------------------------===//

#ifndef PATINA_OPS_TD
#define PATINA_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "PatinaDialect.td"
include "PatinaAttrs.td"
include "PatinaTypes.td"

class Patina_Op<string mnemonic, list<Trait> traits = []>
    : Op<Patina_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Kernel structure
//===----------------------------------------------------------------------===//

def Patina_KernelOp : Patina_Op<"kernel", [IsolatedFromAbove]> {
  let summary = "a warp-specialized kernel: roles, block, arch, version";
  let description = [{
    Lowers to `.entry` + params. `version` is the dialect version ("0.1");
    `arch` is e.g. `#patina.arch<sm_100a>`; `block` is the CTA thread count;
    `roles` is the ordered list of `#patina.role<kind, warps>` attributes
    (warps auto-packed in load/mma/epi order, v1).

    Verifier (role/block warp budget) — the sum of role warp counts x 32
    must equal `block`:
      "roles require 8 warps (256 threads) but block = 128"
  }];
  let arguments = (ins
    SymbolNameAttr : $sym_name,
    StrAttr : $version,
    Patina_ArchAttr : $arch,
    I32Attr : $block,
    ArrayAttr : $roles);
  let regions = (region SizedRegion<1> : $body);
  let hasVerifier = 1;
}

def Patina_WarpSpecializeOp : Patina_Op<"warp_specialize", []> {
  let summary = "the one structured switch(warp_id) over role regions";
  let description = [{
    Holds one `patina.role` region per role of the enclosing kernel. Warps
    are assigned contiguously in role order; the region executes only on the
    warps of its role.
  }];
  let regions = (region SizedRegion<1> : $body);
  let assemblyFormat = "$body attr-dict";
}

def Patina_RoleOp : Patina_Op<"role", []> {
  let summary = "one role region inside patina.warp_specialize";
  let description = [{
    Only legal directly inside `patina.warp_specialize`; `kind` must be one
    of the roles declared by the enclosing `patina.kernel`, at most one
    region per role.
  }];
  let arguments = (ins Patina_RoleKind : $kind);
  let regions = (region SizedRegion<1> : $body);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CTA-scope resources
//===----------------------------------------------------------------------===//

def Patina_SmemAllocOp : Patina_Op<"smem_alloc", []> {
  let summary = "static smem allocation of a typed tensor";
  let description = [{
    Lowers to static smem + `cvta.to.shared`. The result type carries the
    layout, so descriptor derivation exists only where legal.
  }];
  let results = (outs Patina_TensorType : $result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Patina_MBarrierCreateOp : Patina_Op<"mbarrier.create", []> {
  let summary = "mbarrier.init + fence.mbarrier_init, arrival count in type";
  let arguments = (ins I32Attr : $arrivals);
  let results = (outs Patina_MBarrierType : $result);
  let assemblyFormat = "`arrivals` `=` $arrivals attr-dict `:` type($result)";
}

def Patina_TmemAllocOp : Patina_Op<"tmem.alloc", []> {
  let summary = "warp-collective tcgen05.alloc + relinquish_alloc_permit";
  let description = [{
    Runs on one full warp before the role dispatch.

    Verifier (alloc/dealloc pairing) — every result must reach exactly one
    `patina.tmem.dealloc`:
      "tmem.alloc result must be consumed by exactly one patina.tmem.dealloc"
  }];
  let arguments = (ins I32Attr : $cols);
  let results = (outs Patina_TmemType : $result);
  let assemblyFormat = "`cols` `=` $cols attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Patina_TmemDeallocOp : Patina_Op<"tmem.dealloc", []> {
  let summary = "tcgen05.dealloc with the fence pair and bar.sync";
  let description = [{
    Lowers to `tcgen05.fence::before_thread_sync` + `bar.sync` +
    `tcgen05.dealloc` (+ `fence::after_thread_sync`), emitted after the role
    join. RAII in the front-end: dropping the Tmem handle emits this op.
  }];
  let arguments = (ins Patina_TmemType : $acc);
  let assemblyFormat = "$acc attr-dict";
}

//===----------------------------------------------------------------------===//
// Load role
//===----------------------------------------------------------------------===//

def Patina_ExpectTxOp : Patina_Op<"expect_tx", []> {
  let summary = "mbarrier.arrive.expect_tx: publish expected bytes";
  let description = [{
    The byte count is derived in the front-end from the tensors the copies
    cover.

    Verifier (byte match) — `bytes` must equal the sum of the bytes of every
    `patina.tma_load` completing on the same barrier:
      "expect_tx bytes (4096) != sum of tma_load bytes on this barrier (8192)"
  }];
  let arguments = (ins Patina_MBarrierType : $barrier, I64Attr : $bytes);
  let assemblyFormat = "$barrier `,` $bytes attr-dict";
  let hasVerifier = 1;
}

def Patina_TmaLoadOp : Patina_Op<"tma_load", []> {
  let summary = "cp.async.bulk.tensor.4d from a tensor map into smem";
  let arguments = (ins
    Patina_TmaMapType : $map,
    Patina_TensorType : $dst,
    DenseI64ArrayAttr : $coords,
    Patina_MBarrierType : $complete);
  let assemblyFormat =
      "$map `->` $dst `,` `coords` $coords `,` `complete` $complete attr-dict";
}

def Patina_WaitOp : Patina_Op<"wait", []> {
  let summary = "mbarrier.try_wait.parity loop on a typed phase";
  let description = [{
    Phase parity is a tracked attribute, never a raw literal in user code.

    Verifier (phase ordering) — phases of each barrier must be waited in
    order per role region:
      "phase 1 waited before phase 0 on this barrier"
  }];
  let arguments = (ins Patina_MBarrierType : $barrier, I32Attr : $phase);
  let assemblyFormat = "$barrier `,` `phase` `=` $phase attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MMA role
//===----------------------------------------------------------------------===//

def Patina_UmmaBeginOp : Patina_Op<"umma.begin", []> {
  let summary = "begin a UMMA chain: acc + A/B smem operands";
  let description = [{
    The instruction descriptor (f16 in, f32 acc, M/N from the types) is
    packed by lowering from the operand types; the smem descriptors
    (LBO = 128, SBO = 256 for swizzle_none K-major) come from the layout in
    the tensor types.

    Verifier (role legality) — `patina.umma.*` ops are only legal inside an
    `mma` role region (same diagnostic for step/advance_k/commit):
      "'patina.umma.begin' is only legal inside an 'mma' role region"
  }];
  let arguments = (ins
    Patina_TmemType : $acc,
    Patina_TensorType : $a,
    Patina_TensorType : $b);
  let results = (outs Patina_TmemType : $result);
  let assemblyFormat = "$acc `,` $a `,` $b attr-dict";
  let hasVerifier = 1;
}

def Patina_UmmaStepOp : Patina_Op<"umma.step", []> {
  let summary = "one K16 tcgen05.mma; accum=false zeroes the acc input";
  let description = [{
    `accum` maps to the scaleC predicate of `tcgen05.mma` (false on the
    first K16 step). Role-legality verifier as on `patina.umma.begin`.
  }];
  let arguments = (ins Patina_TmemType : $m, BoolAttr : $accum);
  let assemblyFormat = "$m `{` `accum` `=` $accum `}` attr-dict";
  let hasVerifier = 1;
}

def Patina_UmmaAdvanceKOp : Patina_Op<"umma.advance_k", []> {
  let summary = "advance A/B descriptors by K halves";
  let description = [{
    The unit count comes from the layout (`k_advance_units`: one K16 step =
    two 16B core matrices = 256B = 16 descriptor units), never a literal.
    Role-legality verifier as on `patina.umma.begin`.
  }];
  let arguments = (ins Patina_TmemType : $m, I32Attr : $k);
  let assemblyFormat = "$m `,` $k attr-dict";
  let hasVerifier = 1;
}

def Patina_UmmaCommitOp : Patina_Op<"umma.commit", []> {
  let summary = "tcgen05.commit: arrive on an mbarrier when MMAs complete";
  let arguments = (ins Patina_TmemType : $m, Patina_MBarrierType : $barrier);
  let assemblyFormat = "$m `->` $barrier attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Epilogue role
//===----------------------------------------------------------------------===//

def Patina_TmemLdOp : Patina_Op<"tmem_ld", []> {
  let summary = "tcgen05.ld.32x32b.x32 + wait::ld into a register fragment";
  let description = [{
    The TMEM lane field of the address is derived from the warp rank
    (`warp << 21`), never hand-encoded.
  }];
  let arguments = (ins Patina_TmemType : $acc);
  let results = (outs Patina_FragType : $result);
  let assemblyFormat = "$acc attr-dict `:` type($result)";
}

def Patina_StoreOp : Patina_Op<"store", []> {
  let summary = "st.global.v4 of one register fragment, coalesced";
  let arguments = (ins Patina_PtrType : $ptr, Patina_FragType : $frag);
  let assemblyFormat = "$ptr `,` $frag attr-dict";
}

#endif // PATINA_OPS_TD
