//===- LowerToPTX.cpp - patina.kernel -> PTX text (v1 + v2) ---------------===//
//
// Emits PTX source directly (no nvgpu / LLVM IR round-trip).
//
// v1 form: single-warp load and mma roles, an N-warp epilogue, f16 x f16 ->
// f32 UMMA over UmmaKMajor (swizzle_none) smem. Instruction forms follow the
// reference kernel tcgen05_gemm.cu: smem descriptors are
// (addr>>4) | LBO(128)>>4<<16 | SBO(256)>>4<<32 | version 1<<46, the idesc is
// c_format(F32)<<4 | (N>>3)<<17 | (M>>4)<<24, and one K16 step advances the
// descriptors by 16 units (2 core matrices).
//
// v2 form (M3, gated on the presence of patina.pipeline.create): the
// pipelined mainloop. Load and mma roles each get a runtime LOOP over
// k_tiles with loop-carried (stage %s, phase %ph) registers; stage-indexed
// full/empty barrier arrays (base + s*8) and smem tiles (dyn_smem base +
// s*8192); dynamic scaleC via setp/selp on the loop index; a drain commit on
// acc_done after the mma loop. Staged tiles live in DYNAMIC shared memory
// (.extern .shared dyn_smem[]) because 4 stages x 2 tiles x 8 KiB = 64 KiB
// exceeds the 48 KiB static .shared limit; the host passes the byte size at
// launch.
//
//===----------------------------------------------------------------------===//

#include "patina/Passes.h"
#include "patina/PatinaOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <vector>

using namespace mlir;

namespace patina {
namespace {

// (LBO>>4)<<16 | (SBO>>4)<<32 | version(1)<<46 for swizzle_none K-major.
constexpr uint64_t kDescBase =
    ((128ull >> 4) << 16) | ((256ull >> 4) << 32) | (1ull << 46);

struct SmemInfo {
  int64_t m = 0, k = 0;
  unsigned dtypeBits = 0;
  uint64_t bytes = 0; // per tile
  uint32_t stages = 1;
};

struct ParamInfo {
  std::string name;
  bool isTmaMap = false;
  bool isU32 = false;
};

struct Lowerer {
  KernelOp kernel;
  std::string err;

  // CTA scope, in declaration order.
  std::vector<std::string> smemOrder;
  std::map<std::string, SmemInfo> smem;
  std::vector<std::string> barrierOrder;
  std::map<std::string, uint32_t> arrivals;
  std::string tmemName;
  uint32_t tmemCols = 0;
  std::vector<ParamInfo> params;

  // v2 (pipelined) state.
  bool pipelined = false;
  uint32_t pipeStages = 0;
  std::map<std::string, uint64_t> dynOffset; // staged smem -> dyn_smem offset
  int loopCounter = 0;
  int curLoop = -1;

  // Roles: kind -> first warp. v1 packs load / mma / epi contiguously.
  uint32_t loadBase = 0, mmaBase = 0, epiBase = 0, epiWarps = 0;

  int waitLabel = 0;

  std::string out;
  llvm::raw_string_ostream os;

  Lowerer(KernelOp kernel) : kernel(kernel), os(out) {}

  bool fail(const llvm::Twine &msg) {
    err = msg.str();
    return false;
  }

  static std::string nameOf(StringAttr attr) { return attr.getValue().str(); }

  bool gather() {
    // Kernel parameters.
    ArrayAttr argNames = kernel.getArgNames();
    Block &entry = kernel.getBody().front();
    for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
      std::string pname;
      if (i < argNames.size())
        pname = cast<StringAttr>(argNames[i]).getValue().str();
      else
        pname = "arg" + std::to_string(i);
      Type ty = arg.getType();
      if (isa<TmaMapType>(ty))
        params.push_back({pname, true, false});
      else if (isa<PtrType>(ty))
        params.push_back({pname, false, false});
      else if (isa<U32Type>(ty))
        params.push_back({pname, false, true});
      else
        return fail("v1 lowering: unsupported kernel argument type for %" +
                    pname);
    }

    WarpSpecializeOp ws;
    uint64_t dynOff = 0;
    for (Operation &op : entry) {
      if (auto alloc = dyn_cast<SmemAllocOp>(&op)) {
        auto ty = cast<TensorType>(alloc.getResult().getType());
        if (ty.getSpace() != Space::smem)
          return fail("v1 lowering: smem_alloc result must be in smem");
        if (ty.getShape().size() != 2)
          return fail("v1 lowering: smem tiles must be rank-2 [M, K]");
        auto layout = dyn_cast<UmmaKMajorAttr>(ty.getLayout());
        if (!layout || layout.getSwizzle() != Swizzle::swizzle_none)
          return fail("v1 lowering: only UmmaKMajor swizzle_none layouts");
        SmemInfo info;
        info.m = ty.getShape()[0];
        info.k = ty.getShape()[1];
        info.dtypeBits = ty.getDtype().getIntOrFloatBitWidth();
        info.bytes =
            uint64_t(info.m) * uint64_t(info.k) * info.dtypeBits / 8;
        if (auto stages = alloc.getStagesAttr())
          info.stages = static_cast<uint32_t>(stages.getInt());
        std::string name = nameOf(alloc.getValueNameAttr());
        smem[name] = info;
        smemOrder.push_back(name);
        if (info.stages > 1) {
          if ((info.bytes & (info.bytes - 1)) != 0)
            return fail("v2 lowering: staged tile bytes must be a power of 2");
          dynOffset[name] = dynOff;
          dynOff += info.bytes * info.stages;
        }
      } else if (auto pipe = dyn_cast<PipelineCreateOp>(&op)) {
        if (pipelined)
          return fail("v2 lowering: a single pipeline.create is supported");
        pipelined = true;
        pipeStages = pipe.getStages();
        if (pipeStages < 2)
          return fail("v2 lowering: pipeline stages must be >= 2");
      } else if (auto bar = dyn_cast<MBarrierCreateOp>(&op)) {
        std::string name = nameOf(bar.getValueNameAttr());
        barrierOrder.push_back(name);
        arrivals[name] = bar.getArrivals();
      } else if (auto talloc = dyn_cast<TmemAllocOp>(&op)) {
        if (!tmemName.empty())
          return fail("v1 lowering: a single tmem.alloc is supported");
        tmemName = nameOf(talloc.getValueNameAttr());
        tmemCols = talloc.getCols();
      } else if (auto dealloc = dyn_cast<TmemDeallocOp>(&op)) {
        if (nameOf(dealloc.getAccAttr()) != tmemName)
          return fail("v1 lowering: tmem.dealloc must target the tmem.alloc");
      } else if (auto w = dyn_cast<WarpSpecializeOp>(&op)) {
        if (ws)
          return fail("v1 lowering: a single warp_specialize is supported");
        ws = w;
      } else {
        return fail("v1 lowering: unsupported CTA-scope op '" +
                    op.getName().getStringRef().str() + "'");
      }
    }
    if (tmemName.empty())
      return fail("v1 lowering: missing tmem.alloc");
    if (!ws)
      return fail("v1 lowering: missing warp_specialize");

    // Role warp packing from the kernel's roles attribute.
    uint32_t base = 0;
    bool seenLoad = false, seenMma = false, seenEpi = false;
    for (Attribute attr : kernel.getRoles()) {
      auto role = cast<RoleAttr>(attr);
      switch (role.getKind()) {
      case RoleKind::load:
        if (role.getWarps() != 1)
          return fail("v1 lowering: load role must be a single warp");
        loadBase = base;
        seenLoad = true;
        break;
      case RoleKind::mma:
        if (role.getWarps() != 1)
          return fail("v1 lowering: mma role must be a single warp");
        mmaBase = base;
        seenMma = true;
        break;
      case RoleKind::epi:
        epiBase = base;
        epiWarps = role.getWarps();
        seenEpi = true;
        break;
      }
      base += role.getWarps();
    }
    if (!seenLoad || !seenMma || !seenEpi)
      return fail("v1 lowering: roles must include load, mma and epi");
    if (loadBase != 0 || mmaBase <= loadBase || epiBase <= mmaBase)
      return fail("v1 lowering: roles must be packed load, mma, epi in order");
    return true;
  }

  //--------------------------------------------------------------------------
  // Shared instruction fragments
  //--------------------------------------------------------------------------

  void emitBarrierAddr(const std::string &bar) {
    os << "    mov.u32 %bar_" << bar << ", " << bar << "_bar;\n";
  }

  void emitWait(const std::string &bar, uint32_t phase) {
    int n = waitLabel++;
    emitBarrierAddr(bar);
    os << "    {\n"
       << "    .reg .pred P;\n"
       << "WAIT_" << n << ":\n"
       << "    mbarrier.try_wait.parity.shared::cta.b64 P, [%bar_" << bar
       << "], " << phase << ";\n"
       << "    @!P bra WAIT_" << n << ";\n"
       << "    }\n";
  }

  void fragList(int start, int count) {
    for (int i = 0; i < count; ++i) {
      if (i)
        os << ",";
      os << "%f" << (start + i);
    }
  }

  //--------------------------------------------------------------------------
  // Sections
  //--------------------------------------------------------------------------

  void emitHeader() {
    os << ".version 8.7\n"
       << ".target " << stringifyArch(kernel.getArch().getValue()) << "\n"
       << ".address_size 64\n\n";

    if (pipelined) {
      uint64_t dynBytes = 0;
      for (const std::string &name : smemOrder)
        if (smem[name].stages > 1)
          dynBytes += smem[name].bytes * smem[name].stages;
      os << "// Dynamic shared memory: staged smem ring, " << dynBytes
         << " B at launch (exceeds the 48 KiB static .shared limit)\n"
         << ".extern .shared .b8 dyn_smem[];\n\n";
    }

    os << ".visible .entry " << kernel.getSymName() << "(\n";
    for (size_t i = 0; i < params.size(); ++i) {
      const ParamInfo &p = params[i];
      os << "    .param ";
      if (p.isTmaMap)
        os << ".align 64 .b8 " << p.name << "[128]";
      else if (p.isU32)
        os << ".u32 " << p.name;
      else
        os << ".u64 " << p.name;
      if (i + 1 < params.size())
        os << ",";
      os << "\n";
    }
    os << ")\n";
    os << ".maxntid " << kernel.getBlock() << ", 1, 1\n";
    os << "{\n";

    os << "    .reg .pred %p<" << (pipelined ? 8 : 3) << ">;\n";
    os << "    .reg .b32 %tidx, %warpid, %laneid, %ewarp;\n";
    os << "    .reg .b32 ";
    for (size_t i = 0; i < barrierOrder.size(); ++i)
      os << (i ? ", " : "") << "%bar_" << barrierOrder[i];
    os << ", %cols, %tx;\n";
    os << "    .reg .b32 %tmem, %tmem_w, %lane_off, %tmem_addr, %smem_addr;\n";
    os << "    .reg .b32 %idesc, %scale_c, %zero, %row;\n";
    os << "    .reg .b64 ";
    bool first = true;
    for (size_t i = 0; i < params.size(); ++i) {
      if (params[i].isU32)
        continue;
      os << (first ? "" : ", ") << "%" << params[i].name << "_ptr";
      first = false;
    }
    os << ";\n";
    os << "    .reg .b64 %da, %db, %off, %row_base, %st_base, %st_addr;\n";
    os << "    .reg .f32 %f<" << fragElems() << ">;\n";
    if (pipelined)
      os << "    .reg .b32 %kt, %i, %s, %ph, %i4, %cont;\n"
         << "    .reg .b32 %bar_base, %bar_s, %tmp, %sa_s, %sb_s;\n";
    os << "\n";

    os << "    // CTA scope: smem tiles, barriers, and the TMEM column slot\n";
    if (pipelined) {
      os << "    .shared .align 8 .b64 full_bar[" << pipeStages << "];\n"
         << "    .shared .align 8 .b64 empty_bar[" << pipeStages << "];\n";
    } else {
      for (const std::string &name : smemOrder)
        os << "    .shared .align 1024 .b8 " << name << "_smem["
           << smem[name].bytes << "];\n";
    }
    for (const std::string &name : barrierOrder)
      os << "    .shared .align 8 .b64 " << name << "_bar;\n";
    os << "    .shared .align 4 .b32 tmem_base;\n\n";
  }

  uint32_t fragElems() {
    uint32_t elems = 0;
    kernel.walk([&](TmemLdOp ld) {
      elems = std::max(elems, cast<FragType>(ld.getResult().getType()).getElems());
    });
    return elems;
  }

  void emitPrologue() {
    os << "    mov.u32 %tidx, %tid.x;\n"
       << "    shr.u32 %warpid, %tidx, 5;\n"
       << "    and.b32 %laneid, %tidx, 31;\n\n";

    os << "    // barriers: thread 0\n";
    os << "    setp.ne.s32 %p1, %tidx, 0;\n"
       << "    @%p1 bra L_no_init;\n";
    if (pipelined) {
      // full[s] / empty[s] arrays: unrolled, 8-byte stride.
      for (const char *arr : {"full_bar", "empty_bar"}) {
        os << "    mov.u32 %bar_base, " << arr << ";\n";
        os << "    mbarrier.init.shared::cta.b64 [%bar_base], 1;\n";
        for (uint32_t j = 1; j < pipeStages; ++j) {
          os << "    add.s32 %bar_s, "
             << (j == 1 ? "%bar_base" : "%bar_s") << ", 8;\n";
          os << "    mbarrier.init.shared::cta.b64 [%bar_s], 1;\n";
        }
      }
    }
    for (const std::string &name : barrierOrder) {
      emitBarrierAddr(name);
      os << "    mbarrier.init.shared::cta.b64 [%bar_" << name << "], "
         << arrivals[name] << ";\n";
    }
    os << "    fence.mbarrier_init.release.cluster;\n"
       << "L_no_init:\n\n";

    os << "    // tmem columns: warp " << loadBase
       << " (tcgen05.alloc is warp-collective)\n";
    os << "    setp.ne.s32 %p2, %warpid, " << loadBase << ";\n"
       << "    @%p2 bra L_no_alloc;\n"
       << "    mov.u32 %tmem_addr, tmem_base;\n"
       << "    mov.b32 %cols, " << tmemCols << ";\n"
       << "    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
          "[%tmem_addr], %cols;\n"
       << "    tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n"
       << "L_no_alloc:\n"
       << "    bar.sync 0;\n\n";

    os << "    // role dispatch: warp " << loadBase << " = load, warp "
       << mmaBase << " = mma, warps " << epiBase << ".."
       << (epiBase + epiWarps - 1) << " = epi\n";
    os << "    setp.eq.s32 %p1, %warpid, " << loadBase << ";\n"
       << "    @%p1 bra L_load;\n"
       << "    setp.eq.s32 %p2, %warpid, " << mmaBase << ";\n"
       << "    @%p2 bra L_mma;\n"
       << "    bra L_epi;\n";
  }

  bool emitLoadRole(RoleOp role) {
    os << "\n    // ---- load role: lane 0 of warp " << loadBase
       << " issues expect_tx + the TMA copies\n";
    os << "L_load:\n"
       << "    setp.ne.s32 %p1, %laneid, 0;\n"
       << "    @%p1 bra L_join;\n";
    bool zeroEmitted = false;
    for (Operation &op : role.getBody().front()) {
      if (auto tx = dyn_cast<ExpectTxOp>(&op)) {
        std::string bar = nameOf(tx.getBarrierAttr());
        emitBarrierAddr(bar);
        os << "    mov.b32 %tx, " << tx.getBytes() << ";\n"
           << "    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar_" << bar
           << "], %tx;\n";
      } else if (auto load = dyn_cast<TmaLoadOp>(&op)) {
        std::string map = nameOf(load.getMapAttr());
        std::string dst = nameOf(load.getDstAttr());
        std::string bar = nameOf(load.getCompleteAttr());
        for (Attribute c : load.getCoords()) {
          if (isa<StringAttr>(c))
            return fail("v1 lowering: runtime TMA coords need the pipelined "
                        "(M3) lowering");
          if (cast<IntegerAttr>(c).getInt() != 0)
            return fail("v1 lowering: only zero TMA coords are supported");
        }
        if (!smem.count(dst))
          return fail("v1 lowering: tma_load dst is not a smem_alloc");
        if (!zeroEmitted) {
          os << "    mov.b32 %zero, 0;\n";
          zeroEmitted = true;
        }
        os << "    mov.b64 %" << map << "_ptr, " << map << ";\n"
           << "    cvta.param.u64 %" << map << "_ptr, %" << map << "_ptr;\n"
           << "    mov.u32 %smem_addr, " << dst << "_smem;\n"
           << "    "
              "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::"
              "complete_tx::bytes [%smem_addr], [%"
           << map << "_ptr, {%zero, %zero, %zero, %zero}], [%bar_" << bar
           << "];\n";
      } else {
        return fail("v1 lowering: unsupported load-role op '" +
                    op.getName().getStringRef().str() + "'");
      }
    }
    os << "    bra L_join;\n";
    return true;
  }

  bool emitMmaRole(RoleOp role) {
    os << "\n    // ---- mma role: whole warp waits, lane 0 drives the UMMA "
          "chain\n";
    os << "L_mma:\n";
    bool guarded = false;
    for (Operation &op : role.getBody().front()) {
      if (auto wait = dyn_cast<WaitOp>(&op)) {
        emitWait(nameOf(wait.getBarrierAttr()), wait.getPhase());
        continue;
      }
      if (!guarded) {
        os << "    setp.ne.s32 %p1, %laneid, 0;\n"
           << "    @%p1 bra L_join;\n"
           << "    ld.shared.u32 %tmem, [tmem_base];\n";
        guarded = true;
      }
      if (auto begin = dyn_cast<UmmaBeginOp>(&op)) {
        if (nameOf(begin.getAccAttr()) != tmemName)
          return fail("v1 lowering: umma.begin acc is not the tmem.alloc");
        const SmemInfo &a = smem[nameOf(begin.getAAttr())];
        const SmemInfo &b = smem[nameOf(begin.getBAttr())];
        if (a.dtypeBits != 16 || b.dtypeBits != 16)
          return fail("v1 lowering: umma operands must be f16");
        os << "    // smem descriptors: (addr >> 4) | LBO(128)>>4<<16 | "
              "SBO(256)>>4<<32 | version 1<<46\n";
        for (auto [operand, reg] :
             {std::pair{nameOf(begin.getAAttr()), "da"},
              {nameOf(begin.getBAttr()), "db"}}) {
          os << "    mov.u32 %smem_addr, " << operand << "_smem;\n"
             << "    shr.u32 %lane_off, %smem_addr, 4;\n"
             << "    cvt.u64.u32 %" << reg << ", %lane_off;\n"
             << "    or.b64 %" << reg << ", %" << reg << ", " << kDescBase
             << ";\n";
        }
        uint32_t idesc =
            (1u << 4) | (uint32_t(b.m >> 3) << 17) | (uint32_t(a.m >> 4) << 24);
        os << "    // idesc: f16 in, f32 acc, M=" << a.m << ", N=" << b.m
           << ", K-major\n"
           << "    mov.b32 %idesc, " << idesc << ";\n"
           << "    mov.b32 %zero, 0;\n";
      } else if (auto step = dyn_cast<UmmaStepOp>(&op)) {
        os << "    mov.b32 %scale_c, " << (step.getAccum() ? 1 : 0) << ";\n"
           << "    {\n"
           << "    .reg .pred SC;\n"
           << "    setp.ne.b32 SC, %scale_c, 0;\n"
           << "    tcgen05.mma.cta_group::1.kind::f16 [%tmem], %da, %db, "
              "%idesc, {%zero,%zero,%zero,%zero}, SC;\n"
           << "    }\n";
      } else if (auto adv = dyn_cast<UmmaAdvanceKOp>(&op)) {
        os << "    // K advances between steps: 16 halves = 2 core matrices = "
              "16 desc units\n"
           << "    add.s64 %da, %da, " << adv.getK() << ";\n"
           << "    add.s64 %db, %db, " << adv.getK() << ";\n";
      } else if (auto commit = dyn_cast<UmmaCommitOp>(&op)) {
        std::string bar = nameOf(commit.getBarrierAttr());
        emitBarrierAddr(bar);
        os << "    "
              "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
              "cluster.b64 [%bar_"
           << bar << "];\n";
      } else {
        return fail("v1 lowering: unsupported mma-role op '" +
                    op.getName().getStringRef().str() + "'");
      }
    }
    os << "    bra L_join;\n";
    return true;
  }

  //--------------------------------------------------------------------------
  // v2 (pipelined mainloop) instruction fragments
  //--------------------------------------------------------------------------

  // Stage-indexed barrier address: %bar_s = <kind>_bar + %s * 8.
  void emitStageBarrierAddr(StringRef kind) {
    os << "    mov.u32 %bar_base, " << (kind == "full" ? "full_bar" : "empty_bar")
       << ";\n"
       << "    shl.b32 %tmp, %s, 3;\n"
       << "    add.s32 %bar_s, %bar_base, %tmp;\n";
  }

  // Warp-wide wait on <kind>[s] with the loop-carried phase register %ph.
  void emitStageWait(StringRef kind) {
    emitStageBarrierAddr(kind);
    os << "    {\n"
       << "    .reg .pred P;\n"
       << "LOOP_" << curLoop << "_WAIT:\n"
       << "    mbarrier.try_wait.parity.shared::cta.b64 P, [%bar_s], %ph;\n"
       << "    @!P bra LOOP_" << curLoop << "_WAIT;\n"
       << "    }\n";
  }

  // Lane-0 guard skipping to the loop's stage-advance point.
  void emitLoopLaneGuard() {
    os << "    setp.ne.s32 %p2, %laneid, 0;\n"
       << "    @%p2 bra LOOP_" << curLoop << "_NEXT;\n";
  }

  // Loop preamble: bound load, loop-carried inits, condition check.
  bool emitForBegin(ForOp forOp) {
    curLoop = loopCounter++;
    // The bound must be a !patina.u32 kernel parameter.
    const ParamInfo *bound = nullptr;
    for (const ParamInfo &p : params)
      if (p.isU32 && p.name == forOp.getBound())
        bound = &p;
    if (!bound)
      return fail("v2 lowering: patina.for bound %" + forOp.getBound().str() +
                  " is not a !patina.u32 kernel parameter");
    os << "    ld.param.u32 %kt, [" << bound->name << "];\n";
    os << "    mov.b32 %" << forOp.getIndexName() << ", 0;\n";
    ArrayAttr names = forOp.getIterNames();
    auto inits = forOp.getIterInits();
    for (size_t j = 0; j < names.size(); ++j)
      os << "    mov.b32 %" << cast<StringAttr>(names[j]).getValue() << ", "
         << inits[j] << ";\n";
    os << "LOOP_" << curLoop << ":\n"
       << "    setp.ge.u32 %p1, %i, %kt;\n"
       << "    @%p1 bra LOOP_" << curLoop << "_END;\n";
    return true;
  }

  void emitForEnd() {
    os << "    add.s32 %i, %i, 1;\n"
       << "    bra LOOP_" << curLoop << ";\n"
       << "LOOP_" << curLoop << "_END:\n";
    curLoop = -1;
  }

  // %res = %base + %s * tile_bytes (tile bytes are a power of two).
  bool emitStageBuf(StageBufOp sb) {
    std::string base = nameOf(sb.getBaseAttr());
    auto it = dynOffset.find(base);
    if (it == dynOffset.end())
      return fail("v2 lowering: stage_buf base %" + base +
                  " is not a staged smem_alloc");
    uint64_t shift = llvm::countr_zero(smem[base].bytes);
    os << "    mov.u32 %smem_addr, dyn_smem;\n";
    if (it->second)
      os << "    add.s32 %smem_addr, %smem_addr, " << it->second << ";\n";
    os << "    shl.b32 %tmp, %s, " << shift << ";\n"
       << "    add.s32 %" << sb.getValueName() << ", %smem_addr, %tmp;\n";
    return true;
  }

  bool emitMuli(MuliOp mul) {
    int64_t k = mul.getK();
    std::string res = "%" + mul.getValueName().str();
    std::string x = "%" + nameOf(mul.getXAttr());
    if (k > 0 && (k & (k - 1)) == 0)
      os << "    shl.b32 " << res << ", " << x << ", "
         << llvm::countr_zero(static_cast<uint64_t>(k)) << ";\n";
    else
      os << "    mul.lo.u32 " << res << ", " << x << ", " << k << ";\n";
    return true;
  }

  void emitNez(NezOp nez) {
    os << "    setp.ne.u32 %p4, %" << nameOf(nez.getXAttr()) << ", 0;\n"
       << "    selp.b32 %" << nez.getValueName() << ", 1, 0, %p4;\n";
  }

  // Ring advance: %s = (%s + 1) mod S, phase flips on wrap.
  void emitStageNext() {
    os << "LOOP_" << curLoop << "_NEXT:\n"
       << "    add.s32 %s, %s, 1;\n"
       << "    setp.eq.u32 %p3, %s, " << pipeStages << ";\n"
       << "    @%p3 mov.b32 %s, 0;\n"
       << "    @%p3 xor.b32 %ph, %ph, 1;\n";
  }

  bool emitTmaLoadV2(TmaLoadOp load, bool &zeroEmitted) {
    std::string map = nameOf(load.getMapAttr());
    std::string dst = nameOf(load.getDstAttr());
    // Coordinate text: zero constants via %zero, runtime scalars by name.
    SmallVector<std::string> coords;
    for (Attribute c : load.getCoords()) {
      if (auto name = dyn_cast<StringAttr>(c)) {
        coords.push_back("%" + name.getValue().str());
      } else {
        int64_t v = cast<IntegerAttr>(c).getInt();
        if (v != 0)
          return fail("v2 lowering: TMA coords must be 0 or runtime scalars");
        coords.push_back("%zero");
      }
    }
    auto kind = load.getCompleteKindAttr();
    if (!kind)
      return fail("v2 lowering: pipelined tma_load needs a staged complete");
    if (!zeroEmitted) {
      os << "    mov.b32 %zero, 0;\n";
      zeroEmitted = true;
    }
    os << "    mov.b64 %" << map << "_ptr, " << map << ";\n"
       << "    cvta.param.u64 %" << map << "_ptr, %" << map << "_ptr;\n";
    emitStageBarrierAddr(kind.getValue());
    os << "    "
          "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::"
          "complete_tx::bytes [%"
       << dst << "], [%" << map << "_ptr, {";
    llvm::interleaveComma(coords, os);
    os << "}], [%bar_s];\n";
    return true;
  }

  // smem descriptor from a stage_buf register (v1 moves the tile symbol
  // first; here the stage address is already in %<name>).
  void emitStageDesc(const std::string &operand, const char *reg) {
    os << "    shr.u32 %lane_off, %" << operand << ", 4;\n"
       << "    cvt.u64.u32 %" << reg << ", %lane_off;\n"
       << "    or.b64 %" << reg << ", %" << reg << ", " << kDescBase << ";\n";
  }

  bool emitUmmaBeginV2(UmmaBeginOp begin) {
    if (nameOf(begin.getAccAttr()) != tmemName)
      return fail("v2 lowering: umma.begin acc is not the tmem.alloc");
    auto aTy = dyn_cast_or_null<TensorType>(
        resolveValueType(kernel, nameOf(begin.getAAttr())));
    auto bTy = dyn_cast_or_null<TensorType>(
        resolveValueType(kernel, nameOf(begin.getBAttr())));
    if (!aTy || !bTy)
      return fail("v2 lowering: cannot resolve umma operand types");
    if (aTy.getDtype().getIntOrFloatBitWidth() != 16 ||
        bTy.getDtype().getIntOrFloatBitWidth() != 16)
      return fail("v2 lowering: umma operands must be f16");
    int64_t m = aTy.getShape()[0], n = bTy.getShape()[0];
    os << "    ld.shared.u32 %tmem, [tmem_base];\n";
    os << "    // smem descriptors: (addr >> 4) | LBO(128)>>4<<16 | "
          "SBO(256)>>4<<32 | version 1<<46\n";
    emitStageDesc(nameOf(begin.getAAttr()), "da");
    emitStageDesc(nameOf(begin.getBAttr()), "db");
    uint32_t idesc =
        (1u << 4) | (uint32_t(n >> 3) << 17) | (uint32_t(m >> 4) << 24);
    os << "    // idesc: f16 in, f32 acc, M=" << m << ", N=" << n
       << ", K-major\n"
       << "    mov.b32 %idesc, " << idesc << ";\n"
       << "    mov.b32 %zero, 0;\n";
    return true;
  }

  void emitUmmaStepV2(UmmaStepOp step) {
    if (auto ref = step.getAccumRefAttr(); ref && !ref.getValue().empty())
      os << "    mov.b32 %scale_c, %" << ref.getValue() << ";\n";
    else
      os << "    mov.b32 %scale_c, " << (step.getAccum() ? 1 : 0) << ";\n";
    os << "    {\n"
       << "    .reg .pred SC;\n"
       << "    setp.ne.b32 SC, %scale_c, 0;\n"
       << "    tcgen05.mma.cta_group::1.kind::f16 [%tmem], %da, %db, "
          "%idesc, {%zero,%zero,%zero,%zero}, SC;\n"
       << "    }\n";
  }

  void emitUmmaCommitV2(UmmaCommitOp commit) {
    if (auto kind = commit.getBarrierKindAttr()) {
      emitStageBarrierAddr(kind.getValue());
      os << "    "
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
            "cluster.b64 [%bar_s];\n";
      return;
    }
    std::string bar = nameOf(commit.getBarrierAttr());
    emitBarrierAddr(bar);
    os << "    "
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
          "cluster.b64 [%bar_"
       << bar << "];\n";
  }

  // The shared v2 loop-body walker: emits the patina.for skeleton and
  // dispatches each body op. `loadRole` selects the accepted op set.
  bool emitPipeLoop(ForOp forOp, bool loadRole, bool &zeroEmitted) {
    if (!emitForBegin(forOp))
      return false;
    int loop = curLoop;
    bool guarded = false;
    for (Operation &op : forOp.getBody().front()) {
      if (auto w = dyn_cast<WaitStageOp>(&op)) {
        emitStageWait(w.getKind());
        if (!guarded) {
          emitLoopLaneGuard();
          guarded = true;
        }
      } else if (auto tx = dyn_cast<ExpectTxStageOp>(&op)) {
        if (!loadRole)
          return fail("v2 lowering: expect_tx_stage is load-role only");
        emitStageBarrierAddr(tx.getKind());
        os << "    mov.b32 %tx, " << tx.getBytes() << ";\n"
           << "    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar_s], "
              "%tx;\n";
      } else if (auto sb = dyn_cast<StageBufOp>(&op)) {
        if (!emitStageBuf(sb))
          return false;
      } else if (auto mul = dyn_cast<MuliOp>(&op)) {
        if (!emitMuli(mul))
          return false;
      } else if (auto nez = dyn_cast<NezOp>(&op)) {
        emitNez(nez);
      } else if (auto load = dyn_cast<TmaLoadOp>(&op)) {
        if (!loadRole)
          return fail("v2 lowering: tma_load is load-role only");
        if (!emitTmaLoadV2(load, zeroEmitted))
          return false;
      } else if (auto begin = dyn_cast<UmmaBeginOp>(&op)) {
        if (loadRole)
          return fail("v2 lowering: umma.begin is mma-role only");
        if (!emitUmmaBeginV2(begin))
          return false;
      } else if (auto step = dyn_cast<UmmaStepOp>(&op)) {
        emitUmmaStepV2(step);
      } else if (auto adv = dyn_cast<UmmaAdvanceKOp>(&op)) {
        os << "    // K advances between steps: 16 halves = 2 core matrices "
              "= 16 desc units\n"
           << "    add.s64 %da, %da, " << adv.getK() << ";\n"
           << "    add.s64 %db, %db, " << adv.getK() << ";\n";
      } else if (auto commit = dyn_cast<UmmaCommitOp>(&op)) {
        emitUmmaCommitV2(commit);
      } else if (auto next = dyn_cast<StageNextOp>(&op)) {
        (void)next;
        emitStageNext();
      } else if (isa<YieldOp>(&op)) {
        // The for skeleton owns the increment and backedge.
      } else {
        return fail("v2 lowering: unsupported loop-body op '" +
                    op.getName().getStringRef().str() + "'");
      }
      (void)loop;
    }
    emitForEnd();
    return true;
  }

  bool emitLoadRoleV2(RoleOp role) {
    os << "\n    // ---- load role: producer loop, one K tile of A/B per "
          "stage\n";
    os << "L_load:\n";
    bool zeroEmitted = false;
    for (Operation &op : role.getBody().front()) {
      if (auto forOp = dyn_cast<ForOp>(&op)) {
        if (!emitPipeLoop(forOp, /*loadRole=*/true, zeroEmitted))
          return false;
      } else {
        return fail("v2 lowering: unsupported load-role op '" +
                    op.getName().getStringRef().str() + "'");
      }
    }
    os << "    bra L_join;\n";
    return true;
  }

  bool emitMmaRoleV2(RoleOp role) {
    os << "\n    // ---- mma role: consumer loop, K16 steps per tile into "
          "the TMEM acc\n";
    os << "L_mma:\n";
    bool zeroEmitted = false;
    bool drainGuarded = false;
    for (Operation &op : role.getBody().front()) {
      if (auto forOp = dyn_cast<ForOp>(&op)) {
        if (!emitPipeLoop(forOp, /*loadRole=*/false, zeroEmitted))
          return false;
      } else if (auto commit = dyn_cast<UmmaCommitOp>(&op)) {
        // Drain commit after the loop (acc_done): lane 0 only.
        if (!drainGuarded) {
          os << "    setp.ne.s32 %p2, %laneid, 0;\n"
             << "    @%p2 bra L_join;\n";
          drainGuarded = true;
        }
        emitUmmaCommitV2(commit);
      } else {
        return fail("v2 lowering: unsupported mma-role op '" +
                    op.getName().getStringRef().str() + "'");
      }
    }
    os << "    bra L_join;\n";
    return true;
  }

  bool emitEpiRole(RoleOp role) {
    os << "\n    // ---- epi role: warps " << epiBase << ".."
       << (epiBase + epiWarps - 1) << ", all lanes\n";
    os << "L_epi:\n";
    const ParamInfo *ptr = nullptr;
    for (const ParamInfo &p : params)
      if (!p.isTmaMap && !p.isU32) {
        if (ptr)
          return fail("v1 lowering: a single gmem pointer parameter");
        ptr = &p;
      }
    if (!ptr)
      return fail("v1 lowering: no gmem pointer parameter for the epilogue");

    bool prologue = false;
    uint32_t pendingElems = 0;
    for (Operation &op : role.getBody().front()) {
      if (auto wait = dyn_cast<WaitOp>(&op)) {
        emitWait(nameOf(wait.getBarrierAttr()), wait.getPhase());
      } else if (auto ld = dyn_cast<TmemLdOp>(&op)) {
        if (nameOf(ld.getAccAttr()) != tmemName)
          return fail("v1 lowering: tmem_ld acc is not the tmem.alloc");
        auto fragTy = cast<FragType>(ld.getResult().getType());
        if (!fragTy.getDtype().isF32())
          return fail("v1 lowering: fragments must be f32");
        pendingElems = fragTy.getElems();
        if (pendingElems % 32 != 0)
          return fail("v1 lowering: fragment size must be a multiple of 32");
        if (!prologue) {
          uint32_t pitch = pendingElems * 4;
          os << "    // per-warp TMEM lane window: tmem + ((warp - " << epiBase
             << ") << 21)\n"
             << "    ld.shared.u32 %tmem, [tmem_base];\n"
             << "    sub.s32 %ewarp, %warpid, " << epiBase << ";\n"
             << "    shl.b32 %lane_off, %ewarp, 21;\n"
             << "    add.s32 %tmem_w, %tmem, %lane_off;\n"
             << "    // row this thread stores: tid - " << (epiBase * 32)
             << "; row pitch = " << pendingElems << " f32 = " << pitch
             << " B\n"
             << "    sub.s32 %row, %tidx, " << (epiBase * 32) << ";\n"
             << "    ld.param.u64 %" << ptr->name << "_ptr, [" << ptr->name
             << "];\n"
             << "    mul.wide.u32 %off, %row, " << pitch << ";\n"
             << "    add.s64 %row_base, %" << ptr->name << "_ptr, %off;\n\n";
          prologue = true;
        }
      } else if (auto store = dyn_cast<StoreOp>(&op)) {
        if (nameOf(store.getPtrAttr()) != ptr->name)
          return fail("v1 lowering: store pointer is not the gmem parameter");
        if (pendingElems == 0)
          return fail("v1 lowering: store without a preceding tmem_ld");
        emitStoreChunks(pendingElems);
        pendingElems = 0;
      } else {
        return fail("v1 lowering: unsupported epi-role op '" +
                    op.getName().getStringRef().str() + "'");
      }
    }
    return true;
  }

  // One 32-wide tcgen05.ld chunk plus its 8 coalesced st.global.v4 stores.
  void emitStoreChunks(uint32_t elems) {
    uint32_t chunks = elems / 32;
    for (uint32_t j = 0; j < chunks; ++j) {
      const char *addr = "%tmem_w";
      if (j > 0) {
        os << "    add.s32 %tmem_addr, %tmem_w, " << (j * 32) << ";\n";
        addr = "%tmem_addr";
      }
      os << "    tcgen05.ld.sync.aligned.32x32b.x32.b32 {";
      fragList(j * 32, 32);
      os << "}, [" << addr << "];\n"
         << "    tcgen05.wait::ld.sync.aligned;\n";
      const char *base = "%row_base";
      if (j > 0) {
        os << "    add.s64 %st_base, %row_base, " << (j * 128) << ";\n";
        base = "%st_base";
      }
      for (uint32_t k = 0; k < 8; ++k) {
        if (k > 0)
          os << "    add.s64 %st_addr, " << base << ", " << (k * 16) << ";\n";
        os << "    st.global.v4.f32 [" << (k ? "%st_addr" : base) << "], {";
        fragList(j * 32 + k * 4, 4);
        os << "};\n";
      }
      os << "\n";
    }
  }

  void emitJoin() {
    os << "    // ---- join: fence, CTA barrier, warp " << loadBase
       << " deallocates the TMEM columns\n";
    os << "L_join:\n"
       << "    tcgen05.fence::before_thread_sync;\n"
       << "    bar.sync 0;\n"
       << "    tcgen05.fence::after_thread_sync;\n"
       << "    setp.ne.s32 %p1, %warpid, " << loadBase << ";\n"
       << "    @%p1 bra L_ret;\n"
       << "    ld.shared.u32 %tmem, [tmem_base];\n"
       << "    mov.b32 %cols, " << tmemCols << ";\n"
       << "    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %tmem, %cols;\n"
       << "L_ret:\n"
       << "    ret;\n"
       << "}\n";
  }

  bool run() {
    if (kernel.getArch().getValue() != Arch::sm_100a)
      return fail("v1 lowering: only arch sm_100a is supported");
    if (!gather())
      return false;
    emitHeader();
    emitPrologue();
    auto ws = *kernel.getBody().front().getOps<WarpSpecializeOp>().begin();
    for (RoleOp role : ws.getBody().front().getOps<RoleOp>()) {
      bool ok = false;
      switch (role.getKind()) {
      case RoleKind::load:
        ok = pipelined ? emitLoadRoleV2(role) : emitLoadRole(role);
        break;
      case RoleKind::mma:
        ok = pipelined ? emitMmaRoleV2(role) : emitMmaRole(role);
        break;
      case RoleKind::epi:
        ok = emitEpiRole(role);
        break;
      }
      if (!ok)
        return false;
    }
    emitJoin();
    os.flush();
    return true;
  }
};

class LowerToPTXPass
    : public PassWrapper<LowerToPTXPass, OperationPass<ModuleOp>> {
public:
  llvm::StringRef getArgument() const override { return "lower-to-ptx"; }
  llvm::StringRef getDescription() const override {
    return "Lower patina.kernel to PTX text (v1 + v2 pipelined forms)";
  }
  void runOnOperation() override {
    std::string err;
    std::optional<std::string> ptx = lowerModuleToPTX(getOperation(), err);
    if (!ptx) {
      getOperation()->emitError() << err;
      signalPassFailure();
      return;
    }
    llvm::outs() << *ptx;
    // The PTX went to stdout; drop the IR so the module dump adds nothing.
    getOperation().getBody()->clear();
  }
};

} // namespace

std::optional<std::string> lowerModuleToPTX(ModuleOp module, std::string &err) {
  std::vector<KernelOp> kernels;
  for (Operation &op : module.getBody()->getOperations())
    if (auto k = dyn_cast<KernelOp>(&op))
      kernels.push_back(k);
  if (kernels.empty()) {
    err = "no patina.kernel in module";
    return std::nullopt;
  }
  if (kernels.size() > 1) {
    err = "v1 lowering: a single patina.kernel per module is supported";
    return std::nullopt;
  }
  Lowerer lowerer(kernels.front());
  if (!lowerer.run()) {
    err = lowerer.err;
    return std::nullopt;
  }
  return lowerer.out;
}

void registerPatinaPasses() {
  registerPass([] { return std::make_unique<LowerToPTXPass>(); });
}

} // namespace patina
