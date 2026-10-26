//===- LowerToPTX.cpp - patina.kernel -> PTX text (v1) --------------------===//
//
// Emits PTX source directly (no nvgpu / LLVM IR round-trip) for the v1
// warp-specialized form: single-warp load and mma roles, an N-warp epilogue,
// f16 x f16 -> f32 UMMA over UmmaKMajor (swizzle_none) smem. Instruction
// forms follow the reference kernel tcgen05_gemm.cu: smem descriptors are
// (addr>>4) | LBO(128)>>4<<16 | SBO(256)>>4<<32 | version 1<<46, the idesc is
// c_format(F32)<<4 | (N>>3)<<17 | (M>>4)<<24, and one K16 step advances the
// descriptors by 16 units (2 core matrices).
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
  uint64_t bytes = 0;
};

struct ParamInfo {
  std::string name;
  bool isTmaMap;
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
        params.push_back({pname, true});
      else if (isa<PtrType>(ty))
        params.push_back({pname, false});
      else
        return fail("v1 lowering: unsupported kernel argument type for %" +
                    pname);
    }

    WarpSpecializeOp ws;
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
        std::string name = nameOf(alloc.getValueNameAttr());
        smem[name] = info;
        smemOrder.push_back(name);
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

    os << ".visible .entry " << kernel.getSymName() << "(\n";
    for (size_t i = 0; i < params.size(); ++i) {
      const ParamInfo &p = params[i];
      os << "    .param ";
      if (p.isTmaMap)
        os << ".align 64 .b8 " << p.name << "[128]";
      else
        os << ".u64 " << p.name;
      if (i + 1 < params.size())
        os << ",";
      os << "\n";
    }
    os << ")\n";
    os << ".maxntid " << kernel.getBlock() << ", 1, 1\n";
    os << "{\n";

    os << "    .reg .pred %p<3>;\n";
    os << "    .reg .b32 %tidx, %warpid, %laneid, %ewarp;\n";
    os << "    .reg .b32 ";
    for (size_t i = 0; i < barrierOrder.size(); ++i)
      os << (i ? ", " : "") << "%bar_" << barrierOrder[i];
    os << ", %cols, %tx;\n";
    os << "    .reg .b32 %tmem, %tmem_w, %lane_off, %tmem_addr, %smem_addr;\n";
    os << "    .reg .b32 %idesc, %scale_c, %zero, %row;\n";
    os << "    .reg .b64 ";
    for (size_t i = 0; i < params.size(); ++i)
      os << (i ? ", " : "") << "%" << params[i].name << "_ptr";
    os << ";\n";
    os << "    .reg .b64 %da, %db, %off, %row_base, %st_base, %st_addr;\n";
    os << "    .reg .f32 %f<" << fragElems() << ">;\n\n";

    os << "    // CTA scope: smem tiles, barriers, and the TMEM column slot\n";
    for (const std::string &name : smemOrder)
      os << "    .shared .align 1024 .b8 " << name << "_smem["
         << smem[name].bytes << "];\n";
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

  bool emitEpiRole(RoleOp role) {
    os << "\n    // ---- epi role: warps " << epiBase << ".."
       << (epiBase + epiWarps - 1) << ", all lanes\n";
    os << "L_epi:\n";
    const ParamInfo *ptr = nullptr;
    for (const ParamInfo &p : params)
      if (!p.isTmaMap) {
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
      case RoleKind::load: ok = emitLoadRole(role); break;
      case RoleKind::mma:  ok = emitMmaRole(role); break;
      case RoleKind::epi:  ok = emitEpiRole(role); break;
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
    return "Lower patina.kernel to PTX text (v1 warp-specialized form)";
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
