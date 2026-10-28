//===- PatinaOps.cpp - op parsers, printers, verifiers --------------------===//
//
// Value references are symbolic names: the golden syntax annotates types
// only at definitions, so uses are parsed as `%name` tokens and stored as
// bare-name StringAttrs; result names are captured via getResultName and
// suggested back to the printer through OpAsmOpInterface. Verifiers resolve
// names within the enclosing patina.kernel.
//
// Diagnostic strings are the contract in dialect/tests/verifier.mlir.
//
//===----------------------------------------------------------------------===//

#include "patina/PatinaOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "patina/PatinaOps.cpp.inc"

using namespace mlir;
using namespace patina;

//===----------------------------------------------------------------------===//
// Parsing/printing helpers
//===----------------------------------------------------------------------===//

/// Parse `%name` and store the bare name (no '%') in the given attribute.
static ParseResult parseNameRef(OpAsmParser &parser, OperationState &result,
                                StringRef attrName) {
  OpAsmParser::UnresolvedOperand ref;
  if (parser.parseOperand(ref, /*allowResultNumber=*/false))
    return failure();
  result.addAttribute(
      attrName, parser.getBuilder().getStringAttr(ref.name.drop_front()));
  return success();
}

static void printNameRef(OpAsmPrinter &p, StringRef name) {
  p << "%" << name;
}

/// Capture the parsed result name (`%sa =` -> "sa") into `value_name`.
static void captureResultName(OpAsmParser &parser, OperationState &result) {
  result.addAttribute(
      "value_name",
      parser.getBuilder().getStringAttr(parser.getResultName(0).first));
}

//===----------------------------------------------------------------------===//
// Name resolution within the enclosing kernel (for verifiers/lowering)
//===----------------------------------------------------------------------===//

namespace patina {
/// Find the single-result patina op inside `kernel` whose captured
/// `value_name` equals `name`.
Operation *findDefByName(KernelOp kernel, StringRef name) {
  Operation *found = nullptr;
  kernel.walk([&](Operation *op) {
    if (found)
      return;
    if (auto attr = op->getAttrOfType<StringAttr>("value_name"))
      if (attr.getValue() == name)
        found = op;
  });
  return found;
}

/// The type of the value called `name`: a patina op result type, or a kernel
/// entry-block argument type (via the kernel's `arg_names` attribute).
Type resolveValueType(KernelOp kernel, StringRef name) {
  if (Operation *def = findDefByName(kernel, name))
    if (def->getNumResults() == 1)
      return def->getResult(0).getType();
  ArrayAttr argNames = kernel.getArgNames();
  Block &entry = kernel.getBody().front();
  for (auto [i, arg] : llvm::enumerate(entry.getArguments()))
    if (i < argNames.size() &&
        cast<StringAttr>(argNames[i]).getValue() == name)
      return arg.getType();
  return Type();
}
} // namespace patina

//===----------------------------------------------------------------------===//
// patina.kernel
//===----------------------------------------------------------------------===//

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &b = parser.getBuilder();

  StringAttr sym;
  if (parser.parseSymbolName(sym))
    return failure();
  result.addAttribute("sym_name", sym);

  // version = 0.1, arch = #patina.arch<sm_100a>, block = 192,
  if (parser.parseKeyword("version") || parser.parseEqual())
    return failure();
  Attribute version;
  if (parser.parseAttribute(version, b.getF64Type()))
    return failure();
  result.addAttribute("version", version);

  if (parser.parseComma() || parser.parseKeyword("arch") ||
      parser.parseEqual())
    return failure();
  Attribute arch;
  if (parser.parseAttribute(arch))
    return failure();
  result.addAttribute("arch", arch);

  if (parser.parseComma() || parser.parseKeyword("block") ||
      parser.parseEqual())
    return failure();
  int32_t block;
  if (parser.parseInteger(block))
    return failure();
  result.addAttribute("block", b.getI32IntegerAttr(block));

  // roles = [#patina.role<load, 1>, ...]
  if (parser.parseComma() || parser.parseKeyword("roles") ||
      parser.parseEqual() || parser.parseLSquare())
    return failure();
  SmallVector<Attribute> roles;
  if (failed(parser.parseOptionalRSquare())) {
    do {
      Attribute role;
      if (parser.parseAttribute(role))
        return failure();
      roles.push_back(role);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return failure();
  }
  result.addAttribute("roles", b.getArrayAttr(roles));

  // (%tma_a: !patina.tma_map<...>, ...)
  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();
  SmallVector<Attribute> argNames;
  for (auto &arg : args)
    argNames.push_back(b.getStringAttr(arg.ssaName.name.drop_front()));
  result.addAttribute("arg_names", b.getArrayAttr(argNames));

  return parser.parseRegion(*result.addRegion(), args,
                            /*enableNameShadowing=*/true);
}

void KernelOp::print(OpAsmPrinter &p) {
  p << " @" << getSymName() << "\n    version = ";
  char versionBuf[32];
  std::snprintf(versionBuf, sizeof(versionBuf), "%g",
                getVersionAttr().getValueAsDouble());
  p << versionBuf;
  p << ", arch = " << getArchAttr() << ", block = " << getBlock() << ",\n";
  p << "    roles = [";
  llvm::interleaveComma(getRoles(), p);
  p << "]\n";

  Block &entry = getBody().front();
  ArrayAttr argNames = getArgNames();
  size_t width = 0;
  for (auto i : llvm::seq<size_t>(0, entry.getNumArguments()))
    width = std::max(width,
                     cast<StringAttr>(argNames[i]).getValue().size() + 1);
  for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
    p << (i == 0 ? "  (" : ",\n   ");
    std::string head =
        ("%" + cast<StringAttr>(argNames[i]).getValue() + ":").str();
    p << head;
    for (size_t k = head.size(); k < width + 1; ++k)
      p << " ";
    p << " " << arg.getType();
  }
  p << ") ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

void KernelOp::getAsmBlockArgumentNames(Region &region,
                                        OpAsmSetValueNameFn setName) {
  ArrayAttr argNames = getArgNames();
  for (auto [i, arg] : llvm::enumerate(region.front().getArguments()))
    if (i < argNames.size())
      setName(arg, cast<StringAttr>(argNames[i]).getValue());
}

LogicalResult KernelOp::verify() {
  uint64_t warps = 0;
  for (Attribute attr : getRoles()) {
    auto role = dyn_cast<RoleAttr>(attr);
    if (!role)
      return emitOpError("roles must be #patina.role attributes");
    warps += role.getWarps();
  }
  if (warps * 32 != static_cast<uint64_t>(getBlock()))
    return emitOpError("roles require ")
           << warps << " warps (" << warps * 32
           << " threads) but block = " << getBlock();
  return success();
}

//===----------------------------------------------------------------------===//
// patina.role
//===----------------------------------------------------------------------===//

ParseResult RoleOp::parse(OpAsmParser &parser, OperationState &result) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  std::optional<RoleKind> kind = symbolizeRoleKind(keyword);
  if (!kind) {
    parser.emitError(parser.getCurrentLocation(), "unknown role: " + keyword);
    return failure();
  }
  result.addAttribute("kind", RoleKindAttr::get(parser.getContext(), *kind));
  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();
  // An empty `{ }` parses to zero blocks; the region must hold one.
  if (body->empty())
    body->emplaceBlock();
  return success();
}

void RoleOp::print(OpAsmPrinter &p) {
  p << " " << stringifyRoleKind(getKind()) << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// Runtime control flow (M3)
//===----------------------------------------------------------------------===//

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &b = parser.getBuilder();

  // `patina.for %i = 0 to %bound iter_args(%s = 0, %ph = 1) { ... }`
  OpAsmParser::UnresolvedOperand ix;
  int64_t lo;
  if (parser.parseOperand(ix, /*allowResultNumber=*/false) ||
      parser.parseEqual() || parser.parseInteger(lo))
    return failure();
  if (lo != 0) {
    parser.emitError(parser.getCurrentLocation(), "patina.for is zero-based");
    return failure();
  }
  OpAsmParser::UnresolvedOperand bound;
  if (parser.parseKeyword("to") ||
      parser.parseOperand(bound, /*allowResultNumber=*/false))
    return failure();
  result.addAttribute("index_name", b.getStringAttr(ix.name.drop_front()));
  result.addAttribute("bound", b.getStringAttr(bound.name.drop_front()));

  if (parser.parseKeyword("iter_args") || parser.parseLParen())
    return failure();
  SmallVector<Attribute> names;
  SmallVector<int64_t> inits;
  SmallVector<OpAsmParser::UnresolvedOperand> iterOperands;
  do {
    OpAsmParser::UnresolvedOperand v;
    int64_t init;
    if (parser.parseOperand(v, /*allowResultNumber=*/false) ||
        parser.parseEqual() || parser.parseInteger(init))
      return failure();
    iterOperands.push_back(v);
    names.push_back(b.getStringAttr(v.name.drop_front()));
    inits.push_back(init);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRParen())
    return failure();
  result.addAttribute("iter_names", b.getArrayAttr(names));
  result.addAttribute("iter_inits", b.getDenseI64ArrayAttr(inits));

  // Body block: (%i, %s, %ph, ...) all !patina.u32, named from the syntax.
  Type u32 = U32Type::get(b.getContext());
  SmallVector<OpAsmParser::Argument> regionArgs;
  OpAsmParser::Argument iArg;
  iArg.ssaName = ix;
  iArg.type = u32;
  regionArgs.push_back(iArg);
  for (auto &v : iterOperands) {
    OpAsmParser::Argument arg;
    arg.ssaName = v;
    arg.type = u32;
    regionArgs.push_back(arg);
  }
  return parser.parseRegion(*result.addRegion(), regionArgs,
                            /*enableNameShadowing=*/true);
}

void ForOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getIndexName());
  p << " = 0 to ";
  printNameRef(p, getBound());
  p << " iter_args(";
  ArrayAttr names = getIterNames();
  auto inits = getIterInits();
  for (size_t i = 0; i < names.size(); ++i) {
    if (i)
      p << ", ";
    printNameRef(p, cast<StringAttr>(names[i]).getValue());
    p << " = " << inits[i];
  }
  p << ") ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

void ForOp::getAsmBlockArgumentNames(Region &region,
                                     OpAsmSetValueNameFn setName) {
  ArrayAttr names = getIterNames();
  for (auto [i, arg] : llvm::enumerate(region.front().getArguments())) {
    if (i == 0)
      setName(arg, getIndexName());
    else if (i - 1 < names.size())
      setName(arg, cast<StringAttr>(names[i - 1]).getValue());
  }
}

ParseResult YieldOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &b = parser.getBuilder();
  SmallVector<Attribute> vals;
  do {
    OpAsmParser::UnresolvedOperand v;
    if (parser.parseOperand(v, /*allowResultNumber=*/false))
      return failure();
    vals.push_back(b.getStringAttr(v.name.drop_front()));
  } while (succeeded(parser.parseOptionalComma()));
  result.addAttribute("values", b.getArrayAttr(vals));
  return success();
}

void YieldOp::print(OpAsmPrinter &p) {
  p << " ";
  llvm::interleaveComma(getValues(), p, [&](Attribute a) {
    printNameRef(p, cast<StringAttr>(a).getValue());
  });
}

LogicalResult YieldOp::verify() {
  auto forOp = getOperation()->getParentOfType<ForOp>();
  if (!forOp)
    return emitOpError("patina.yield must be inside a patina.for region");
  size_t want = forOp.getIterNames().size();
  if (getValues().size() != want)
    return emitOpError("patina.yield operand count (")
           << getValues().size() << ") does not match patina.for iter_args ("
           << want << ")";
  return success();
}

ParseResult MuliOp::parse(OpAsmParser &parser, OperationState &result) {
  int32_t k;
  if (parseNameRef(parser, result, "x") || parser.parseComma() ||
      parser.parseInteger(k))
    return failure();
  result.addAttribute("k", parser.getBuilder().getI32IntegerAttr(k));
  result.addTypes(U32Type::get(parser.getContext()));
  captureResultName(parser, result);
  return success();
}

void MuliOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getX());
  p << ", " << getK();
}

void MuliOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getValueName().empty())
    setName(getResult(), getValueName());
}

ParseResult NezOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseNameRef(parser, result, "x"))
    return failure();
  result.addTypes(U32Type::get(parser.getContext()));
  captureResultName(parser, result);
  return success();
}

void NezOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getX());
}

void NezOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getValueName().empty())
    setName(getResult(), getValueName());
}

//===----------------------------------------------------------------------===//
// CTA-scope resources
//===----------------------------------------------------------------------===//

ParseResult PipelineCreateOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  int32_t stages;
  Type ty;
  if (parser.parseKeyword("stages") || parser.parseEqual() ||
      parser.parseInteger(stages) || parser.parseColon() ||
      parser.parseType(ty))
    return failure();
  result.addAttribute("stages", parser.getBuilder().getI32IntegerAttr(stages));
  result.addTypes(ty);
  captureResultName(parser, result);
  return success();
}

void PipelineCreateOp::print(OpAsmPrinter &p) {
  p << " stages = " << getStages() << " : " << getResult().getType();
}

void PipelineCreateOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getValueName().empty())
    setName(getResult(), getValueName());
}

ParseResult SmemAllocOp::parse(OpAsmParser &parser, OperationState &result) {
  // Optional `stages = S`: a staged (pipelined) buffer of S tiles.
  if (succeeded(parser.parseOptionalKeyword("stages"))) {
    int32_t stages;
    if (parser.parseEqual() || parser.parseInteger(stages))
      return failure();
    result.addAttribute("stages",
                        parser.getBuilder().getI32IntegerAttr(stages));
  }
  Type ty;
  if (parser.parseColon() || parser.parseType(ty))
    return failure();
  result.addTypes(ty);
  captureResultName(parser, result);
  return success();
}

void SmemAllocOp::print(OpAsmPrinter &p) {
  if (auto stages = getStagesAttr())
    p << " stages = " << stages.getInt();
  p << " : " << getResult().getType();
}

void SmemAllocOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getValueName().empty())
    setName(getResult(), getValueName());
}

ParseResult StageBufOp::parse(OpAsmParser &parser, OperationState &result) {
  Type ty;
  if (parseNameRef(parser, result, "base") || parser.parseComma() ||
      parseNameRef(parser, result, "stage") || parser.parseColon() ||
      parser.parseType(ty))
    return failure();
  result.addTypes(ty);
  captureResultName(parser, result);
  return success();
}

void StageBufOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getBase());
  p << ", ";
  printNameRef(p, getStage());
  p << " : " << getResult().getType();
}

void StageBufOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getValueName().empty())
    setName(getResult(), getValueName());
}

LogicalResult StageBufOp::verify() {
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return success();
  auto alloc = dyn_cast_or_null<SmemAllocOp>(findDefByName(kernel, getBase()));
  int64_t stages = 1;
  if (alloc)
    if (auto s = alloc.getStagesAttr())
      stages = s.getInt();
  if (!alloc || stages <= 1)
    return emitOpError(
        "patina.stage_buf requires a staged smem_alloc (stages > 1)");
  return success();
}

ParseResult MBarrierCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  int32_t arrivals;
  Type ty;
  if (parser.parseKeyword("arrivals") || parser.parseEqual() ||
      parser.parseInteger(arrivals) || parser.parseColon() ||
      parser.parseType(ty))
    return failure();
  result.addAttribute("arrivals",
                      parser.getBuilder().getI32IntegerAttr(arrivals));
  result.addTypes(ty);
  captureResultName(parser, result);
  return success();
}

void MBarrierCreateOp::print(OpAsmPrinter &p) {
  p << " arrivals = " << getArrivals() << " : " << getResult().getType();
}

void MBarrierCreateOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getValueName().empty())
    setName(getResult(), getValueName());
}

ParseResult TmemAllocOp::parse(OpAsmParser &parser, OperationState &result) {
  int32_t cols;
  Type ty;
  if (parser.parseKeyword("cols") || parser.parseEqual() ||
      parser.parseInteger(cols) || parser.parseColon() || parser.parseType(ty))
    return failure();
  result.addAttribute("cols", parser.getBuilder().getI32IntegerAttr(cols));
  result.addTypes(ty);
  captureResultName(parser, result);
  return success();
}

void TmemAllocOp::print(OpAsmPrinter &p) {
  p << " cols = " << getCols() << " : " << getResult().getType();
}

void TmemAllocOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getValueName().empty())
    setName(getResult(), getValueName());
}

LogicalResult TmemAllocOp::verify() {
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return success();
  unsigned deallocs = 0;
  kernel.walk([&](TmemDeallocOp dealloc) {
    if (dealloc.getAcc() == getValueName())
      ++deallocs;
  });
  if (deallocs != 1)
    return emitOpError(
        "tmem.alloc result must be consumed by exactly one patina.tmem.dealloc");
  return success();
}

ParseResult TmemDeallocOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseNameRef(parser, result, "acc");
}

void TmemDeallocOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getAcc());
}

//===----------------------------------------------------------------------===//
// Load role
//===----------------------------------------------------------------------===//

ParseResult ExpectTxOp::parse(OpAsmParser &parser, OperationState &result) {
  int64_t bytes;
  if (parseNameRef(parser, result, "barrier") || parser.parseComma() ||
      parser.parseInteger(bytes))
    return failure();
  result.addAttribute("bytes", parser.getBuilder().getI64IntegerAttr(bytes));
  return success();
}

void ExpectTxOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getBarrier());
  p << ", " << getBytes();
}

/// Byte size of a tensor value: product(shape) * dtype bits / 8.
static uint64_t tensorBytes(patina::TensorType ty) {
  uint64_t elems = 1;
  for (int64_t d : ty.getShape())
    elems *= static_cast<uint64_t>(d);
  return elems * ty.getDtype().getIntOrFloatBitWidth() / 8;
}

LogicalResult ExpectTxOp::verify() {
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return success();
  uint64_t sum = 0;
  LogicalResult resolved = success();
  kernel.walk([&](TmaLoadOp load) {
    if (load.getCompleteKindAttr())
      return; // staged completion: checked by ExpectTxStageOp
    if (load.getComplete() != getBarrier())
      return;
    auto ty = dyn_cast_or_null<patina::TensorType>(
        resolveValueType(kernel, load.getDst()));
    if (!ty) {
      resolved = emitOpError("cannot resolve type of tma_load destination '%")
                 << load.getDst() << "'";
      return;
    }
    sum += tensorBytes(ty);
  });
  if (failed(resolved))
    return failure();
  if (sum != static_cast<uint64_t>(getBytes()))
    return emitOpError("expect_tx bytes (")
           << getBytes() << ") != sum of tma_load bytes on this barrier ("
           << sum << ")";
  return success();
}

ParseResult ExpectTxStageOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  int64_t bytes;
  if (parseNameRef(parser, result, "pipe") || parser.parseComma())
    return failure();
  StringRef kind;
  if (parser.parseKeyword(&kind))
    return failure();
  result.addAttribute("kind", parser.getBuilder().getStringAttr(kind));
  if (parser.parseComma() || parseNameRef(parser, result, "stage") ||
      parser.parseComma() || parser.parseInteger(bytes))
    return failure();
  result.addAttribute("bytes", parser.getBuilder().getI64IntegerAttr(bytes));
  return success();
}

void ExpectTxStageOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getPipe());
  p << ", " << getKind() << ", ";
  printNameRef(p, getStage());
  p << ", " << getBytes();
}

LogicalResult ExpectTxStageOp::verify() {
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return success();
  uint64_t sum = 0;
  LogicalResult resolved = success();
  kernel.walk([&](TmaLoadOp load) {
    auto kindAttr = load.getCompleteKindAttr();
    if (!kindAttr)
      return; // standalone-barrier completion: checked by ExpectTxOp
    if (load.getComplete() != getPipe() ||
        kindAttr.getValue() != getKind())
      return;
    auto ty = dyn_cast_or_null<patina::TensorType>(
        resolveValueType(kernel, load.getDst()));
    if (!ty) {
      resolved = emitOpError("cannot resolve type of tma_load destination '%")
                 << load.getDst() << "'";
      return;
    }
    sum += tensorBytes(ty);
  });
  if (failed(resolved))
    return failure();
  if (sum != static_cast<uint64_t>(getBytes()))
    return emitOpError("expect_tx_stage bytes (")
           << getBytes()
           << ") != sum of tma_load bytes on this stage barrier (" << sum
           << ")";
  return success();
}

ParseResult TmaLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &b = parser.getBuilder();
  if (parseNameRef(parser, result, "map") || parser.parseArrow() ||
      parseNameRef(parser, result, "dst") || parser.parseComma() ||
      parser.parseKeyword("coords") || parser.parseLSquare())
    return failure();
  // Coordinates: integers (compile-time) or %name (runtime scalar).
  SmallVector<Attribute> coords;
  if (failed(parser.parseOptionalRSquare())) {
    do {
      int64_t c;
      OptionalParseResult parsed = parser.parseOptionalInteger(c);
      if (parsed.has_value()) {
        if (failed(*parsed))
          return failure();
        coords.push_back(b.getI64IntegerAttr(c));
      } else {
        OpAsmParser::UnresolvedOperand ref;
        if (parser.parseOperand(ref, /*allowResultNumber=*/false))
          return failure();
        coords.push_back(b.getStringAttr(ref.name.drop_front()));
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return failure();
  }
  result.addAttribute("coords", b.getArrayAttr(coords));
  if (parser.parseComma() || parser.parseKeyword("complete") ||
      parseNameRef(parser, result, "complete"))
    return failure();
  // Optional staged completion: `complete %pipe, full, %s`.
  if (succeeded(parser.parseOptionalComma())) {
    StringRef kind;
    if (parser.parseKeyword(&kind))
      return failure();
    result.addAttribute("complete_kind", b.getStringAttr(kind));
    if (parser.parseComma() ||
        parseNameRef(parser, result, "complete_stage"))
      return failure();
  }
  return success();
}

void TmaLoadOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getMap());
  p << " -> ";
  printNameRef(p, getDst());
  p << ", coords [";
  llvm::interleaveComma(getCoords(), p, [&](Attribute c) {
    if (auto name = dyn_cast<StringAttr>(c))
      p << "%" << name.getValue();
    else
      p << cast<IntegerAttr>(c).getInt();
  });
  p << "], complete ";
  printNameRef(p, getComplete());
  if (auto kind = getCompleteKindAttr()) {
    p << ", " << kind.getValue() << ", ";
    printNameRef(p, getCompleteStageAttr().getValue());
  }
}

ParseResult WaitOp::parse(OpAsmParser &parser, OperationState &result) {
  int32_t phase;
  if (parseNameRef(parser, result, "barrier") || parser.parseComma() ||
      parser.parseKeyword("phase") || parser.parseEqual() ||
      parser.parseInteger(phase))
    return failure();
  result.addAttribute("phase", parser.getBuilder().getI32IntegerAttr(phase));
  return success();
}

void WaitOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getBarrier());
  p << ", phase = " << getPhase();
}

LogicalResult WaitOp::verify() {
  // Phases of each barrier must be waited in order within a role region.
  int32_t maxPrior = -1;
  for (Operation &op : *getOperation()->getBlock()) {
    if (&op == getOperation())
      break;
    auto prior = dyn_cast<WaitOp>(&op);
    if (prior && prior.getBarrier() == getBarrier())
      maxPrior = std::max<int32_t>(maxPrior, prior.getPhase());
  }
  if (static_cast<int32_t>(getPhase()) > maxPrior + 1)
    return emitOpError("phase ")
           << getPhase() << " waited before phase " << (maxPrior + 1)
           << " on this barrier";
  return success();
}

ParseResult WaitStageOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseNameRef(parser, result, "pipe") || parser.parseComma())
    return failure();
  StringRef kind;
  if (parser.parseKeyword(&kind))
    return failure();
  result.addAttribute("kind", parser.getBuilder().getStringAttr(kind));
  if (parser.parseComma() || parseNameRef(parser, result, "stage") ||
      parser.parseComma() || parseNameRef(parser, result, "phase"))
    return failure();
  return success();
}

void WaitStageOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getPipe());
  p << ", " << getKind() << ", ";
  printNameRef(p, getStage());
  p << ", ";
  printNameRef(p, getPhase());
}

LogicalResult WaitStageOp::verify() {
  // Kinds are bare keywords parsed from text; reject typos rather than
  // treating everything that is not 'empty' as 'full'.
  StringRef kind = getKind();
  if (kind != "empty" && kind != "full")
    return emitOpError("unknown pipeline barrier kind '")
           << kind << "' (must be 'empty' or 'full')";
  // Producer waits `empty` (load role), consumer waits `full` (mma role).
  auto role = getOperation()->getParentOfType<RoleOp>();
  StringRef need = kind == "empty" ? "load" : "mma";
  StringRef have = role ? stringifyRoleKind(role.getKind()) : "";
  if (have != need)
    return emitOpError()
           << "'patina.wait_stage' on '" << kind
           << "' barriers is only legal in a '" << need << "' role region";
  return success();
}

ParseResult StageNextOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &b = parser.getBuilder();
  if (parseNameRef(parser, result, "s") || parser.parseComma() ||
      parseNameRef(parser, result, "ph") || parser.parseKeyword("of") ||
      parseNameRef(parser, result, "pipe"))
    return failure();
  Type u32 = U32Type::get(b.getContext());
  result.addTypes({u32, u32});
  result.addAttribute("stage_name",
                      b.getStringAttr(parser.getResultName(0).first));
  result.addAttribute("phase_name",
                      b.getStringAttr(parser.getResultName(1).first));
  return success();
}

void StageNextOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getS());
  p << ", ";
  printNameRef(p, getPh());
  p << " of ";
  printNameRef(p, getPipe());
}

void StageNextOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getStageName().empty())
    setName(getNextStage(), getStageName());
  if (!getPhaseName().empty())
    setName(getNextPhase(), getPhaseName());
}

//===----------------------------------------------------------------------===//
// MMA role
//===----------------------------------------------------------------------===//

static LogicalResult verifyInMmaRole(Operation *op) {
  auto role = op->getParentOfType<RoleOp>();
  if (!role || role.getKind() != RoleKind::mma)
    return op->emitOpError() << "'" << op->getName().getStringRef()
                             << "' is only legal inside an 'mma' role region";
  return success();
}

ParseResult UmmaBeginOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseNameRef(parser, result, "acc") || parser.parseComma() ||
      parseNameRef(parser, result, "a") || parser.parseComma() ||
      parseNameRef(parser, result, "b"))
    return failure();
  result.addTypes(UmmaType::get(parser.getContext()));
  captureResultName(parser, result);
  return success();
}

void UmmaBeginOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getAcc());
  p << ", ";
  printNameRef(p, getA());
  p << ", ";
  printNameRef(p, getB());
}

void UmmaBeginOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getValueName().empty())
    setName(getResult(), getValueName());
}

LogicalResult UmmaBeginOp::verify() { return verifyInMmaRole(*this); }

ParseResult UmmaStepOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseNameRef(parser, result, "m"))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    // Dynamic scaleC: `patina.umma.step %m, accum %cond`.
    if (parser.parseKeyword("accum") ||
        parseNameRef(parser, result, "accum_ref"))
      return failure();
    result.addAttribute("accum", parser.getBuilder().getBoolAttr(false));
    return success();
  }
  return parser.parseOptionalAttrDict(result.attributes);
}

void UmmaStepOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getM());
  if (auto ref = getAccumRefAttr(); ref && !ref.getValue().empty()) {
    p << ", accum ";
    printNameRef(p, ref.getValue());
  } else {
    p << " {accum = " << (getAccum() ? "true" : "false") << "}";
  }
}

LogicalResult UmmaStepOp::verify() { return verifyInMmaRole(*this); }

ParseResult UmmaAdvanceKOp::parse(OpAsmParser &parser, OperationState &result) {
  int32_t k;
  if (parseNameRef(parser, result, "m") || parser.parseComma() ||
      parser.parseInteger(k))
    return failure();
  result.addAttribute("k", parser.getBuilder().getI32IntegerAttr(k));
  return success();
}

void UmmaAdvanceKOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getM());
  p << ", " << getK();
}

LogicalResult UmmaAdvanceKOp::verify() { return verifyInMmaRole(*this); }

ParseResult UmmaCommitOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseNameRef(parser, result, "m") || parser.parseArrow() ||
      parseNameRef(parser, result, "barrier"))
    return failure();
  // Optional staged target: `-> %pipe, empty, %s`.
  if (succeeded(parser.parseOptionalComma())) {
    StringRef kind;
    if (parser.parseKeyword(&kind))
      return failure();
    result.addAttribute("barrier_kind",
                        parser.getBuilder().getStringAttr(kind));
    if (parser.parseComma() ||
        parseNameRef(parser, result, "barrier_stage"))
      return failure();
  }
  return success();
}

void UmmaCommitOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getM());
  p << " -> ";
  printNameRef(p, getBarrier());
  if (auto kind = getBarrierKindAttr()) {
    p << ", " << kind.getValue() << ", ";
    printNameRef(p, getBarrierStageAttr().getValue());
  }
}

LogicalResult UmmaCommitOp::verify() { return verifyInMmaRole(*this); }

//===----------------------------------------------------------------------===//
// Epilogue role
//===----------------------------------------------------------------------===//

ParseResult TmemLdOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseNameRef(parser, result, "acc"))
    return failure();
  Type ty;
  if (parser.parseColon() || parser.parseType(ty))
    return failure();
  result.addTypes(ty);
  captureResultName(parser, result);
  return success();
}

void TmemLdOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getAcc());
  p << " : " << getResult().getType();
}

void TmemLdOp::getAsmResultNames(OpAsmSetValueNameFn setName) {
  if (!getValueName().empty())
    setName(getResult(), getValueName());
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseNameRef(parser, result, "ptr") || parser.parseComma() ||
      parseNameRef(parser, result, "frag"))
    return failure();
  return success();
}

void StoreOp::print(OpAsmPrinter &p) {
  p << " ";
  printNameRef(p, getPtr());
  p << ", ";
  printNameRef(p, getFrag());
}
