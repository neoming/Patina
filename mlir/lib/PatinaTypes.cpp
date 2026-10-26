//===- PatinaTypes.cpp - custom type parsers/printers ---------------------===//
//
// Syntax (contract: crates/patina/tests/golden/v1_gemm.mlir):
//   !patina.tensor<f16, 128x32, smem, #lay>
//   !patina.tmem<f32, 128>     !patina.mbarrier<1>   !patina.umma
//   !patina.tma_map<f16, #lay> !patina.frag<f32, 128>  !patina.ptr<f32, gmem>
//
//===----------------------------------------------------------------------===//

#include "patina/PatinaDialect.h"

using namespace mlir;
using namespace patina;

// Parse `<f16, 128x32, smem, #lay>`.
Type patina::TensorType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  Type dtype;
  if (parser.parseType(dtype) || parser.parseComma())
    return Type();
  SmallVector<int64_t> shape;
  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                       /*withTrailingX=*/false)))
    return Type();
  if (parser.parseComma())
    return Type();
  StringRef spaceKeyword;
  if (parser.parseKeyword(&spaceKeyword) || parser.parseComma())
    return Type();
  std::optional<Space> space = symbolizeSpace(spaceKeyword);
  if (!space) {
    parser.emitError(parser.getCurrentLocation(),
                     "unknown memory space: " + spaceKeyword);
    return Type();
  }
  Attribute layout;
  if (parser.parseAttribute(layout) || parser.parseGreater())
    return Type();
  return patina::TensorType::get(parser.getContext(), dtype, shape, *space,
                                 layout);
}

void patina::TensorType::print(AsmPrinter &printer) const {
  printer << "<" << getDtype() << ", ";
  llvm::interleave(
      getShape(), printer, [&](int64_t d) { printer << d; }, "x");
  printer << ", " << stringifySpace(getSpace()) << ", " << getLayout() << ">";
}

Type TmemType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  Type dtype;
  uint32_t cols;
  if (parser.parseType(dtype) || parser.parseComma() ||
      parser.parseInteger(cols) || parser.parseGreater())
    return Type();
  return TmemType::get(parser.getContext(), dtype, cols);
}

void TmemType::print(AsmPrinter &printer) const {
  printer << "<" << getDtype() << ", " << getCols() << ">";
}

Type MBarrierType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  uint32_t arrivals;
  if (parser.parseInteger(arrivals) || parser.parseGreater())
    return Type();
  return MBarrierType::get(parser.getContext(), arrivals);
}

void MBarrierType::print(AsmPrinter &printer) const {
  printer << "<" << getArrivals() << ">";
}

Type TmaMapType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  Type dtype;
  Attribute layout;
  if (parser.parseType(dtype) || parser.parseComma() ||
      parser.parseAttribute(layout) || parser.parseGreater())
    return Type();
  return TmaMapType::get(parser.getContext(), dtype, layout);
}

void TmaMapType::print(AsmPrinter &printer) const {
  printer << "<" << getDtype() << ", " << getLayout() << ">";
}

Type FragType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  Type dtype;
  uint32_t elems;
  if (parser.parseType(dtype) || parser.parseComma() ||
      parser.parseInteger(elems) || parser.parseGreater())
    return Type();
  return FragType::get(parser.getContext(), dtype, elems);
}

void FragType::print(AsmPrinter &printer) const {
  printer << "<" << getDtype() << ", " << getElems() << ">";
}

Type PtrType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  Type dtype;
  if (parser.parseType(dtype) || parser.parseComma())
    return Type();
  StringRef spaceKeyword;
  if (parser.parseKeyword(&spaceKeyword) || parser.parseGreater())
    return Type();
  std::optional<Space> space = symbolizeSpace(spaceKeyword);
  if (!space) {
    parser.emitError(parser.getCurrentLocation(),
                     "unknown memory space: " + spaceKeyword);
    return Type();
  }
  return PtrType::get(parser.getContext(), dtype, *space);
}

void PtrType::print(AsmPrinter &printer) const {
  printer << "<" << getDtype() << ", " << stringifySpace(getSpace()) << ">";
}

Type UmmaType::parse(AsmParser &parser) {
  // Parameterless: `!patina.umma`.
  return UmmaType::get(parser.getContext());
}

void UmmaType::print(AsmPrinter &printer) const {}
