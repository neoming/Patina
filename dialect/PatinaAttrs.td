//===- PatinaAttrs.td - Patina dialect attributes -----------*- tablegen -*-===//
//
// Attributes of the Patina dialect: layouts, roles, arch, memory spaces.
// Requires an LLVM/MLIR dev tree (mlir-tblgen); M1c build deferred.
//
//===----------------------------------------------------------------------===//

#ifndef PATINA_ATTRS_TD
#define PATINA_ATTRS_TD

include "mlir/IR/EnumAttr.td"
include "PatinaDialect.td"

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

// TMA/UMMA swizzle modes. v1 supports only SWIZZLE_NONE.
def Patina_SwizzleNone : I32EnumAttrCase<"swizzle_none", 0>;

def Patina_Swizzle : I32EnumAttr<"Swizzle", "TMA/UMMA smem swizzle mode",
    [Patina_SwizzleNone]> {
  let cppNamespace = "::patina";
  let genSpecializedAttr = 0;
}

// Role kinds, in canonical warp-assignment order.
def Patina_RoleLoad : I32EnumAttrCase<"load", 0>;
def Patina_RoleMma  : I32EnumAttrCase<"mma",  1>;
def Patina_RoleEpi  : I32EnumAttrCase<"epi",  2>;

def Patina_RoleKind : I32EnumAttr<"RoleKind", "warp-specialization role",
    [Patina_RoleLoad, Patina_RoleMma, Patina_RoleEpi]> {
  let cppNamespace = "::patina";
  let genSpecializedAttr = 0;
}

// Target architectures.
def Patina_ArchSm100a : I32EnumAttrCase<"sm_100a", 0>;

def Patina_Arch : I32EnumAttr<"Arch", "target GPU architecture",
    [Patina_ArchSm100a]> {
  let cppNamespace = "::patina";
  let genSpecializedAttr = 0;
}

// Memory spaces (appear bare inside type parameter lists, e.g.
// `!patina.tensor<f16, 128x32, smem, #lay>`).
def Patina_SpaceGmem : I32EnumAttrCase<"gmem", 0>;
def Patina_SpaceSmem : I32EnumAttrCase<"smem", 1>;
def Patina_SpaceTmem : I32EnumAttrCase<"tmem", 2>;

def Patina_Space : I32EnumAttr<"Space", "memory space",
    [Patina_SpaceGmem, Patina_SpaceSmem, Patina_SpaceTmem]> {
  let cppNamespace = "::patina";
  let genSpecializedAttr = 0;
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

class Patina_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Patina_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Canonical UMMA K-major smem layout. Printed `#patina.umma_k_major<...>`;
// the front-end aliases it as `#lay`.
def Patina_UmmaKMajorAttr : Patina_Attr<"UmmaKMajor", "umma_k_major"> {
  let summary = "canonical UMMA K-major smem layout (8x16B core matrices)";
  let description = [{
    Core matrix = 8 rows x 16B; core matrices tiled `(8, m) stride SBO` over
    rows and `(T, 2) stride LBO` over K. For swizzle_none the derived
    encodings are LBO = 128B, SBO = 256B, descriptor version = 1; the 4D TMA
    view is `{16B/elt, 8, K*elt/16, M/8}` with byte strides
    `{K*elt, 8*elt, 8*K*elt}` — one layout value drives host TMA encode and
    device descriptor packing (see patina::layout::UmmaKMajor).
  }];
  let parameters = (ins "Patina_Space" : $space, "Patina_Swizzle" : $swizzle);
  // NOTE: the golden syntax `#patina.umma_k_major<swizzle_none>` carries only
  // the swizzle; the space parameter defaults to smem in the custom parser.
}

// One role of a kernel's warp specialization: `#patina.role<load, 1>`.
def Patina_RoleAttr : Patina_Attr<"Role", "role"> {
  let summary = "a warp-specialization role and its warp count";
  let description = [{
    Warps are auto-packed in load/mma/epi order (v1). The `patina.kernel`
    verifier checks that the sum of warp counts x 32 equals `block` threads:
    "roles require 8 warps (256 threads) but block = 128".
  }];
  let parameters = (ins "Patina_RoleKind" : $kind, "uint32_t" : $warps);
  let assemblyFormat = "`<` $kind `,` $warps `>`";
}

// Target arch: `#patina.arch<sm_100a>`.
def Patina_ArchAttr : Patina_Attr<"ArchAttr", "arch"> {
  let summary = "target GPU architecture";
  let parameters = (ins "Patina_Arch" : $value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // PATINA_ATTRS_TD
