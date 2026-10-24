//! Layout algebra and every derived encoding (M1a).
//!
//! The math here mirrors the reference kernel `tcgen05_gemm.cu`
//! (`make_umma_tmap_f16` / `make_desc` / `make_idesc_f16`), so that the
//! descriptor bit-packing, the 4D TMA view and the K-advance are *derived*
//! from one layout value instead of being hand-encoded.
//!
//! Canonical UMMA K-major, SWIZZLE_NONE smem layout (byte offsets):
//!   offset = kc*128 + (r/8)*256 + (r%8)*16 + i*elt_bytes
//! where `kc` indexes 16B core matrices along K, `r` is the row, and `i` the
//! element within a 16B core-matrix row.

use std::marker::PhantomData;

/// Shape literal: `shape![128, 32]`.
#[macro_export]
macro_rules! shape {
    ($($d:expr),+ $(,)?) => {
        $crate::layout::Shape::from(vec![$($d as u64),+])
    };
}
pub use crate::shape;

/// A tensor shape, in elements, outermost first (`[M, K]` for a GEMM tile).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(pub Vec<u64>);

impl From<Vec<u64>> for Shape {
    fn from(dims: Vec<u64>) -> Self {
        Shape(dims)
    }
}

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    /// Total element count.
    pub fn elems(&self) -> u64 {
        self.0.iter().product()
    }
    /// Dialect spelling: `128x32`.
    pub fn mlir(&self) -> String {
        self.0
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }
}

/// Element dtype for host-side layout math.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
}

pub const F16: DType = DType::F16;
pub const F32: DType = DType::F32;

impl DType {
    pub fn bytes(self) -> u64 {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
        }
    }
}

/// TMA/UMMA swizzle modes. v1 supports only SWIZZLE_NONE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swizzle {
    None,
}

/// Marker type selecting `Swizzle::None` at the type level.
pub struct NoSwizzle;

/// The 4D TMA tensor-map view derived from an [`UmmaKMajor`] layout.
///
/// For a `[128, 32]` f16 K-major tile: `{8 f16, 8 rows, 4 k-chunks,
/// 16 row-blocks}` with byte strides `{64, 16, 512}` for dims 1..=3
/// (dim0's stride is implicitly the element size). TMA fills smem linearly
/// in dim order, reproducing the canonical UMMA layout exactly.
pub struct TmaView {
    pub rank: usize,
    pub global_dim: [u64; 4],
    pub box_dim: [u64; 4],
    /// Strides of dims 1..=3, in bytes.
    pub global_stride_bytes: [u64; 3],
    pub swizzle: Swizzle,
    elt_bytes: u64,
}

impl TmaView {
    /// Byte offset in smem that a TMA copy writes coordinate
    /// `(i, mi, kc, mo)` to: element `i` of core-matrix row `mi`, k-chunk
    /// `kc`, 8-row block `mo`. Reproduces `kc*128 + mo*256 + mi*16 + i*elt`.
    pub fn smem_offset(&self, i: u64, mi: u64, kc: u64, mo: u64) -> u64 {
        kc * 128 + mo * 256 + mi * 16 + i * self.elt_bytes
    }
}

/// UMMA shared-memory descriptor fields (`make_desc` in the reference .cu).
/// All byte quantities are reported raw; the hardware encoding shifts them
/// right by 4 (16-byte units).
pub struct Descriptor {
    start_addr: u64,
    lbo: u64,
    sbo: u64,
    version: u64,
    swizzle_mode: u64,
}

impl Descriptor {
    /// Bits 0..13 of the descriptor: the smem address in 16-byte units.
    pub fn start_addr_enc(&self) -> u64 {
        self.start_addr >> 4
    }
    /// Leading byte offset: distance between core matrices adjacent in K.
    pub fn lbo(&self) -> u64 {
        self.lbo
    }
    /// Stride byte offset: distance between adjacent 8-row blocks.
    pub fn sbo(&self) -> u64 {
        self.sbo
    }
    /// Descriptor version; 1 on Blackwell.
    pub fn version(&self) -> u64 {
        self.version
    }
    /// 0 == SWIZZLE_NONE.
    pub fn swizzle_mode(&self) -> u64 {
        self.swizzle_mode
    }
}

/// Canonical UMMA K-major smem layout (CuTe-style layout value).
///
/// Core matrix = 8 rows x 16B; core matrices are tiled `(8, m) stride SBO`
/// over rows and `(T, 2) stride LBO` over K (`T = 16B/elt`).
pub struct UmmaKMajor<S = NoSwizzle> {
    shape: Shape,
    dtype: DType,
    _swizzle: PhantomData<S>,
}

impl UmmaKMajor<NoSwizzle> {
    pub fn new(shape: Shape, dtype: DType) -> Self {
        assert_eq!(shape.rank(), 2, "UmmaKMajor expects a rank-2 [M, K] shape");
        UmmaKMajor {
            shape,
            dtype,
            _swizzle: PhantomData,
        }
    }
}

impl<S> UmmaKMajor<S> {
    /// LBO for SWIZZLE_NONE K-major: one core matrix row-block, 8 x 16B.
    pub const LBO: u64 = 128;
    /// SBO for SWIZZLE_NONE K-major: two core matrices, 256B.
    pub const SBO: u64 = 256;

    /// Byte size of the whole tile.
    pub fn bytes(&self) -> u64 {
        self.shape.elems() * self.dtype.bytes()
    }

    /// Descriptor units (16B) to advance the K start-address by `k` halves.
    ///
    /// Regression pin for the reference kernel's `da += 8` bug: one K16 MMA
    /// step consumes TWO 16B core matrices (16 halves x 2B = 32B per row),
    /// i.e. 256B == 16 descriptor units, not 8.
    pub fn k_advance_units(&self, k: u64) -> u64 {
        let core_matrices = k * self.dtype.bytes() / 16;
        (core_matrices * Self::LBO) >> 4
    }

    /// UMMA smem descriptor for the tile placed at `smem_addr`.
    pub fn descriptor(&self, smem_addr: u64) -> Descriptor {
        Descriptor {
            start_addr: smem_addr,
            lbo: Self::LBO,
            sbo: Self::SBO,
            version: 1,
            swizzle_mode: 0,
        }
    }

    /// The 4D TMA tensor-map view whose linear fill order reproduces this
    /// exact smem layout (see [`TmaView::smem_offset`]).
    ///
    /// Derivation for an `[M, K]` K-major tile (K contiguous in gmem,
    /// row stride = K elements):
    ///   dim0: one 16B core-matrix row       = 16B/elt elements
    ///   dim1: 8 rows of a core matrix       , stride K*elt bytes
    ///   dim2: K covered in 16B chunks       = K*elt/16, stride 8*elt bytes
    ///   dim3: M covered in 8-row blocks     = M/8       , stride 8*K*elt bytes
    pub fn tma_view(&self) -> TmaView {
        let elt = self.dtype.bytes();
        let m = self.shape.0[0];
        let k = self.shape.0[1];
        let gdim = [16 / elt, 8, k * elt / 16, m / 8];
        let gstrd = [k * elt, 8 * elt, 8 * k * elt];
        TmaView {
            rank: 4,
            global_dim: gdim,
            box_dim: gdim,
            global_stride_bytes: gstrd,
            swizzle: Swizzle::None,
            elt_bytes: elt,
        }
    }
}

/// UMMA instruction descriptor for `kind::f16`, dense (`make_idesc_f16`):
/// `c_format = F32(1)<<4 | n_dim = (N>>3)<<17 | m_dim = (M>>4)<<24`.
pub fn idesc_f16(m: u32, n: u32) -> u32 {
    (1 << 4) | ((n >> 3) << 17) | ((m >> 4) << 24)
}

/// TMEM lane-field addressing: warp `w` owns lanes `[32w, 32w+32)` and the
/// lane base lives at bit 16 of the TMEM address, i.e. `w << 21`.
pub fn tmem_lane_offset(warp: u32) -> u32 {
    warp << 21
}
