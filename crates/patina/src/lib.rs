//! Patina — a thin typed layer over tcgen05.
//!
//! Rust-embedded DSL that traces warp-specialized Blackwell kernels and
//! emits the Patina MLIR dialect. See `PROPOSAL.md`.

pub mod layout;
pub mod testing;
pub mod trace;

pub(crate) mod emit;

/// The full DSL surface: `use patina::prelude::*;`.
pub mod prelude {
    pub use crate::{coord, shape};
    pub use crate::layout::{NoSwizzle, Shape, UmmaKMajor};
    pub use crate::trace::{
        f16, f32, Accum, Arch, BlockCtx, Coord, DType, Epi, EpiWarp, Error, Frag, Kernel,
        KernelCtx, LaneRow, Load, LoadThread, MBarrier, Mma, MmaThread, Phase, Ptr, Result, Role,
        RoleOf, RoleSpec, Roles, SmemTensor, Space, TmaMap, Tmem, Umma, WarpSpecialize, Gmem,
        Smem, phase,
    };
    pub use crate::trace::{FnKernel, KernelArg, KernelArgs, MlirLayout, TraceBuilder,
        RolesBuilder, TracedKernel};
}
