//! Trace IR: the typed builder that user kernels run against (M1b).
//!
//! Kernel code is ordinary Rust; every `Val`-like handle carries an IR name
//! and a reference back to the [`Trace`], so calling a capability method on a
//! role token appends an op to the IR. Role capability is enforced by the
//! type system: `tma_load`/`expect_tx` exist only on [`LoadThread`], `umma_*`
//! only on [`MmaThread`], `tmem_ld`/`store_v4` only on [`EpiWarp`].

use std::any::type_name;
use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Add;
use std::rc::Rc;

use crate::layout::{NoSwizzle, Shape, UmmaKMajor};

/// Shared handle to the trace being built. Implementation detail of the
/// builder; named in public signatures so role tokens can record ops.
#[doc(hidden)]
pub type TraceRef = Rc<RefCell<Trace>>;

/// Dialect version stamped on every kernel (PROPOSAL §6: versioned from day
/// one so lowering-pass churn fails loudly).
pub const VERSION: &str = "0.1";

/// Target architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Sm100a,
}

impl Arch {
    pub fn mlir(self) -> &'static str {
        match self {
            Arch::Sm100a => "sm_100a",
        }
    }
}

// ---------------------------------------------------------------------------
// Trace-time errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Marker traits and types used in kernel signatures
// ---------------------------------------------------------------------------

/// Element dtype at the DSL type level.
pub trait DType {
    const MLIR: &'static str;
    const BYTES: u64;
}

#[allow(non_camel_case_types)]
pub struct f16;
#[allow(non_camel_case_types)]
pub struct f32;

impl DType for f16 {
    const MLIR: &'static str = "f16";
    const BYTES: u64 = 2;
}
impl DType for f32 {
    const MLIR: &'static str = "f32";
    const BYTES: u64 = 4;
}

/// Memory space.
pub trait Space {
    const MLIR: &'static str;
}

pub struct Gmem;
pub struct Smem;

impl Space for Gmem {
    const MLIR: &'static str = "gmem";
}
impl Space for Smem {
    const MLIR: &'static str = "smem";
}

/// A smem layout that has a dialect attribute spelling. Layouts that print
/// through the `#lay` alias report their definition here so the printer can
/// emit the alias line once at the top of the module.
pub trait MlirLayout {
    fn mlir_ref() -> &'static str;
    fn mlir_def() -> Option<&'static str>;
}

impl MlirLayout for UmmaKMajor<NoSwizzle> {
    fn mlir_ref() -> &'static str {
        "#lay"
    }
    fn mlir_def() -> Option<&'static str> {
        Some("#patina.umma_k_major<swizzle_none>")
    }
}

// ---------------------------------------------------------------------------
// IR ops
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) enum Op {
    /// Explicit blank line in the printed dialect.
    Blank,
    SmemAlloc {
        res: String,
        ty: String,
    },
    MBarrierCreate {
        res: String,
        arrivals: usize,
        ty: String,
    },
    TmemAlloc {
        res: String,
        cols: usize,
        ty: String,
    },
    WarpSpecializeBegin,
    RoleBegin(String),
    RoleEnd,
    WarpSpecializeEnd,
    ExpectTx {
        bar: String,
        bytes: u64,
    },
    TmaLoad {
        map: String,
        dst: String,
        coords: Vec<u64>,
        bar: String,
    },
    Wait {
        bar: String,
        phase: u32,
    },
    UmmaBegin {
        res: String,
        acc: String,
        a: String,
        b: String,
    },
    UmmaStep {
        m: String,
        accum: bool,
    },
    UmmaAdvanceK {
        m: String,
        k: u32,
    },
    UmmaCommit {
        m: String,
        bar: String,
    },
    TmemLd {
        res: String,
        acc: String,
        ty: String,
    },
    Store {
        ptr: String,
        frag: String,
    },
    TmemDealloc {
        acc: String,
    },
}

impl Op {
    /// Blank-line grouping: a blank is inserted between two adjacent
    /// top-level ops of different groups. Role-region contents and region
    /// markers share group 0 and manage their blanks explicitly.
    fn group(&self) -> u8 {
        match self {
            Op::SmemAlloc { .. } => 1,
            Op::MBarrierCreate { .. } => 2,
            Op::TmemAlloc { .. } => 3,
            Op::WarpSpecializeBegin => 4,
            Op::TmemDealloc { .. } => 5,
            _ => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// The trace itself
// ---------------------------------------------------------------------------

#[derive(Debug)]
#[doc(hidden)]
pub struct Trace {
    pub(crate) kernel_name: String,
    pub(crate) arch: Arch,
    pub(crate) block: u32,
    /// (role name, warp count), always in load/mma/epi order.
    pub(crate) roles: Vec<(String, usize)>,
    /// (name, dialect type) kernel parameters, in signature order.
    pub(crate) params: Vec<(String, String)>,
    pub(crate) ops: Vec<Op>,
    pub(crate) lay_alias: bool,
    smem_n: usize,
    mbar_n: usize,
    tmem_n: usize,
    umma_n: usize,
    frag_n: usize,
    param_n: std::collections::HashMap<&'static str, usize>,
}

impl Trace {
    fn new(kernel_name: String, arch: Arch, block: u32) -> Self {
        Trace {
            kernel_name,
            arch,
            block,
            roles: Vec::new(),
            params: Vec::new(),
            ops: Vec::new(),
            lay_alias: false,
            smem_n: 0,
            mbar_n: 0,
            tmem_n: 0,
            umma_n: 0,
            frag_n: 0,
            param_n: std::collections::HashMap::new(),
        }
    }

    /// Push a top-level op, inserting a blank line when the op group
    /// changes (e.g. between the smem allocs and the mbarrier creates).
    fn push_top(&mut self, op: Op) {
        if let Some(last) = self.ops.last() {
            if !matches!(last, Op::Blank) && last.group() != op.group() {
                self.ops.push(Op::Blank);
            }
        }
        self.ops.push(op);
    }

    fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    fn next_smem(&mut self) -> String {
        let n = self.smem_n;
        self.smem_n += 1;
        format!("%s{}", (b'a' + n as u8) as char)
    }

    fn next_mbar(&mut self) -> String {
        let n = self.mbar_n;
        self.mbar_n += 1;
        match n {
            0 => "%load_done".to_string(),
            1 => "%mma_done".to_string(),
            _ => format!("%bar{}", n),
        }
    }

    fn next_tmem(&mut self) -> String {
        let n = self.tmem_n;
        self.tmem_n += 1;
        if n == 0 {
            "%acc".to_string()
        } else {
            format!("%acc{}", n)
        }
    }

    fn next_umma(&mut self) -> String {
        let n = self.umma_n;
        self.umma_n += 1;
        if n == 0 {
            "%m".to_string()
        } else {
            format!("%m{}", n)
        }
    }

    fn next_frag(&mut self) -> String {
        let n = self.frag_n;
        self.frag_n += 1;
        if n == 0 {
            "%frag".to_string()
        } else {
            format!("%frag{}", n)
        }
    }

    fn param_count(&mut self, base: &'static str) -> usize {
        let n = *self.param_n.get(base).unwrap_or(&0);
        self.param_n.insert(base, n + 1);
        n
    }
}

// ---------------------------------------------------------------------------
// Value handles
// ---------------------------------------------------------------------------

/// Coordinate literal: `coord![0, 0, 0, 0]`.
#[macro_export]
macro_rules! coord {
    ($($c:expr),+ $(,)?) => {
        $crate::trace::Coord::from(vec![$($c as u64),+])
    };
}
pub use crate::coord;

/// TMA copy coordinates (4D for the canonical view).
#[derive(Clone, Debug)]
pub struct Coord(pub Vec<u64>);

impl From<Vec<u64>> for Coord {
    fn from(v: Vec<u64>) -> Self {
        Coord(v)
    }
}

/// An mbarrier phase parity — tracked, never a raw literal.
#[derive(Clone, Copy, Debug)]
pub struct Phase(pub u32);

pub fn phase(p: u32) -> Phase {
    Phase(p)
}

/// Accumulate flag for one UMMA K16 step (`scaleC` in the PTX).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accum {
    No,
    Yes,
}

/// TMA tensor-map kernel parameter (host-encoded, `__grid_constant__`).
pub struct TmaMap<T: DType, L: MlirLayout> {
    name: String,
    _pd: PhantomData<(T, L)>,
}

/// Typed pointer kernel parameter.
pub struct Ptr<T: DType, S: Space> {
    name: String,
    _pd: PhantomData<(T, S)>,
}

/// Row offset derived from the executing warp's TMEM lane base. Trace-time
/// symbolic: the lane math (`warp_rank << 21`) is derived at lowering.
#[derive(Clone, Copy, Debug)]
pub struct LaneRow(pub u64);

impl<T: DType, S: Space> Add<LaneRow> for Ptr<T, S> {
    type Output = Ptr<T, S>;
    fn add(self, _row: LaneRow) -> Ptr<T, S> {
        self
    }
}

/// A typed smem tensor; layout is part of the type.
pub struct SmemTensor<T: DType, L: MlirLayout> {
    name: String,
    shape: Shape,
    _pd: PhantomData<(T, L)>,
}

impl<T: DType, L: MlirLayout> SmemTensor<T, L> {
    /// Byte size of the tile — feeds `expect_tx`.
    pub fn bytes(&self) -> u64 {
        self.shape.elems() * T::BYTES
    }
}

/// MBarrier with arrival count in the type.
pub struct MBarrier<const ARRIVALS: usize> {
    name: String,
    trace: TraceRef,
}

impl<const ARRIVALS: usize> MBarrier<ARRIVALS> {
    /// `mbarrier.try_wait.parity` loop, phase parity from the typed Phase.
    pub fn wait(&self, p: Phase) {
        self.trace.borrow_mut().push(Op::Wait {
            bar: self.name.clone(),
            phase: p.0,
        });
    }
}

/// TMEM accumulator allocation. RAII: dropping emits the
/// fence/`bar.sync`/`tcgen05.dealloc` sequence (v1: the dealloc op).
pub struct Tmem<T: DType, const COLS: usize> {
    name: String,
    trace: TraceRef,
    _pd: PhantomData<T>,
}

impl<T: DType, const COLS: usize> Drop for Tmem<T, COLS> {
    fn drop(&mut self) {
        self.trace.borrow_mut().push_top(Op::TmemDealloc {
            acc: self.name.clone(),
        });
    }
}

/// In-flight UMMA chain (`begin/step/advance_k/commit`).
pub struct Umma {
    name: String,
    trace: TraceRef,
}

impl Umma {
    /// One K16 `tcgen05.mma`; `Accum::No` zeroes the accumulator input
    /// (scaleC = 0) on the first step.
    pub fn step(&mut self, accum: Accum) {
        self.trace.borrow_mut().push(Op::UmmaStep {
            m: self.name.clone(),
            accum: accum == Accum::Yes,
        });
    }

    /// Advance the A/B descriptor start addresses by K halves. The unit
    /// count comes from the layout (`k_advance_units`), never a literal.
    pub fn advance_k<const K: u32>(&mut self) {
        self.trace.borrow_mut().push(Op::UmmaAdvanceK {
            m: self.name.clone(),
            k: K,
        });
    }

    /// `tcgen05.commit` — arrival on the given barrier.
    pub fn commit<const N: usize>(&mut self, bar: &MBarrier<N>) {
        self.trace.borrow_mut().push(Op::UmmaCommit {
            m: self.name.clone(),
            bar: bar.name.clone(),
        });
    }
}

/// Register fragment loaded back from TMEM by the epilogue warps.
pub struct Frag<T: DType, const N: usize> {
    name: String,
    _pd: PhantomData<T>,
}

// ---------------------------------------------------------------------------
// Execution scopes: kernel/block ctx and role tokens
// ---------------------------------------------------------------------------

/// Root trace context handed to the kernel function.
pub struct KernelCtx {
    pub(crate) trace: TraceRef,
}

impl KernelCtx {
    /// The CTA body. Trace-time errors (e.g. `tmem_alloc`) panic here:
    /// a failed trace means the kernel is malformed, not recoverable.
    pub fn block(&mut self, f: impl FnOnce(&BlockCtx) -> Result<()>) {
        let blk = BlockCtx {
            trace: Rc::clone(&self.trace),
        };
        if let Err(e) = f(&blk) {
            panic!("patina trace failed: {}", e);
        }
    }
}

/// CTA-scope ops: allocations and the warp-specialized dispatch.
pub struct BlockCtx {
    trace: TraceRef,
}

impl BlockCtx {
    /// Static smem allocation (`cvta.to.shared` at lowering).
    pub fn smem<T: DType, S: Into<Shape>, L: MlirLayout>(&self, shape: S) -> SmemTensor<T, L> {
        let shape = shape.into();
        let mut tr = self.trace.borrow_mut();
        if L::mlir_def().is_some() {
            tr.lay_alias = true;
        }
        let name = tr.next_smem();
        let ty = format!(
            "!patina.tensor<{}, {}, {}, {}>",
            T::MLIR,
            shape.mlir(),
            Smem::MLIR,
            L::mlir_ref()
        );
        tr.push_top(Op::SmemAlloc {
            res: name.clone(),
            ty,
        });
        SmemTensor {
            name,
            shape,
            _pd: PhantomData,
        }
    }

    /// `mbarrier.init` + `fence.mbarrier_init`, arrival count in the type.
    pub fn mbarrier<const N: usize>(&self) -> MBarrier<N> {
        let mut tr = self.trace.borrow_mut();
        let name = tr.next_mbar();
        tr.push_top(Op::MBarrierCreate {
            res: name.clone(),
            arrivals: N,
            ty: format!("!patina.mbarrier<{}>", N),
        });
        MBarrier {
            name,
            trace: Rc::clone(&self.trace),
        }
    }

    /// Warp-collective `tcgen05.alloc` + `relinquish_alloc_permit`.
    pub fn tmem_alloc<T: DType, const COLS: usize>(&self) -> Result<Tmem<T, COLS>> {
        if !(32..=512).contains(&COLS) || !COLS.is_power_of_two() {
            return Err(Error(format!(
                "tmem_alloc: cols must be a power of two in 32..=512, got {}",
                COLS
            )));
        }
        let mut tr = self.trace.borrow_mut();
        let name = tr.next_tmem();
        tr.push_top(Op::TmemAlloc {
            res: name.clone(),
            cols: COLS,
            ty: format!("!patina.tmem<{}, {}>", T::MLIR, COLS),
        });
        Ok(Tmem {
            name,
            trace: Rc::clone(&self.trace),
            _pd: PhantomData,
        })
    }

    /// The one structured `switch (warp_id)`: role regions in registration
    /// order, warps auto-packed from the [`Roles`] declaration.
    pub fn warp_specialize<Rs: Roles>(&self, f: impl FnOnce(WarpSpecialize<Rs>)) {
        {
            let mut tr = self.trace.borrow_mut();
            tr.push_top(Op::WarpSpecializeBegin);
            tr.push(Op::Blank);
        }
        let ws = WarpSpecialize::<Rs> {
            trace: Rc::clone(&self.trace),
            _pd: PhantomData,
        };
        f(ws);
        self.trace.borrow_mut().push(Op::WarpSpecializeEnd);
    }
}

/// Registrar for role regions inside `patina.warp_specialize`.
pub struct WarpSpecialize<Rs: Roles> {
    trace: TraceRef,
    _pd: PhantomData<Rs>,
}

impl<Rs: Roles> WarpSpecialize<Rs> {
    /// Open the region for role `R` and run `f` with its capability token.
    pub fn role<R: RoleOf<Rs>>(&self, f: impl FnOnce(R::Token)) {
        {
            let mut tr = self.trace.borrow_mut();
            if matches!(tr.ops.last(), Some(Op::RoleEnd)) {
                tr.push(Op::Blank);
            }
            tr.push(Op::RoleBegin(R::NAME.to_string()));
        }
        f(R::token(Rc::clone(&self.trace)));
        self.trace.borrow_mut().push(Op::RoleEnd);
    }
}

/// Load role token: one elected thread of warp 0. Has `tma_load` +
/// `expect_tx`; lacks `umma`, `tmem_ld`, `tmem_alloc` (compile error).
pub struct LoadThread {
    trace: TraceRef,
}

impl LoadThread {
    /// Publish the expected transaction bytes; must equal the sum of the
    /// bytes of every `tma_load` completing on this barrier (dialect
    /// verifier enforces the match).
    pub fn expect_tx<const N: usize>(&self, bar: &MBarrier<N>, bytes: u64) {
        self.trace.borrow_mut().push(Op::ExpectTx {
            bar: bar.name.clone(),
            bytes,
        });
    }

    /// `cp.async.bulk.tensor.4d` from `map` into `dst`, completing on `bar`.
    pub fn tma_load<T: DType, L: MlirLayout, const N: usize>(
        &self,
        map: &TmaMap<T, L>,
        dst: &SmemTensor<T, L>,
        coords: Coord,
        bar: &MBarrier<N>,
    ) {
        self.trace.borrow_mut().push(Op::TmaLoad {
            map: map.name.clone(),
            dst: dst.name.clone(),
            coords: coords.0,
            bar: bar.name.clone(),
        });
    }
}

/// MMA role token: one elected thread of the MMA warp. Has `umma_*` +
/// `commit`; lacks `tma_load`, `tmem_ld`.
pub struct MmaThread {
    trace: TraceRef,
}

impl MmaThread {
    /// Begin a UMMA chain (f16 in, f32 acc, K-major). The instruction
    /// descriptor is packed by lowering from the operand types; the smem
    /// descriptors (LBO/SBO) come from the layout in the tensor types.
    pub fn umma_f16<const COLS: usize, L: MlirLayout>(
        &self,
        acc: &Tmem<f32, COLS>,
        a: &SmemTensor<f16, L>,
        b: &SmemTensor<f16, L>,
    ) -> Umma {
        let mut tr = self.trace.borrow_mut();
        let name = tr.next_umma();
        tr.push(Op::UmmaBegin {
            res: name.clone(),
            acc: acc.name.clone(),
            a: a.name.clone(),
            b: b.name.clone(),
        });
        Umma {
            name,
            trace: Rc::clone(&self.trace),
        }
    }
}

/// Epilogue role token: a full warp. Has `tmem_ld`/`store_v4` + lane math;
/// lacks `umma`, `tma_load`.
pub struct EpiWarp {
    trace: TraceRef,
}

impl EpiWarp {
    /// `tcgen05.ld.32x32b.x32` + `wait::ld`; N columns per row of the
    /// accumulator.
    pub fn tmem_ld<T: DType, const COLS: usize>(&self, acc: &Tmem<T, COLS>) -> Frag<T, COLS> {
        let mut tr = self.trace.borrow_mut();
        let name = tr.next_frag();
        tr.push(Op::TmemLd {
            res: name.clone(),
            acc: acc.name.clone(),
            ty: format!("!patina.frag<{}, {}>", T::MLIR, COLS),
        });
        Frag {
            name,
            _pd: PhantomData,
        }
    }

    /// This warp's row offset into an `N`-row tile, derived from the warp's
    /// TMEM lane base.
    pub fn lane_row<const N: usize>(&self) -> LaneRow {
        LaneRow(N as u64)
    }

    /// Coalesced `st.global.v4` of one register fragment.
    pub fn store_v4<T: DType, const N: usize>(&self, ptr: Ptr<T, Gmem>, frag: &Frag<T, N>) {
        self.trace.borrow_mut().push(Op::Store {
            ptr: ptr.name,
            frag: frag.name.clone(),
        });
    }
}

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

/// A role with a const-generic warp count.
pub trait RoleSpec {
    const WARPS: usize;
}

pub struct Role<const N: usize>;

impl<const N: usize> RoleSpec for Role<N> {
    const WARPS: usize = N;
}

/// The role set of a kernel: which role gets how many warps.
pub trait Roles {
    type Load: RoleSpec;
    type Mma: RoleSpec;
    type Epi: RoleSpec;
}

/// Role selectors used with [`WarpSpecialize::role`].
pub struct Load;
pub struct Mma;
pub struct Epi;

/// Connects a role selector to a [`Roles`] set: name, warp count, and the
/// capability token its region closure receives.
pub trait RoleOf<Rs: Roles> {
    type Token;
    const NAME: &'static str;
    const WARPS: usize;
    fn token(trace: TraceRef) -> Self::Token;
}

impl<Rs: Roles> RoleOf<Rs> for Load {
    type Token = LoadThread;
    const NAME: &'static str = "load";
    const WARPS: usize = <Rs::Load as RoleSpec>::WARPS;
    fn token(trace: TraceRef) -> LoadThread {
        LoadThread { trace }
    }
}

impl<Rs: Roles> RoleOf<Rs> for Mma {
    type Token = MmaThread;
    const NAME: &'static str = "mma";
    const WARPS: usize = <Rs::Mma as RoleSpec>::WARPS;
    fn token(trace: TraceRef) -> MmaThread {
        MmaThread { trace }
    }
}

impl<Rs: Roles> RoleOf<Rs> for Epi {
    type Token = EpiWarp;
    const NAME: &'static str = "epi";
    const WARPS: usize = <Rs::Epi as RoleSpec>::WARPS;
    fn token(trace: TraceRef) -> EpiWarp {
        EpiWarp { trace }
    }
}

// ---------------------------------------------------------------------------
// Kernel tracing entry point
// ---------------------------------------------------------------------------

/// A kernel parameter type: knows its dialect type spelling and how to
/// register itself (name + type) with the trace.
pub trait KernelArg: Sized {
    fn register(tr: &mut Trace) -> Self;
}

impl<T: DType, L: MlirLayout> KernelArg for TmaMap<T, L> {
    fn register(tr: &mut Trace) -> Self {
        if L::mlir_def().is_some() {
            tr.lay_alias = true;
        }
        let n = tr.param_count("tma");
        let name = format!("%tma_{}", (b'a' + n as u8) as char);
        tr.params.push((
            name.clone(),
            format!("!patina.tma_map<{}, {}>", T::MLIR, L::mlir_ref()),
        ));
        TmaMap {
            name,
            _pd: PhantomData,
        }
    }
}

impl<T: DType, S: Space> KernelArg for Ptr<T, S> {
    fn register(tr: &mut Trace) -> Self {
        let n = tr.param_count("c");
        let name = if n == 0 {
            "%c".to_string()
        } else {
            format!("%c{}", n)
        };
        tr.params
            .push((name.clone(), format!("!patina.ptr<{}, {}>", T::MLIR, S::MLIR)));
        Ptr {
            name,
            _pd: PhantomData,
        }
    }
}

/// A tuple of kernel parameters.
pub trait KernelArgs: Sized {
    fn make(tr: &mut Trace) -> Self;
}

/// A traceable kernel function: `fn(&mut KernelCtx, args...)`.
pub trait FnKernel<A> {
    fn invoke(self, ctx: &mut KernelCtx, args: A);
}

macro_rules! impl_kernel_arity {
    ($($T:ident),+) => {
        impl<$($T: KernelArg),+> KernelArgs for ($($T,)+) {
            fn make(tr: &mut Trace) -> Self {
                ($($T::register(tr),)+)
            }
        }
        impl<F, $($T),+> FnKernel<($($T,)+)> for F
        where
            F: FnOnce(&mut KernelCtx, $($T),+),
        {
            fn invoke(self, ctx: &mut KernelCtx, args: ($($T,)+)) {
                #[allow(non_snake_case)]
                let ($($T,)+) = args;
                self(ctx, $($T),+)
            }
        }
    };
}

impl_kernel_arity!(A);
impl_kernel_arity!(A, B);
impl_kernel_arity!(A, B, C);
impl_kernel_arity!(A, B, C, D);

/// Entry point: `Kernel::trace(kernel_fn, Arch::Sm100a)`.
pub struct Kernel;

impl Kernel {
    pub fn trace<F, A>(f: F, arch: Arch) -> TraceBuilder<F, A>
    where
        F: FnKernel<A>,
        A: KernelArgs,
    {
        TraceBuilder {
            f,
            arch,
            _pd: PhantomData,
        }
    }
}

pub struct TraceBuilder<F, A> {
    f: F,
    arch: Arch,
    _pd: PhantomData<A>,
}

impl<F, A> TraceBuilder<F, A>
where
    F: FnKernel<A>,
    A: KernelArgs,
{
    pub fn roles<Rs: Roles>(self) -> RolesBuilder<F, A, Rs> {
        RolesBuilder {
            f: self.f,
            arch: self.arch,
            _pd: PhantomData,
        }
    }
}

pub struct RolesBuilder<F, A, Rs> {
    f: F,
    arch: Arch,
    _pd: PhantomData<(A, Rs)>,
}

impl<F, A, Rs> RolesBuilder<F, A, Rs>
where
    F: FnKernel<A>,
    A: KernelArgs,
    Rs: Roles,
{
    /// Set the CTA thread count and run the trace. The role warp budget
    /// must exactly fill the block.
    pub fn block(self, threads: u32) -> TracedKernel {
        let warps = Rs::Load::WARPS + Rs::Mma::WARPS + Rs::Epi::WARPS;
        if warps * 32 != threads as usize {
            panic!(
                "patina: roles require {} warps ({} threads) but block = {}",
                warps,
                warps * 32,
                threads
            );
        }
        let name = type_name::<F>()
            .rsplit("::")
            .next()
            .unwrap_or("kernel")
            .to_string();
        let mut tr = Trace::new(name, self.arch, threads);
        tr.roles = vec![
            ("load".to_string(), Rs::Load::WARPS),
            ("mma".to_string(), Rs::Mma::WARPS),
            ("epi".to_string(), Rs::Epi::WARPS),
        ];
        let trace = Rc::new(RefCell::new(tr));
        let args = A::make(&mut trace.borrow_mut());
        let mut ctx = KernelCtx {
            trace: Rc::clone(&trace),
        };
        self.f.invoke(&mut ctx, args);
        TracedKernel { trace }
    }
}

/// A fully traced kernel, ready to emit.
pub struct TracedKernel {
    pub(crate) trace: TraceRef,
}

impl TracedKernel {
    /// Emit the Patina dialect MLIR text for this kernel.
    pub fn emit_dialect(&self) -> String {
        crate::emit::emit(&self.trace.borrow())
    }
}
