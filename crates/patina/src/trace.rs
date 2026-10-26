//! Trace IR: the typed builder that user kernels run against (M1b, M3).
//!
//! Kernel code is ordinary Rust; every `Val`-like handle carries an IR name
//! and a reference back to the [`Trace`], so calling a capability method on a
//! role token appends an op to the IR. Role capability is enforced by the
//! type system: `tma_load`/`expect_tx` exist only on [`LoadThread`], `umma_*`
//! only on [`MmaThread`], `tmem_ld`/`store_v4` only on [`EpiWarp`].
//!
//! M3 adds the pipelined-mainloop surface: [`Pipeline`] ring buffers, runtime
//! `for_each` loops with loop-carried [`PipeStage`] (stage index + phase as
//! `!patina.u32` values), staged TMA/UMMA barrier forms, and scalar ops
//! ([`LoopIx::muli`], [`LoopIx::nez`]).

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

/// Which barrier array of a pipeline a staged synchronization op targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StageKind {
    Empty,
    Full,
}

impl StageKind {
    pub(crate) fn mlir(self) -> &'static str {
        match self {
            StageKind::Empty => "empty",
            StageKind::Full => "full",
        }
    }
}

/// Completion target of a staged-capable op (`tma_load`, `umma.commit`):
/// either a standalone mbarrier (v1) or a `(pipeline, kind, stage)` triple.
#[derive(Clone, Debug)]
pub(crate) enum StageTarget {
    Bar(String),
    Stage { pipe: String, kind: StageKind, s: String },
}

/// The scaleC operand of one UMMA K16 step: a compile-time flag (v1) or a
/// runtime scalar (M3 pipelined mainloop: `i != 0`).
#[derive(Clone, Debug)]
pub(crate) enum AccumOp {
    Static(bool),
    Dyn(String),
}

#[derive(Debug)]
pub(crate) enum Op {
    /// Explicit blank line in the printed dialect.
    Blank,
    PipelineCreate {
        res: String,
        stages: usize,
        ty: String,
    },
    SmemAlloc {
        res: String,
        stages: Option<usize>,
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
    ForBegin {
        i: String,
        bound: String,
        s: String,
        ph: String,
        init_ph: u32,
    },
    Yield {
        vals: Vec<String>,
    },
    ForEnd,
    WaitStage {
        pipe: String,
        kind: StageKind,
        s: String,
        ph: String,
    },
    ExpectTxStage {
        pipe: String,
        kind: StageKind,
        s: String,
        bytes: u64,
    },
    StageBuf {
        res: String,
        base: String,
        s: String,
        ty: String,
    },
    StageNext {
        res_s: String,
        res_ph: String,
        s: String,
        ph: String,
        pipe: String,
    },
    Muli {
        res: String,
        x: String,
        k: u32,
    },
    Nez {
        res: String,
        x: String,
    },
    ExpectTx {
        bar: String,
        bytes: u64,
    },
    TmaLoad {
        map: String,
        dst: String,
        coords: Vec<CoordElem>,
        complete: StageTarget,
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
        accum: AccumOp,
    },
    UmmaAdvanceK {
        m: String,
        k: u32,
    },
    UmmaCommit {
        m: String,
        target: StageTarget,
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
            Op::PipelineCreate { .. } => 1,
            Op::SmemAlloc { .. } => 2,
            Op::MBarrierCreate { .. } => 3,
            Op::TmemAlloc { .. } => 4,
            Op::WarpSpecializeBegin => 5,
            Op::TmemDealloc { .. } => 6,
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
    pipe_n: usize,
    cont_n: usize,
    /// Name of the most recently begun UMMA chain (for `commit_tail`).
    last_umma: Option<String>,
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
            pipe_n: 0,
            cont_n: 0,
            last_umma: None,
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

    fn next_pipe(&mut self) -> String {
        let n = self.pipe_n;
        self.pipe_n += 1;
        if n == 0 {
            "%pipe".to_string()
        } else {
            format!("%pipe{}", n)
        }
    }

    fn next_cont(&mut self) -> String {
        let n = self.cont_n;
        self.cont_n += 1;
        if n == 0 {
            "%cont".to_string()
        } else {
            format!("%cont{}", n)
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

/// Coordinate list: `coord![0, 0, 0, 0]`, optionally with runtime scalars
/// (`coord![0, 0, i4, 0]` where `i4` comes from [`LoopIx::muli`]).
#[macro_export]
macro_rules! coord {
    ($($c:expr),+ $(,)?) => {
        $crate::trace::Coord::from_elems(vec![
            $($crate::trace::CoordElem::from(($c).clone())),+
        ])
    };
}
pub use crate::coord;

/// One TMA coordinate: a compile-time constant or a runtime scalar value.
#[derive(Clone, Debug)]
pub enum CoordElem {
    Const(u64),
    Dyn(String),
}

impl From<i32> for CoordElem {
    fn from(v: i32) -> Self {
        CoordElem::Const(v as u64)
    }
}

impl From<Scalar> for CoordElem {
    fn from(v: Scalar) -> Self {
        CoordElem::Dyn(v.name)
    }
}

/// TMA copy coordinates (4D for the canonical view).
#[derive(Clone, Debug)]
pub struct Coord(pub Vec<CoordElem>);

impl Coord {
    pub fn from_elems(elems: Vec<CoordElem>) -> Self {
        Coord(elems)
    }
}

impl From<Vec<u64>> for Coord {
    fn from(v: Vec<u64>) -> Self {
        Coord(v.into_iter().map(CoordElem::Const).collect())
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

/// A runtime `!patina.u32` scalar kernel parameter. Unlike trace-time Rust
/// values, a `Val` is an IR node: it appears in the kernel signature and can
/// bound a runtime loop.
pub struct Val<T> {
    name: String,
    _pd: PhantomData<T>,
}

/// A runtime scalar produced inside the kernel body (by [`LoopIx::muli`] or
/// [`LoopIx::nez`]); usable in TMA coordinates and as the dynamic UMMA
/// scaleC operand.
#[derive(Clone, Debug)]
pub struct Scalar {
    name: String,
}

/// The index of a runtime `patina.for` loop (zero-based, `!patina.u32`).
pub struct LoopIx {
    name: String,
    trace: TraceRef,
}

impl LoopIx {
    /// `patina.muli` — runtime multiply by a compile-time constant (e.g. the
    /// TMA K-coordinate advance per tile).
    pub fn muli<const K: u32>(&self) -> Scalar {
        let name = format!("{}{}", self.name, K);
        self.trace.borrow_mut().push(Op::Muli {
            res: name.clone(),
            x: self.name.clone(),
            k: K,
        });
        Scalar { name }
    }

    /// `patina.nez` — 1 while the index is nonzero; the dynamic scaleC of a
    /// pipelined mainloop (overwrite on the first tile, accumulate after).
    pub fn nez(&self) -> Scalar {
        let mut tr = self.trace.borrow_mut();
        let name = tr.next_cont();
        tr.push(Op::Nez {
            res: name.clone(),
            x: self.name.clone(),
        });
        Scalar { name }
    }
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

/// A typed smem tensor; layout is part of the type. Created directly by
/// [`BlockCtx::smem`] (single buffer) or staged by [`Pipeline::buf`].
pub struct SmemTensor<T: DType, L: MlirLayout> {
    name: String,
    ty: String,
    shape: Shape,
    stages: usize,
    _pd: PhantomData<(T, L)>,
}

impl<T: DType, L: MlirLayout> SmemTensor<T, L> {
    /// Byte size of one tile — feeds `expect_tx`/`publish`.
    pub fn bytes(&self) -> u64 {
        self.shape.elems() * T::BYTES
    }
}

/// Anything that can serve as a UMMA smem operand: a plain smem tensor or
/// one stage of a staged (pipelined) buffer.
pub trait UmmaOperand {
    #[doc(hidden)]
    fn ir_name(&self) -> &str;
}

impl<T: DType, L: MlirLayout> UmmaOperand for SmemTensor<T, L> {
    fn ir_name(&self) -> &str {
        &self.name
    }
}

/// The stage-`s` slice of a staged smem buffer (`patina.stage_buf`).
pub struct StageBuf<T: DType, L: MlirLayout> {
    name: String,
    _pd: PhantomData<(T, L)>,
}

impl<T: DType, L: MlirLayout> UmmaOperand for StageBuf<T, L> {
    fn ir_name(&self) -> &str {
        &self.name
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

/// An `S`-stage smem ring pipeline: owns `full[S]` + `empty[S]` mbarrier
/// arrays. Phases are runtime values, loop-carried with the stage index in
/// a [`PipeStage`].
///
/// Convention: the producer waits `empty[s]` with initial phase 1 and
/// publishes `full[s]`; the consumer waits `full[s]` with initial phase 0
/// and releases `empty[s]`. The phase flips when the stage wraps mod `S`.
pub struct Pipeline<const S: usize> {
    name: String,
    trace: TraceRef,
}

impl<const S: usize> Pipeline<S> {
    /// Allocate a staged smem buffer: `S` contiguous tiles of `shape`.
    pub fn buf<T: DType, L: MlirLayout, Sh: Into<Shape>>(&self, shape: Sh) -> SmemTensor<T, L> {
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
            stages: Some(S),
            ty: ty.clone(),
        });
        SmemTensor {
            name,
            ty,
            shape,
            stages: S,
            _pd: PhantomData,
        }
    }

    /// The stage-`st` slice of a staged buffer (`patina.stage_buf`).
    pub fn stage_buf<T: DType, L: MlirLayout>(
        &self,
        base: &SmemTensor<T, L>,
        st: &PipeStage,
    ) -> StageBuf<T, L> {
        if base.stages <= 1 {
            panic!(
                "patina: stage_buf requires a staged buffer (pipeline.buf(...)), got {}",
                base.name
            );
        }
        let name = format!("{}_s", base.name);
        self.trace.borrow_mut().push(Op::StageBuf {
            res: name.clone(),
            base: base.name.clone(),
            s: st.s.clone(),
            ty: base.ty.clone(),
        });
        StageBuf {
            name,
            _pd: PhantomData,
        }
    }
}

/// Loop-carried pipeline state: the ring stage index and phase parity, both
/// runtime `!patina.u32` values. Produced by `for_each` and threaded through
/// the loop body; the closure returns `st.next(pipe)` so the loop can yield
/// the advanced state.
pub struct PipeStage {
    s: String,
    ph: String,
    trace: TraceRef,
}

impl PipeStage {
    /// Advance the ring: stage +1 mod S, phase flip on wrap
    /// (`patina.stage_next`).
    pub fn next<const S: usize>(self, pipe: &Pipeline<S>) -> PipeStage {
        let res_s = format!("{}1", self.s);
        let res_ph = format!("{}1", self.ph);
        self.trace.borrow_mut().push(Op::StageNext {
            res_s: res_s.clone(),
            res_ph: res_ph.clone(),
            s: self.s.clone(),
            ph: self.ph.clone(),
            pipe: pipe.name.clone(),
        });
        PipeStage {
            s: res_s,
            ph: res_ph,
            trace: Rc::clone(&self.trace),
        }
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
            accum: AccumOp::Static(accum == Accum::Yes),
        });
    }

    /// One K16 step with a *runtime* scaleC (`accum %cond`); the pipelined
    /// mainloop passes `i.nez()` so the first tile overwrites and the rest
    /// accumulate.
    pub fn step_dyn(&mut self, cond: &Scalar) {
        self.trace.borrow_mut().push(Op::UmmaStep {
            m: self.name.clone(),
            accum: AccumOp::Dyn(cond.name.clone()),
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
            target: StageTarget::Bar(bar.name.clone()),
        });
    }

    /// `tcgen05.commit` arriving on the pipeline's `empty[s]` barrier —
    /// releases the consumed stage back to the producer.
    pub fn commit_stage<const S: usize>(&mut self, pipe: &Pipeline<S>, st: &PipeStage) {
        self.trace.borrow_mut().push(Op::UmmaCommit {
            m: self.name.clone(),
            target: StageTarget::Stage {
                pipe: pipe.name.clone(),
                kind: StageKind::Empty,
                s: st.s.clone(),
            },
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
            stages: None,
            ty: ty.clone(),
        });
        SmemTensor {
            name,
            ty,
            shape,
            stages: 1,
            _pd: PhantomData,
        }
    }

    /// Create an `S`-stage ring pipeline (full[S] + empty[S] barrier
    /// arrays, staged smem via [`Pipeline::buf`]).
    pub fn pipeline<const S: usize>(&self) -> Pipeline<S> {
        let mut tr = self.trace.borrow_mut();
        let name = tr.next_pipe();
        tr.push_top(Op::PipelineCreate {
            res: name.clone(),
            stages: S,
            ty: format!("!patina.pipeline<{}>", S),
        });
        Pipeline {
            name,
            trace: Rc::clone(&self.trace),
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

    /// A named mbarrier, for barriers outside the v1 load/mma pair (e.g.
    /// the pipelined kernel's accumulator-drain barrier `%acc_done`).
    pub fn mbarrier_named<const N: usize>(&self, name: &str) -> MBarrier<N> {
        let name = format!("%{}", name);
        self.trace.borrow_mut().push_top(Op::MBarrierCreate {
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

/// Shared implementation of the role-token `for_each`: emits
/// `patina.for %i = 0 to bound iter_args(%s = 0, %ph = init_ph)`, runs the
/// body, then yields the advanced stage state returned by the closure.
/// Loop value names restart per loop (`%i`, `%s`, `%ph`); region-local
/// shadowing keeps the dialect refs unambiguous.
fn for_each_impl(
    trace: &TraceRef,
    bound: &str,
    init_ph: u32,
    f: impl FnOnce(LoopIx, PipeStage) -> PipeStage,
) {
    trace.borrow_mut().push(Op::ForBegin {
        i: "%i".to_string(),
        bound: bound.to_string(),
        s: "%s".to_string(),
        ph: "%ph".to_string(),
        init_ph,
    });
    let i = LoopIx {
        name: "%i".to_string(),
        trace: Rc::clone(trace),
    };
    let st = PipeStage {
        s: "%s".to_string(),
        ph: "%ph".to_string(),
        trace: Rc::clone(trace),
    };
    let st = f(i, st);
    let mut tr = trace.borrow_mut();
    tr.push(Op::Yield {
        vals: vec![st.s, st.ph],
    });
    tr.push(Op::ForEnd);
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
            complete: StageTarget::Bar(bar.name.clone()),
        });
    }

    /// Runtime loop `for i in 0..bound` emitting `patina.for` with
    /// loop-carried pipeline state. Producer side: the stage phase starts
    /// at 1, so the first `empty[s]` acquisitions pass on fresh barriers.
    pub fn for_each(&self, bound: &Val<u32>, f: impl FnOnce(LoopIx, PipeStage) -> PipeStage) {
        for_each_impl(&self.trace, &bound.name, 1, f);
    }

    /// Wait the stage's `empty` barrier with the carried phase
    /// (`patina.wait_stage pipe, empty, s, ph`).
    pub fn producer_acquire<const S: usize>(&self, pipe: &Pipeline<S>, st: &PipeStage) {
        self.trace.borrow_mut().push(Op::WaitStage {
            pipe: pipe.name.clone(),
            kind: StageKind::Empty,
            s: st.s.clone(),
            ph: st.ph.clone(),
        });
    }

    /// Publish the expected transaction bytes on the stage's `full`
    /// barrier; must equal the sum of the staged `tma_load` bytes on it
    /// (dialect verifier enforces the match).
    pub fn publish<const S: usize>(&self, pipe: &Pipeline<S>, st: &PipeStage, bytes: u64) {
        self.trace.borrow_mut().push(Op::ExpectTxStage {
            pipe: pipe.name.clone(),
            kind: StageKind::Full,
            s: st.s.clone(),
            bytes,
        });
    }

    /// `cp.async.bulk.tensor.4d` into one stage of a staged buffer,
    /// completing on the pipeline's `full[s]` barrier.
    pub fn tma_load_stage<T: DType, L: MlirLayout, const S: usize>(
        &self,
        map: &TmaMap<T, L>,
        dst: &StageBuf<T, L>,
        coords: Coord,
        pipe: &Pipeline<S>,
        st: &PipeStage,
    ) {
        self.trace.borrow_mut().push(Op::TmaLoad {
            map: map.name.clone(),
            dst: dst.name.clone(),
            coords: coords.0,
            complete: StageTarget::Stage {
                pipe: pipe.name.clone(),
                kind: StageKind::Full,
                s: st.s.clone(),
            },
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
    /// Operands are plain smem tensors or pipeline stage buffers.
    pub fn umma_f16<const COLS: usize, A: UmmaOperand, B: UmmaOperand>(
        &self,
        acc: &Tmem<f32, COLS>,
        a: &A,
        b: &B,
    ) -> Umma {
        let mut tr = self.trace.borrow_mut();
        let name = tr.next_umma();
        tr.last_umma = Some(name.clone());
        tr.push(Op::UmmaBegin {
            res: name.clone(),
            acc: acc.name.clone(),
            a: a.ir_name().to_string(),
            b: b.ir_name().to_string(),
        });
        Umma {
            name,
            trace: Rc::clone(&self.trace),
        }
    }

    /// Runtime loop `for i in 0..bound` emitting `patina.for` with
    /// loop-carried pipeline state. Consumer side: the stage phase starts
    /// at 0.
    pub fn for_each(&self, bound: &Val<u32>, f: impl FnOnce(LoopIx, PipeStage) -> PipeStage) {
        for_each_impl(&self.trace, &bound.name, 0, f);
    }

    /// Wait the stage's `full` barrier with the carried phase
    /// (`patina.wait_stage pipe, full, s, ph`).
    pub fn consumer_acquire<const S: usize>(&self, pipe: &Pipeline<S>, st: &PipeStage) {
        self.trace.borrow_mut().push(Op::WaitStage {
            pipe: pipe.name.clone(),
            kind: StageKind::Full,
            s: st.s.clone(),
            ph: st.ph.clone(),
        });
    }

    /// Drain commit on a standalone barrier after the mainloop, referencing
    /// the most recently begun UMMA chain: the whole MMA group is complete
    /// once `bar` flips (the epilogue waits it).
    pub fn commit_tail<const N: usize>(&self, bar: &MBarrier<N>) {
        let mut tr = self.trace.borrow_mut();
        let m = tr
            .last_umma
            .clone()
            .expect("patina: commit_tail requires a prior umma chain");
        tr.push(Op::UmmaCommit {
            m,
            target: StageTarget::Bar(bar.name.clone()),
        });
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

impl KernelArg for Val<u32> {
    fn register(tr: &mut Trace) -> Self {
        let n = tr.param_count("k_tiles");
        let name = if n == 0 {
            "%k_tiles".to_string()
        } else {
            format!("%k_tiles{}", n)
        };
        tr.params.push((name.clone(), "!patina.u32".to_string()));
        Val {
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
