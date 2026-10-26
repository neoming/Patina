//! Dialect printer: renders a [`Trace`] as Patina-dialect MLIR text.
//!
//! The printer emits no comments of its own; blank-line structure and
//! result-name alignment follow the trace (see `Op::Blank` and `Op::group`).

use crate::trace::{AccumOp, CoordElem, Op, StageTarget, Trace, VERSION};

/// The printed result text of an op (`%m`, or `%s1, %ph1` for the
/// two-result `patina.stage_next`).
fn result_name(op: &Op) -> Option<String> {
    match op {
        Op::PipelineCreate { res, .. }
        | Op::SmemAlloc { res, .. }
        | Op::MBarrierCreate { res, .. }
        | Op::TmemAlloc { res, .. }
        | Op::UmmaBegin { res, .. }
        | Op::TmemLd { res, .. }
        | Op::StageBuf { res, .. }
        | Op::Muli { res, .. }
        | Op::Nez { res, .. } => Some(res.clone()),
        Op::StageNext { res_s, res_ph, .. } => Some(format!("{}, {}", res_s, res_ph)),
        _ => None,
    }
}

/// Width to pad each result-bearing op's name to: the max name length of
/// its contiguous run of result ops (so adjacent `mbarrier.create` results
/// line up on `=`, but groups split by blank lines stay independent).
fn run_widths(ops: &[Op]) -> Vec<usize> {
    let mut widths = vec![0; ops.len()];
    let mut run: Vec<usize> = Vec::new();
    let flush = |run: &mut Vec<usize>, widths: &mut Vec<usize>| {
        if run.is_empty() {
            return;
        }
        let w = run
            .iter()
            .map(|&i| result_name(&ops[i]).unwrap().len())
            .max()
            .unwrap();
        for &i in run.iter() {
            widths[i] = w;
        }
        run.clear();
    };
    for (i, op) in ops.iter().enumerate() {
        if result_name(op).is_some() {
            run.push(i);
        } else {
            flush(&mut run, &mut widths);
        }
    }
    flush(&mut run, &mut widths);
    widths
}

fn coord_elems(coords: &[CoordElem]) -> String {
    coords
        .iter()
        .map(|c| match c {
            CoordElem::Const(v) => v.to_string(),
            CoordElem::Dyn(name) => name.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn stage_target(t: &StageTarget) -> String {
    match t {
        StageTarget::Bar(bar) => bar.clone(),
        StageTarget::Stage { pipe, kind, s } => format!("{}, {}, {}", pipe, kind.mlir(), s),
    }
}

pub(crate) fn emit(tr: &Trace) -> String {
    let mut out = String::new();

    if tr.lay_alias {
        out.push_str("#lay = #patina.umma_k_major<swizzle_none>\n\n");
    }

    out.push_str(&format!("patina.kernel @{}\n", tr.kernel_name));
    out.push_str(&format!(
        "    version = {}, arch = #patina.arch<{}>, block = {},\n",
        VERSION,
        tr.arch.mlir(),
        tr.block
    ));
    let roles = tr
        .roles
        .iter()
        .map(|(name, warps)| format!("#patina.role<{}, {}>", name, warps))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("    roles = [{}]\n", roles));

    if tr.params.is_empty() {
        out.push_str("  () {\n");
    } else {
        let width = tr
            .params
            .iter()
            .map(|(name, _)| name.len() + 1)
            .max()
            .unwrap();
        for (i, (name, ty)) in tr.params.iter().enumerate() {
            let prefix = if i == 0 { "  (" } else { "   " };
            let suffix = if i + 1 == tr.params.len() { ") {" } else { "," };
            out.push_str(&format!(
                "{}{:<width$} {}{}\n",
                prefix,
                format!("{}:", name),
                ty,
                suffix,
                width = width
            ));
        }
    }

    if !tr.ops.is_empty() {
        out.push('\n');
        let widths = run_widths(&tr.ops);
        let mut depth = 1usize;
        for (i, op) in tr.ops.iter().enumerate() {
            if matches!(op, Op::RoleEnd | Op::WarpSpecializeEnd | Op::ForEnd) {
                depth -= 1;
            }
            if matches!(op, Op::Blank) {
                out.push('\n');
                continue;
            }
            let indent = "  ".repeat(depth);
            let line = match op {
                Op::Blank => unreachable!(),
                Op::PipelineCreate { res, stages, ty } => format!(
                    "{}{:<w$} = patina.pipeline.create stages = {} : {}",
                    indent,
                    res,
                    stages,
                    ty,
                    w = widths[i]
                ),
                Op::SmemAlloc { res, stages, ty } => match stages {
                    Some(s) => format!(
                        "{}{:<w$} = patina.smem_alloc stages = {} : {}",
                        indent,
                        res,
                        s,
                        ty,
                        w = widths[i]
                    ),
                    None => format!(
                        "{}{:<w$} = patina.smem_alloc : {}",
                        indent,
                        res,
                        ty,
                        w = widths[i]
                    ),
                },
                Op::MBarrierCreate { res, arrivals, ty } => format!(
                    "{}{:<w$} = patina.mbarrier.create arrivals = {} : {}",
                    indent,
                    res,
                    arrivals,
                    ty,
                    w = widths[i]
                ),
                Op::TmemAlloc { res, cols, ty } => format!(
                    "{}{:<w$} = patina.tmem.alloc cols = {} : {}",
                    indent,
                    res,
                    cols,
                    ty,
                    w = widths[i]
                ),
                Op::WarpSpecializeBegin => format!("{}patina.warp_specialize {{", indent),
                Op::RoleBegin(name) => format!("{}patina.role {} {{", indent, name),
                Op::RoleEnd | Op::WarpSpecializeEnd | Op::ForEnd => format!("{}}}", indent),
                Op::ForBegin {
                    i: ix,
                    bound,
                    s,
                    ph,
                    init_ph,
                } => format!(
                    "{}patina.for {} = 0 to {} iter_args({} = 0, {} = {}) {{",
                    indent, ix, bound, s, ph, init_ph
                ),
                Op::Yield { vals } => format!("{}patina.yield {}", indent, vals.join(", ")),
                Op::WaitStage { pipe, kind, s, ph } => format!(
                    "{}patina.wait_stage {}, {}, {}, {}",
                    indent,
                    pipe,
                    kind.mlir(),
                    s,
                    ph
                ),
                Op::ExpectTxStage {
                    pipe,
                    kind,
                    s,
                    bytes,
                } => format!(
                    "{}patina.expect_tx_stage {}, {}, {}, {}",
                    indent,
                    pipe,
                    kind.mlir(),
                    s,
                    bytes
                ),
                Op::StageBuf { res, base, s, ty } => format!(
                    "{}{:<w$} = patina.stage_buf {}, {} : {}",
                    indent,
                    res,
                    base,
                    s,
                    ty,
                    w = widths[i]
                ),
                Op::StageNext {
                    res_s,
                    res_ph,
                    s,
                    ph,
                    pipe,
                } => format!(
                    "{}{:<w$} = patina.stage_next {}, {} of {}",
                    indent,
                    format!("{}, {}", res_s, res_ph),
                    s,
                    ph,
                    pipe,
                    w = widths[i]
                ),
                Op::Muli { res, x, k } => format!(
                    "{}{:<w$} = patina.muli {}, {}",
                    indent,
                    res,
                    x,
                    k,
                    w = widths[i]
                ),
                Op::Nez { res, x } => format!(
                    "{}{:<w$} = patina.nez {}",
                    indent,
                    res,
                    x,
                    w = widths[i]
                ),
                Op::ExpectTx { bar, bytes } => {
                    format!("{}patina.expect_tx {}, {}", indent, bar, bytes)
                }
                Op::TmaLoad {
                    map,
                    dst,
                    coords,
                    complete,
                } => {
                    format!(
                        "{}patina.tma_load {} -> {}, coords [{}], complete {}",
                        indent,
                        map,
                        dst,
                        coord_elems(coords),
                        stage_target(complete)
                    )
                }
                Op::Wait { bar, phase } => {
                    format!("{}patina.wait {}, phase = {}", indent, bar, phase)
                }
                Op::UmmaBegin { res, acc, a, b } => format!(
                    "{}{:<w$} = patina.umma.begin {}, {}, {}",
                    indent,
                    res,
                    acc,
                    a,
                    b,
                    w = widths[i]
                ),
                Op::UmmaStep { m, accum } => match accum {
                    AccumOp::Static(b) => {
                        format!("{}patina.umma.step {} {{accum = {}}}", indent, m, b)
                    }
                    AccumOp::Dyn(c) => format!("{}patina.umma.step {}, accum {}", indent, m, c),
                },
                Op::UmmaAdvanceK { m, k } => {
                    format!("{}patina.umma.advance_k {}, {}", indent, m, k)
                }
                Op::UmmaCommit { m, target } => {
                    format!(
                        "{}patina.umma.commit {} -> {}",
                        indent,
                        m,
                        stage_target(target)
                    )
                }
                Op::TmemLd { res, acc, ty } => format!(
                    "{}{:<w$} = patina.tmem_ld {} : {}",
                    indent,
                    res,
                    acc,
                    ty,
                    w = widths[i]
                ),
                Op::Store { ptr, frag } => format!("{}patina.store {}, {}", indent, ptr, frag),
                Op::TmemDealloc { acc } => format!("{}patina.tmem.dealloc {}", indent, acc),
            };
            if matches!(op, Op::WarpSpecializeBegin | Op::RoleBegin(_) | Op::ForBegin { .. }) {
                depth += 1;
            }
            out.push_str(&line);
            out.push('\n');
        }
    }

    out.push_str("}\n");
    out
}
