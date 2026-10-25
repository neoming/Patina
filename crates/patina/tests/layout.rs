//! Unit tests for the layout algebra and every derived encoding.
//! These are pure host-side Rust and should be the FIRST tests to go green
//! in M1 — they pin down the math that the reference .cu got wrong by hand.
//!
//! Reference: ~/WorkSpace/tcgen05_gemm/tcgen05_gemm.cu,
//!            make_umma_tmap_f16 / make_desc / make_idesc_f16.

use patina::layout::*;

fn tile() -> UmmaKMajor<NoSwizzle> {
    UmmaKMajor::<NoSwizzle>::new(shape![128, 32], F16)
}

// --- TMA tensor-map derivation -------------------------------------------
// [128,32] f16 K-major -> 4D view {8 f16, 8 rows, 4 k-chunks, 16 blocks}
#[test]
fn tma_view_dims() {
    let v = tile().tma_view();
    assert_eq!(v.rank, 4);
    assert_eq!(v.global_dim,  [8, 8, 4, 16]);
    assert_eq!(v.box_dim,     [8, 8, 4, 16]);
    // strides of dims 1..=3, in BYTES (dim0 stride is always element-size)
    assert_eq!(v.global_stride_bytes, [64, 16, 512]);
    assert_eq!(v.swizzle, Swizzle::None);
    // TMA fills smem linearly in dim order; assert it reproduces the
    // canonical UMMA layout: offset = kc*128 + (r/8)*256 + (r%8)*16
    assert_eq!(v.smem_offset(0, 0, 2, 3), 2 * 128 + 3 * 256); // kc=2, r=24
    assert_eq!(v.smem_offset(0, 5, 1, 0), 1 * 128 + 5 * 16);  // kc=1, r=5
}

// --- UMMA shared-memory descriptor ----------------------------------------
#[test]
fn smem_descriptor_fields() {
    let d = tile().descriptor(/*smem addr*/ 0x4000);
    assert_eq!(d.start_addr_enc(), 0x4000 >> 4);   // bits 0..13,  >>4 units
    assert_eq!(d.lbo(), 128);                       // core matrices adj. in K
    assert_eq!(d.sbo(), 256);                       // adjacent 8-row blocks
    assert_eq!(d.version(), 1);                     // Blackwell
    assert_eq!(d.swizzle_mode(), 0);                // SWIZZLE_NONE
}

// --- K advance: regression test for the da += 8 bug ------------------------
// One K16 MMA step consumes TWO 16B core matrices (16 f16 x 2B = 32B/row),
// so advancing K by 16 halves = 256 B = 16 descriptor units, NOT 8.
#[test]
fn k_advance_is_two_core_matrices() {
    assert_eq!(tile().k_advance_units(16), 16);
    // and the raw byte distance, for the human reading the failure message
    assert_eq!(tile().k_advance_units(16) * 16, 256);
}

// --- UMMA instruction descriptor (kind::f16, dense) ------------------------
// c_format=F32(1)<<4 | n_dim=(N>>3)<<17 | m_dim=(M>>4)<<24; rest zero
// (qualified paths: the test fns below would otherwise shadow the imports)
#[test]
fn idesc_f16() {
    assert_eq!(patina::layout::idesc_f16(128, 128), (1 << 4) | ((128 >> 3) << 17) | ((128 >> 4) << 24));
    assert_eq!(patina::layout::idesc_f16(64, 256),  (1 << 4) | ((256 >> 3) << 17) | ((64  >> 4) << 24));
}

// --- TMEM lane addressing ---------------------------------------------------
// warp w owns TMEM lanes [32w, 32w+32); lane base lives at bit 16 of the
// TMEM address, i.e. warp rank shifts in as (w << 21).
#[test]
fn tmem_lane_offset() {
    assert_eq!(patina::layout::tmem_lane_offset(0), 0 << 21);
    assert_eq!(patina::layout::tmem_lane_offset(3), 3 << 21);
}

// --- byte counts used by expect_tx ------------------------------------------
#[test]
fn tile_bytes() {
    assert_eq!(tile().bytes(), 8192); // 128*32 f16
}
