//! End-to-end tests of 64-bit image atomics (`image_atomic_*` with two
//! data dwords): synthetic GFX10 programs lowered to DXIL, validated by
//! `dxil.dll` when present, and run on the hardware GPU and on WARP.

use super::host::View;
use super::scalar_buffer_tests::{compute, names, range, run};
use crate::native::ResourceType;

const S_ENDPGM: u32 = 0xBF81_0000;

/// 32 lanes each add 0x1_8000_0000 to texel (0, 0), which carries from
/// the low dword into the high one, and take the unsigned maximum of
/// (lane << 32 | !lane) at texel (1, 0), of a 2x1 texture at heap index 0.
const ADD_AND_MAX: &[u32] = &[
    0xBE82_0300, // s_mov_b32 s2, s0
    0xBE83_0380, // s_mov_b32 s3, 0
    0xF40C_0501,
    0xFA00_0000, // s_load_dwordx8 s[20:27], s[2:3], null
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E04_0280, // v_mov_b32 v2, 0
    0x7E06_0280, // v_mov_b32 v3, 0
    0x7E08_02FF,
    0x8000_0000, // v_mov_b32 v4, 0x80000000
    0x7E0A_0281, // v_mov_b32 v5, 1
    0xF044_1308,
    0x0005_0402, // image_atomic_add v[4:5], v[2:3], s[20:27] dmask:0x3 dim:2D unorm
    0x7E04_0281, // v_mov_b32 v2, 1
    0x7E0A_0300, // v_mov_b32 v5, v0
    0x7E08_6F00, // v_not_b32 v4, v0
    0xF05C_1308,
    0x0005_0402, // image_atomic_umax v[4:5], v[2:3], s[20:27] dmask:0x3 dim:2D unorm
    S_ENDPGM,
];

/// Texel (0, 0): 32 times 0x1_8000_0000; texel (1, 0): lane 31's value.
const WANT: [u32; 4] = [0, 0x30, !31, 31];

/// The texture's four dwords after [`ADD_AND_MAX`], per adapter. The
/// host has no 64-bit integer format: 64-bit atomics run on an
/// R32G32_UINT view, as the console's.
fn texels() -> Vec<(String, Result<Vec<u32>, String>)> {
    let shader = compute(ADD_AND_MAX, 32, vec![range(ResourceType::Uav, 0, 8)]);
    let views = [View::Uint2TextureUav(2, 1)];
    run(&shader, &views, false)
        .into_iter()
        .map(|(name, r)| (name, r.map(|mut uavs| uavs.remove(0))))
        .collect()
}

#[test]
fn sixty_four_bit_image_atomics_carry_and_compare_all_64_bits() {
    assert_eq!(
        names(ADD_AND_MAX),
        [
            "s_mov_b32",
            "s_mov_b32",
            "s_load_dwordx8",
            "v_mov_b32",
            "v_mov_b32",
            "v_mov_b32",
            "v_mov_b32",
            "image_atomic_add",
            "v_mov_b32",
            "v_mov_b32",
            "v_not_b32",
            "image_atomic_umax",
            "s_endpgm",
        ]
    );
    for (name, got) in texels() {
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got, WANT, "{name}");
        eprintln!("{name}: 64-bit image atomics exact");
    }
}
