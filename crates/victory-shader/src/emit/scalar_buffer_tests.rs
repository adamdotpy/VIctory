//! End-to-end tests of scalar buffer loads (`s_buffer_load_dword*`)
//! through a descriptor whose record count the program also reads as a
//! number: synthetic GFX10 programs lowered to DXIL, validated by
//! `dxil.dll` when present, and run on the hardware GPU and on WARP.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// The names `words` decode to, to check hand-written encodings.
pub(super) fn names(words: &[u32]) -> Vec<&'static str> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, false, 256).unwrap();
    p.blocks
        .iter()
        .flat_map(|b| b.insts.iter().map(|i| i.name()))
        .collect()
}

/// A table range of `kind` at table dword `offset` (`dwords` per
/// descriptor), with the table in user SGPR 0.
pub(super) fn range(kind: ResourceType, offset: u32, dwords: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: offset,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    }
}

/// Lowers a wave32 compute program of `threads` threads with one
/// descriptor table in user SGPR 0 and the ranges `used`, and validates
/// it when `dxil.dll` is present.
pub(super) fn compute(words: &[u32], threads: u32, used: Vec<UsedResource>) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// The name of the adapter `host` runs on.
fn adapter(host: &Host) -> String {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
    };
    // SAFETY: plain DXGI queries with valid out values.
    unsafe {
        let luid = host.device.GetAdapterLuid();
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(luid))
            .and_then(|a| a.GetDesc1())
            .map_or_else(
                |e| format!("unknown adapter ({e})"),
                |d| {
                    String::from_utf16_lossy(&d.Description)
                        .trim_end_matches('\0')
                        .to_string()
                },
            )
    }
}

/// Per adapter that ran: its name and the UAVs' contents, or why the
/// pipeline or dispatch failed.
pub(super) type Runs = Vec<(String, Result<Vec<Vec<u32>>, String>)>;

/// Runs `shader` once (one group) with `views` at heap indices 0.., the
/// table at heap index 0, on the hardware GPU and on WARP, and returns
/// per adapter its name and the UAVs' contents. The hardware GPU must be
/// there: these tests are about what it computes.
pub(super) fn run(shader: &HostShader, views: &[View], warp_only: bool) -> Runs {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    let backends: &[bool] = if warp_only { &[true] } else { &[false, true] };
    for &warp in backends {
        let host = match Host::new(warp) {
            Ok(h) => h,
            Err(e) if warp => {
                eprintln!("no SM 6.6 WARP device ({e}); skipped");
                continue;
            }
            Err(e) => panic!("no SM 6.6 hardware device: {e}"),
        };
        let name = adapter(&host);
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table, "{u:?}");
            ud[u.index as usize] = 0;
        }
        let result = host.compute_pipeline(shader).and_then(|(root, pso)| {
            host.dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
        });
        out.push((name, result));
    }
    out
}

#[test]
fn scalar_loads_read_a_raw_buffer_and_its_record_count() {
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null (raw SRV, heap 0)
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (raw UAV, heap 1)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF420_0404,
        0xFA00_0008, // s_buffer_load_dword s16, s[8:11], 0x8
        0x8011_C40A, // s_add_u32 s17, s10, -4 (the last dword, from the record count)
        0xF420_0484,
        0x2200_0000, // s_buffer_load_dword s18, s[8:11], s17
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_0210, // v_mov_b32 v1, s16
        0x7E04_0212, // v_mov_b32 v2, s18
        0x7E06_020A, // v_mov_b32 v3, s10
        0xE070_0000,
        0x8003_0100, // buffer_store_dword v1, off, s[12:15], 0
        0xE070_0004,
        0x8003_0200, // buffer_store_dword v2, off, s[12:15], 0 offset:4
        0xE070_0008,
        0x8003_0300, // buffer_store_dword v3, off, s[12:15], 0 offset:8
        S_ENDPGM,
    ];
    // (The IR leaves out `s_waitcnt`.)
    assert_eq!(
        names(&words),
        [
            "s_mov_b32",
            "s_mov_b32",
            "s_load_dwordx4",
            "s_load_dwordx4",
            "s_buffer_load_dword",
            "s_add_u32",
            "s_buffer_load_dword",
            "v_mov_b32",
            "v_mov_b32",
            "v_mov_b32",
            "buffer_store_dword",
            "buffer_store_dword",
            "buffer_store_dword",
            "s_endpgm",
        ]
    );
    let shader = compute(
        &words,
        1,
        vec![
            range(ResourceType::Srv, 0, 4),
            range(ResourceType::Uav, 8, 4),
        ],
    );
    // 20 dwords: a record count of 80 bytes, not a power of two.
    let data: Vec<u32> = (0..20).map(|i| 0x1000 + i).collect();
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(4),
    ];
    for (name, result) in run(&shader, &views, false) {
        let uavs = result.unwrap_or_else(|e| panic!("{name}: {e}"));
        // Dword 2, the last dword (through the record count), the count.
        assert_eq!(uavs[0], [0x1002, 0x1013, 80, 0], "{name}");
        eprintln!("{name}: scalar loads and the record count exact");
    }
}
