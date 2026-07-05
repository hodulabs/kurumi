//! Weight-only dequant matmul kernel sources: a decode/small-batch GEMV and a prefill GEMM,
//! each baked per (bits, sym) and templated on the activation dtype. Launched by
//! `context/dispatch/quant.rs`.

// Emit MSL statements that decode the packed weight at (`row`, column `k`) into a fresh
// `float w`, given `ng`, `rb`, `G`, `scales`/`mins`/`qw` in scope. bits/sym are baked at
// codegen time so each kernel variant compiles a lean loop (a runtime bits/sym branch keeps
// both int4 and int8 decode paths live, inflating registers and stalling the loop). asym:
// q*scale+min; sym: signed(q)*scale, int4 sign-extending the low nibble.
fn decode_w(bits: u8, sym: bool, row: &str, k: &str) -> String {
    let wrow = format!("(qw + {row} * rb)");
    let scale = format!("float(scales[{row} * ng + ({k}) / G])");
    // general bits-wide field: 8/bits fields per byte, low field first (branchless shift).
    let (per, mask, sh) = (8 / bits, (1u32 << bits) - 1, 32 - bits);
    let extract = format!("uint q = (uint({wrow}[({k}) / {per}u]) >> ((({k}) % {per}u) * {bits}u)) & {mask:#x}u;");
    if sym {
        format!("{extract} float w = float((int(q) << {sh}u) >> {sh}u) * {scale};")
    } else {
        let mn = format!("float(mins[{row} * ng + ({k}) / G])");
        format!("{extract} float w = float(q) * {scale} + {mn};")
    }
}

// row stride in bytes = cols * bits / 8.
fn rb_expr(bits: u8) -> String {
    format!("(K * {bits}u / 8u)")
}

// Emit MSL decoding field `j` (0-based) of a packed 32-bit `word` into a fresh `float w`,
// given `scale`/`mn` in scope. A word holds 32/bits weights (16 int2 / 8 int4 / 4 int8);
// loading the row as 32-bit words (vs byte-at-a-time) is what makes the GEMV reads coalesce.
fn decode_word(bits: u8, sym: bool) -> String {
    let (mask, sh) = ((1u32 << bits) - 1, 32 - bits);
    let extract = format!("uint q = (word >> ({bits}u * j)) & {mask:#x}u;");
    if sym {
        format!("{extract} float w = float((int(q) << {sh}u) >> {sh}u) * scale;")
    } else {
        format!("{extract} float w = float(q) * scale + mn;")
    }
}

// weights packed per 32-bit word: 16 int2 / 8 int4 / 4 int8.
fn wper_expr(bits: u8) -> String {
    format!("{}u", 32 / bits)
}

// Decode/small-batch GEMV: one simdgroup (32 lanes) per output column n. Each lane loads the
// packed weight row as 32-bit words (WPER weights each), so the simdgroup's loads coalesce
// into wide transactions (byte-at-a-time reads leave int4 bandwidth-starved); the lanes split
// the words, accumulate partial dots, and `simd_sum` reduces. Holds up to BM=8 activation rows
// in registers, decoding each weight once and reusing it across them. `mn` line and word
// decode are baked per (bits, sym); act/out are `at`, accumulate in float.
pub(crate) fn dequant_gemv_msl(at: &str, bits: u8, sym: bool, single: bool) -> String {
    let (rb, wper, decode) = (rb_expr(bits), wper_expr(bits), decode_word(bits, sym));
    let mn_line = if sym { "" } else { "float mn = float(mins[n * ng + grp]);" };
    // M==1 decode: a scalar accumulator (fewest registers -> highest occupancy, best latency
    // hiding). 2..=8 small batch: a register-kept acc array with i<M guards (a runtime-M bound
    // would spill it to thread-local memory).
    let (init, accum, reduce) = if single {
        (
            "float acc = 0.0f;".to_string(),
            "acc += float(act[k]) * w;".to_string(),
            format!("float s = simd_sum(acc); if (lane == 0u) out[n] = ({at})s;"),
        )
    } else {
        (
            "float acc[BM]; for (uint i = 0; i < BM; i++) acc[i] = 0.0f;".to_string(),
            "for (uint i = 0; i < BM; i++) if (i < M) acc[i] += float(act[i * K + k]) * w;".to_string(),
            format!(
                "for (uint i = 0; i < BM; i++) if (i < M) {{ float s = simd_sum(acc[i]); if (lane == 0u) out[i * N + n] = ({at})s; }}"
            ),
        )
    };
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
#define BM 8u
#define WPER {wper}
kernel void dequant_gemv(device const {at}*  act    [[buffer(0)]],
                         device const uchar* qw     [[buffer(1)]],
                         device const half*  scales [[buffer(2)]],
                         device const half*  mins   [[buffer(3)]],
                         device {at}*        out    [[buffer(4)]],
                         constant uint& M    [[buffer(5)]],
                         constant uint& K    [[buffer(6)]],
                         constant uint& N    [[buffer(7)]],
                         constant uint& G    [[buffer(8)]],
                         uint n    [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]]) {{
    if (n >= N) return;
    uint ng = K / G, nwords = K / WPER;
    device const uint* wrow = (device const uint*)(qw + n * {rb});
    {init}
    for (uint widx = lane; widx < nwords; widx += 32u) {{
        uint word = wrow[widx];
        uint wbase = widx * WPER;
        uint grp = wbase / G;              // WPER weights share a group (group_size % WPER == 0)
        float scale = float(scales[n * ng + grp]);
        {mn_line}
        for (uint j = 0; j < WPER; j++) {{
            uint k = wbase + j;
            {decode}
            {accum}
        }}
    }}
    {reduce}
}}
"#
    )
}

// Prefill GEMM: each threadgroup computes a BM x BN output tile, streaming K in BK chunks.
// Every K-chunk decodes a BN x BK weight tile into threadgroup memory ONCE and reuses it
// across the tile's BM activation rows (the naive kernel re-decodes per output row).
// the weight tile is re-decoded ceil(M/BM) times across m-tiles; raise BM to cut it.
pub(crate) fn dequant_gemm_msl(at: &str, bits: u8, sym: bool) -> String {
    let (rb, decode) = (rb_expr(bits), decode_w(bits, sym, "wn", "wk"));
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
#define BM 64u
#define BN 64u
#define BK 16u
#define TM 4u
#define TN 4u
#define NT 256u
kernel void dequant_gemm(device const {at}*  act    [[buffer(0)]],
                         device const uchar* qw     [[buffer(1)]],
                         device const half*  scales [[buffer(2)]],
                         device const half*  mins   [[buffer(3)]],
                         device {at}*        out    [[buffer(4)]],
                         constant uint& M    [[buffer(5)]],
                         constant uint& K    [[buffer(6)]],
                         constant uint& N    [[buffer(7)]],
                         constant uint& G    [[buffer(8)]],
                         uint2 tid [[thread_position_in_threadgroup]],
                         uint2 bid [[threadgroup_position_in_grid]]) {{
    threadgroup float As[BM][BK];
    threadgroup float Ws[BN][BK];
    uint ng = K / G, rb = {rb};
    uint t = tid.y * (BN / TN) + tid.x;   // flat thread id, 0..NT-1
    uint row0 = bid.y * BM + tid.y * TM;  // this thread's first output row
    uint col0 = bid.x * BN + tid.x * TN;
    // register-blocked TM x TN micro-tile per thread (high arithmetic intensity)
    float acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) acc[i][j] = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += BK) {{
        // cooperatively stage the act tile and the DECODED weight tile into threadgroup memory
        for (uint e = 0; e < (BM * BK) / NT; e++) {{
            uint idx = t + e * NT, r = idx / BK, c = idx % BK;
            uint m = bid.y * BM + r, kk = k0 + c;
            As[r][c] = (m < M && kk < K) ? float(act[m * K + kk]) : 0.0f;
        }}
        for (uint e = 0; e < (BN * BK) / NT; e++) {{
            uint idx = t + e * NT, r = idx / BK, c = idx % BK;
            uint wn = bid.x * BN + r, wk = k0 + c;
            if (wn < N && wk < K) {{ {decode} Ws[r][c] = w; }} else {{ Ws[r][c] = 0.0f; }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk++) {{
            float a[TM], wv[TN];
            for (uint i = 0; i < TM; i++) a[i] = As[tid.y * TM + i][kk];
            for (uint j = 0; j < TN; j++) wv[j] = Ws[tid.x * TN + j][kk];
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++) acc[i][j] += a[i] * wv[j];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) {{
            uint m = row0 + i, n = col0 + j;
            if (m < M && n < N) out[m * N + n] = ({at})acc[i][j];
        }}
}}
"#
    )
}
