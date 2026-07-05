//! Value-generator kernel sources: iota (index along an axis) and counter-based (threefry)
//! uniform RNG. Launched by `context/dispatch/generate.rs`.

// `iota_k`: out[gid] = (gid / stride) % axis_len, cast to `ty` (the index along an axis).
pub(crate) fn iota_msl(ty: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void iota_k(device {ty}* out [[buffer(0)]], constant uint& stride [[buffer(1)]],\n\
                            constant uint& axis_len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{\n\
             out[gid] = ({ty})((gid / stride) % axis_len); }}"
    )
}

// `rand_k`: counter-based uniform [0,1) F32. Each element is threefry2x32(seed, index),
// bit-for-bit identical to the CPU oracle (see kurumi_core::rng).
pub(crate) const RAND_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
     static inline uint2 tf(uint2 c, uint2 k) {\n\
         uint ks[3] = {k.x, k.y, 0x1BD11BDA ^ k.x ^ k.y};\n\
         uint ROT[8] = {13u,15u,26u,6u,17u,29u,16u,24u};\n\
         uint x0 = c.x + ks[0], x1 = c.y + ks[1];\n\
         for (uint r=0;r<20u;r++) {\n\
             x0 += x1; uint rr = ROT[r%8u];\n\
             x1 = (x1 << rr) | (x1 >> (32u - rr)); x1 ^= x0;\n\
             if (r%4u==3u) { uint inj=r/4u+1u; x0 += ks[inj%3u]; x1 += ks[(inj+1u)%3u] + inj; }\n\
         }\n\
         return uint2(x0, x1); }\n\
     kernel void rand_k(device float* out [[buffer(0)]], constant ulong& seed [[buffer(1)]],\n\
                        uint gid [[thread_position_in_grid]]) {\n\
         uint2 r = tf(uint2(gid, 0u), uint2((uint)seed, (uint)(seed >> 32)));\n\
         out[gid] = (float)(r.x >> 8) / (float)(1u << 24); }";
