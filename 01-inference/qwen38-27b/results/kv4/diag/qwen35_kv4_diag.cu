// Ours: --kv-cache q4 for Qwen3.5: key and value rows as 32-value groups turned by H32, 4-bit codes and fp16 scales.

#include <cuda_fp16.h>

#include "qwen35.cuh"

using namespace tf_q35;

// A row's bytes: HD 4-bit codes, element d's in byte d / 2 (the low nibble first), then HD / 32 fp16 group scales.
#ifndef KRAW
#define KRAW 0
#endif
#ifndef VRAW
#define VRAW 0
#endif
#ifndef SINK
#define SINK 0
#endif
constexpr int QROW = HD / 2 + HD / 32 * 2;
constexpr int ROW = QROW + 2 * HD;
constexpr float INV_SQRT32 = 0.17677669529663688110f;

// A layer's block in Seg.kv: kvh * cap rows of ROW bytes for keys, then as many for values.
__device__ __forceinline__ unsigned char* layer_at(const Seg& sg, int layer, int kvh) {
    return reinterpret_cast<unsigned char*>(sg.kv) + (size_t)layer * 2 * kvh * sg.cap * ROW;
}

// H32 over a warp, element i in lane i: butterflies of stride 1 to 16 (lower a + b, upper a - b), then 1 / sqrt(32).
__device__ __forceinline__ float h32_lanes(float x, int lane) {
#pragma unroll
    for (int lo = 1; lo < 32; lo *= 2) {
        const float o = __shfl_xor_sync(0xffffffffu, x, lo);
        x = (lane & lo) ? __fsub_rn(o, x) : __fadd_rn(x, o);
    }
    return __fmul_rn(x, INV_SQRT32);
}

// h32_lanes' arithmetic on groups held eight a lane: element 8 * (lane % 4) + i of group lane / 4 in x[i].
__device__ __forceinline__ void h32_rows(float (&x)[8], int lane) {
#pragma unroll
    for (int lo = 1; lo < 8; lo *= 2)
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (!(i & lo)) {
                const float a = x[i], b = x[i + lo];
                x[i] = __fadd_rn(a, b);
                x[i + lo] = __fsub_rn(a, b);
            }
#pragma unroll
    for (int lo = 1; lo < 4; lo *= 2)
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float o = __shfl_xor_sync(0xffffffffu, x[i], lo);
            x[i] = (lane & lo) ? __fsub_rn(o, x[i]) : __fadd_rn(x[i], o);
        }
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = __fmul_rn(x[i], INV_SQRT32);
}

// A warp's group turned into row r: codes clamp(floor(8 y / s) + 8, 0, 15), s its fp16 absmax (NaN past finite).
__device__ __forceinline__ void quant_group(float x, int d, unsigned char* r) {
    const int lane = d & 31;
    const float y = h32_lanes(x, lane);
    float amax = fabsf(y);
#pragma unroll
    for (int o = 16; o > 0; o /= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const bool bad = __any_sync(0xffffffffu, !isfinite(y));
    const __half sh = bad ? __float2half_rn(__int_as_float(0x7fc00000)) : __float2half_rn(amax);
    const float s = __half2float(sh);
    const float inv = s > 0.0f ? __fdiv_rn(8.0f, s) : 0.0f;
    const unsigned c = (unsigned)fminf(fmaxf(__fadd_rn(floorf(__fmul_rn(y, inv)), 8.0f), 0.0f), 15.0f);
    unsigned w = c << (4 * (lane & 7));
    w |= __shfl_xor_sync(0xffffffffu, w, 1);
    w |= __shfl_xor_sync(0xffffffffu, w, 2);
    w |= __shfl_xor_sync(0xffffffffu, w, 4);
    if ((lane & 7) == 0) *reinterpret_cast<unsigned*>(r + d / 2) = w;
    if (lane == 0) *reinterpret_cast<__half*>(r + HD / 2 + 2 * (d / 32)) = sh;
}

// A key and value as 4-bit rows in their cache slot; every thread of the block calls it together.
struct StoreQ4 {
    int layer, kvh;
    __device__ __forceinline__ void operator()(const Seg& sg, int h, int slot, int d, __nv_bfloat16 k, __nv_bfloat16 v) const {
        unsigned char* keys = layer_at(sg, layer, kvh);
        const size_t row = (size_t)h * sg.cap + slot;
        quant_group(bf(k), d, keys + row * ROW);
        quant_group(bf(v), d, keys + ((size_t)kvh * sg.cap + row) * ROW);
        reinterpret_cast<__nv_bfloat16*>(keys + row * ROW + QROW)[d] = k;
        reinterpret_cast<__nv_bfloat16*>(keys + ((size_t)kvh * sg.cap + row) * ROW + QROW)[d] = v;
    }
};

// tf_q35_rope's rows and t, h, w positions (rope_rows) with the keys and values written as 4-bit rows.
extern "C" __global__ void __launch_bounds__(256) tf_q35_rope4(Geo geo, __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Kn,
                                                                 const __nv_bfloat16* __restrict__ Vr, const Seg* __restrict__ segs,
                                                                 const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                 const int* __restrict__ P, const int* __restrict__ M) {
    rope_rows(geo, Qn, Kn, Vr, segs, rowseg, rowanc, P, M, StoreQ4{layer, geo.kvh});
}

// Code i of a lane's word as c - 7.5, exactly: c lands in the mantissa of 2^22, whose step is 0.5.
__device__ __forceinline__ float code(unsigned w, int i) {
    return __fsub_rn(__uint_as_float(0x4a800000u | ((w >> (4 * i)) & 15u) << 1), 4194311.5f);
}

// tf_q35_attn_partial over 4-bit rows in turned space: the query turned once, keys in fours, the output turned back.
extern "C" __global__ void tf_q35_attn_partial4(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                  const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                  const int* __restrict__ P, float scale,
                                                  float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    constexpr int DPL = HD / 32, B = 4;
    const int QH = geo.qh, KVH = geo.kvh;
    const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
    const int c = blockIdx.x, h = blockIdx.y;
    if (c >= MAXCH || past_rows(P, blockIdx.z)) return;
    const size_t m = blockIdx.z;
    const int qh = h * (QH / KVH) + g;
    const Seg sg = segs[rowseg[m]];
    const Ancestor anc = rowanc[m];
    const int keys = sg.len + rows_seen(anc), ck = chunk_keys(keys);
    const int k0 = c * ck, k1 = min(k0 + ck, keys);
    const size_t slot = (m * QH + qh) * MAXCH + c;
    if (k0 >= k1) {
        if (lane == 0) {
            PM[slot] = -INFINITY;
            PL[slot] = 0.0f;
        }
        return;
    }
    const unsigned char* K = layer_at(sg, layer, KVH) + (size_t)h * sg.cap * ROW;
    const unsigned char* V = K + (size_t)KVH * sg.cap * ROW;
    const int at = lane * 4, sc = HD / 2 + (lane >> 2) * 2;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[(m * QH + qh) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    h32_rows(q, lane);
    float mx = -INFINITY, l = 0.0f;
    for (int t = k0; t < k1; t++) {
        const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
        const unsigned char* kr = K + (size_t)pos * ROW;
        const unsigned char* vr = V + (size_t)pos * ROW;
        float kf[DPL], vf[DPL];
        const bool rk = KRAW || pos < SINK, rv = VRAW || pos < SINK;
        if (rk) {
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(kr + QROW) + lane * DPL;
            for (int i = 0; i < DPL; i++) kf[i] = bf(p[i]);
            h32_rows(kf, lane);
        } else {
            const unsigned w = *reinterpret_cast<const unsigned*>(kr + at);
            const float s = __fmul_rn(__half2float(*reinterpret_cast<const __half*>(kr + sc)), 0.125f);
            for (int i = 0; i < DPL; i++) kf[i] = code(w, i) * s;
        }
        if (rv) {
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(vr + QROW) + lane * DPL;
            for (int i = 0; i < DPL; i++) vf[i] = bf(p[i]);
            h32_rows(vf, lane);
        } else {
            const unsigned w = *reinterpret_cast<const unsigned*>(vr + at);
            const float s = __fmul_rn(__half2float(*reinterpret_cast<const __half*>(vr + sc)), 0.125f);
            for (int i = 0; i < DPL; i++) vf[i] = code(w, i) * s;
        }
        float dot = 0.0f;
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], kf[i], dot);
        for (int w = 16; w; w >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, w);
        const float sj = dot * scale;
        const float mn = fmaxf(mx, sj);
        const float a = __expf(mx - mn), b = __expf(sj - mn);
        l = l * a + b;
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, vf[i], o[i] * a);
        mx = mn;
    }
    h32_rows(o, lane);
    if (lane == 0) {
        PM[slot] = mx;
        PL[slot] = l;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) PO[slot * HD + lane * DPL + i] = o[i];
}

// tf_q35k_compact over 4-bit rows: the path's key and value rows move down to the slots after the cache.
extern "C" __global__ void __launch_bounds__(256) tf_q35k_compact4(Geo geo, Seg sg, Path path) {
    const int KVH = geo.kvh;
    const int layer = blockIdx.x, h = blockIdx.y;
    unsigned* keys = reinterpret_cast<unsigned*>(layer_at(sg, layer, KVH) + (size_t)h * sg.cap * ROW);
    unsigned* values = keys + (size_t)KVH * sg.cap * ROW / 4;
    const int d = threadIdx.x;
    for (int j = 0; j < path.n; j++) {
        const int from = sg.len + path.rows[j], to = sg.len + j;
        if (from != to && d < ROW / 4) {
            keys[(size_t)to * ROW / 4 + d] = keys[(size_t)from * ROW / 4 + d];
            values[(size_t)to * ROW / 4 + d] = values[(size_t)from * ROW / 4 + d];
        }
        __syncthreads();
    }
}
