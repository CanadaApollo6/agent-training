// Tree attention over a 4-bit cache (kvq BITS=4, head dim 256): a pair of warps per 16 (row, head) pairs and
// 512-key chunk.
//
// The 3090 runs int->float and float->bf16 conversions at 1/8 of its float rate and Triton's kernel stages every
// dequantized tile through shared memory between barriers. Here each lane reads packed codes straight from the
// cache into its mma.sync fragments: a code nibble becomes bf16 by bit placement (0x3F80 | c << 3 = 1 + c/16), two
// packed bf16 FMAs make (c - 7.5)/16 exactly and then times bf16(2 * scale). The head's 256 values are assigned to
// MMA slots so that a lane's share is one contiguous run of a row: 32 bytes of a key row for q.k (the queries
// follow the same order) and 8 bytes of one scale group of four value rows for p.v (the output is written back in
// order). Of a 16-key step, each warp of the pair scores 8 keys, the pair swaps scores through shared memory, both
// run the same softmax and each accumulates half the values: 64 output registers a lane, not 128, so more warps
// fit an SM. Shared chunks and the tail (the last committed keys plus each row's own path) run the same code per
// step, so a full chunk gives identical bits in either and drafted rows equal serial ones.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef ABL
#define ABL 0     // timing ablations: 1 no bf16 FMAs, 2 no nibble math either, 3 no MMAs, 4 no cache loads
#endif

namespace {

constexpr int D = 256, R = 144, CH = 512, MAXD = 128;

// dequant constants, made once by opaque movs: otherwise the compiler rematerializes them before every use (about
// a hundred extra instructions a step)
struct Consts {
    uint32_t mask, one, off, nz;
};
__device__ __forceinline__ Consts consts() {
    Consts k;
    asm volatile("mov.b32 %0, 0x00780078;" : "=r"(k.mask));
    asm volatile("mov.b32 %0, 0x3F803F80;" : "=r"(k.one));
    asm volatile("mov.b32 %0, 0xBFBCBFBC;" : "=r"(k.off));
    asm volatile("mov.b32 %0, 0x80008000;" : "=r"(k.nz));
    return k;
}

__device__ __forceinline__ uint32_t nib_lo(uint32_t x, const Consts& k) {
    if (ABL == 2) return x;
    return ((x << 3) & k.mask) | k.one;
}
__device__ __forceinline__ uint32_t nib_hi(uint32_t x, const Consts& k) {
    if (ABL == 2) return x ^ 0x11u;
    return ((x >> 1) & k.mask) | k.one;
}

__device__ __forceinline__ uint32_t deq(uint32_t raw, uint32_t s, const Consts& k) {
    if (ABL == 1 || ABL == 2) return raw ^ s;
    uint32_t t, o;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(t) : "r"(raw), "r"(k.one), "r"(k.off));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(o) : "r"(t), "r"(s), "r"(k.nz));
    return o;
}

__device__ __forceinline__ uint32_t scale_bf16(uint32_t h16) {          // bf16(2 * fp16 scale) in the low half
    float f = __half2float(__ushort_as_half((unsigned short)(h16 & 0xFFFFu))) * 2.0f;
    return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f));
}

__device__ __forceinline__ uint32_t cvt2(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

__device__ __forceinline__ void mma(float* c, const uint4& a, uint32_t b0, uint32_t b1) {
    if (ABL == 3) {
        c[0] = __int_as_float(__float_as_int(c[0]) ^ b0 ^ b1 ^ a.x);
        return;
    }
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void pair_sync(int id) { asm volatile("bar.sync %0, 64;" ::"r"(id) : "memory"); }

__device__ __forceinline__ uint4 ld16(const int8_t* p) {
    if (ABL == 4) {
        const uint32_t v = (uint32_t)(uintptr_t)p;
        return make_uint4(v, v * 3u, v * 5u, v * 7u);
    }
    return *reinterpret_cast<const uint4*>(p);
}
__device__ __forceinline__ uint2 ld8(const int8_t* p) {
    if (ABL == 4) {
        const uint32_t v = (uint32_t)(uintptr_t)p;
        return make_uint2(v, v * 3u);
    }
    return *reinterpret_cast<const uint2*>(p);
}
__device__ __forceinline__ uint32_t ld4(const int8_t* p) {
    if (ABL == 4) return ((uint32_t)(uintptr_t)p & 0x03FF03FFu) | 0x3C003C00u;
    return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t ld2(const int8_t* p) {
    if (ABL == 4) return ((uint32_t)(uintptr_t)p & 0x03FFu) | 0x3C00u;
    return *reinterpret_cast<const unsigned short*>(p);
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// shared memory a pair: q fragments (16 k-steps x 32 lanes) then two score swap buffers (2 warps x 32 lanes)
constexpr int PAIR_SMEM = 16 * 32 + 2 * 2 * 32;                         // uint4s

template <bool TAIL>
__global__ void attn4_kernel(const __nv_bfloat16* __restrict__ Q, const int8_t* __restrict__ base,
                             const int64_t* __restrict__ OFF, const int* __restrict__ STREAM,
                             const int* __restrict__ ITEMS, int n_items, const int* __restrict__ ROWS,
                             const int* __restrict__ PATHS, const int* __restrict__ DEPTHS,
                             const int8_t* __restrict__ KN, const int8_t* __restrict__ VN, float* __restrict__ PO,
                             float* __restrict__ PM, float* __restrict__ PL, int W, int H, int HK, int G,
                             float scale) {
    extern __shared__ uint4 smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, j = lane & 3;
    const int pair = warp >> 1, half = warp & 1;                // half: q.k keys 8 half.., p.v values 16 half..
    const int hk = blockIdx.y;
    int s, chunk, p, steps, depth = 0, node_t = 0, first = 0, start = 0, rows = 0;
    if (!TAIL) {
        const int item = blockIdx.x * (blockDim.x >> 6) + pair;
        if (item >= n_items) return;                            // both warps of a pair leave together
        s = ITEMS[item * 3];
        first = ITEMS[item * 3 + 1];
        chunk = ITEMS[item * 3 + 2];
        start = STREAM[s * 4];
        rows = STREAM[s * 4 + 1];
        p = STREAM[s * 4 + 2];
        if ((chunk + 1) * CH > p) return;                       // a plan padded for a longer context (a graph's)
        steps = CH / 16;
    } else {
        node_t = blockIdx.x;
        s = ROWS[node_t];
        p = STREAM[s * 4 + 2];
        const int nch = STREAM[s * 4 + 3];
        chunk = p / CH + blockIdx.z;
        if (chunk >= nch) return;
        depth = DEPTHS[node_t];
        const int last = min(p + depth - chunk * CH, CH);      // past it every key is masked: steps there are no-ops
        steps = (last + 15) / 16;
    }
    const int64_t koff = OFF[s * 2], voff = OFF[s * 2 + 1];
    const int bar = 1 + pair;

    // the pair's 16 rows: (node, head) and whether real
    int node_r[2], head_r[2];
    bool ok_r[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = gid + 8 * i;
        if (!TAIL) {
            const int pr = first + r;
            ok_r[i] = pr < rows * G;
            node_r[i] = start + pr / G;
            head_r[i] = hk * G + pr % G;
        } else {
            ok_r[i] = r < G;
            node_r[i] = node_t;
            head_r[i] = hk * G + r;
        }
    }

    // queries as A fragments in the q.k slot order; each warp of the pair writes every other k-step
    uint4* qf = smem + pair * PAIR_SMEM;
    float4* swap = reinterpret_cast<float4*>(qf + 16 * 32);
#pragma unroll
    for (int st = half; st < 16; st += 2) {
        const int d0 = 64 * j + 8 * (st >> 1) + 4 * (st & 1);
        uint32_t lo[2], hi[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint2 v = make_uint2(0u, 0u);
            if (ok_r[i]) v = *reinterpret_cast<const uint2*>(Q + ((int64_t)node_r[i] * H + head_r[i]) * D + d0);
            lo[i] = __byte_perm(v.x, v.y, 0x5410);               // elements d0, d0 + 2
            hi[i] = __byte_perm(v.x, v.y, 0x7632);               // d0 + 1, d0 + 3
        }
        qf[st * 32 + lane] = make_uint4(lo[0], lo[1], hi[0], hi[1]);
    }
    pair_sync(bar);

    float o[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const float LOG2E = 1.4426950408889634f;

    // a key's cache row, or its path row in the window (the tail), or none: then a safe row and zeros
    auto row_of = [&](int key, int64_t off, const int8_t* nodes, bool& valid) -> const int8_t* {
        if (TAIL && key >= p) {
            const int slot = min(key - p, depth - 1);           // path slots past the depth are unset
            valid = key - p < depth;
            return nodes + ((int64_t)PATHS[node_t * MAXD + slot] * HK + hk) * R;
        }
        valid = !TAIL || key < p;                               // shared chunks are full
        if (!TAIL) return base + off + ((int64_t)key * HK + hk) * R;
        return base + off + ((int64_t)(valid ? key : 0) * HK + hk) * R;
    };
    // q.k: this warp's key row is key0 + 8 half + gid, bytes [32 j, 32 j + 32) and 2 scales
    uint4 kw[2];
    uint32_t ks;
    auto load_k = [&](int key0) {
        bool valid;
        const int8_t* row = row_of(key0 + 8 * half + gid, koff, KN, valid);
        const uint4 a = ld16(row + 32 * j), b = ld16(row + 32 * j + 16);
        const uint32_t c = ld4(row + 128 + 4 * j);
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        kw[0] = valid ? a : z;
        kw[1] = valid ? b : z;
        ks = valid ? c : 0u;
    };
    load_k(chunk * CH);
    const Consts kc = consts();

    for (int step = 0; step < steps; ++step) {
        const int key0 = chunk * CH + step * 16;
        // p.v rows, loaded now and used after q.k: keys 2j, 2j+1, 2j+8, 2j+9 of the step, bytes 16 gid + 8 half ..
        uint2 vw[4];
        uint32_t vraw[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            bool valid;
            const int8_t* row = row_of(key0 + 2 * j + (c & 1) + 8 * (c >> 1), voff, VN, valid);
            const uint2 a = ld8(row + 16 * gid + 8 * half);
            const uint32_t sc = ld2(row + 128 + 2 * gid);
            vw[c] = valid ? a : make_uint2(0u, 0u);
            vraw[c] = valid ? sc : 0u;
        }
        float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};     // two chains: half the latency
        {
            const uint32_t a = scale_bf16(ks), b = scale_bf16(ks >> 16);
            const uint32_t sk[2] = {a | (a << 16), b | (b << 16)};
#pragma unroll
            for (int u = 0; u < 8; ++u)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t x = __byte_perm(word(kw[u >> 2], u & 3), 0u, h ? 0x3322 : 0x1100);
                    mma(h ? acc2 : acc, qf[(2 * u + h) * 32 + lane], deq(nib_lo(x, kc), sk[u >> 2], kc),
                        deq(nib_hi(x, kc), sk[u >> 2], kc));
                }
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[c] += acc2[c];
        }
        load_k(chunk * CH + min(step + 1, steps - 1) * 16);     // the next step's key, during softmax and p.v
        // swap scores with the other warp: both then hold the step's 16 keys (C columns 2j, 2j+1 of each tile)
        float4* buf = swap + (step & 1) * 64;
        buf[half * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        pair_sync(bar);
        const float4 other = buf[(1 - half) * 32 + lane];
        const float ot[4] = {other.x, other.y, other.z, other.w};
        float sc[2][4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            sc[0][c] = half ? ot[c] : acc[c];
            sc[1][c] = half ? acc[c] : ot[c];
        }
        // online softmax over these 16 keys
        bool cvalid[2][2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int key = key0 + 8 * nt + 2 * j + e;
                cvalid[nt][e] = key < p || (TAIL && key - p < depth);
            }
        float pr[2][4];
        float alpha[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float sv[4], mx = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float v = cvalid[nt][e] ? sc[nt][2 * i + e] * scale : -INFINITY;
                    sv[nt * 2 + e] = v;
                    mx = fmaxf(mx, v);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const bool active = mx != -INFINITY;
            const float next = active ? fmaxf(m[i], mx) : m[i];
            alpha[i] = active ? (m[i] == -INFINITY ? 0.f : ex2((m[i] - next) * LOG2E)) : 1.f;
            float sum = 0.f;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                float two = 0.f;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float pv = (active && cvalid[nt][e]) ? ex2((sv[nt * 2 + e] - next) * LOG2E) : 0.f;
                    pr[nt][2 * i + e] = pv;
                    two += pv;
                }
                sum += two;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            l[i] = l[i] * alpha[i] + sum;
            m[i] = next;
        }
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            o[n][0] *= alpha[0];
            o[n][1] *= alpha[0];
            o[n][2] *= alpha[1];
            o[n][3] *= alpha[1];
        }
        // p.v over this warp's 16 value tiles
        const uint4 pa = make_uint4(cvt2(pr[0][0], pr[0][1]), cvt2(pr[0][2], pr[0][3]), cvt2(pr[1][0], pr[1][1]),
                                    cvt2(pr[1][2], pr[1][3]));
        uint32_t vs[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) vs[c] = scale_bf16(vraw[c]);
        const uint32_t s01 = vs[0] | (vs[1] << 16), s89 = vs[2] | (vs[3] << 16);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t a0 = i ? vw[0].y : vw[0].x, a1 = i ? vw[1].y : vw[1].x;
            const uint32_t a8 = i ? vw[2].y : vw[2].x, a9 = i ? vw[3].y : vw[3].x;
#pragma unroll
            for (int bb = 0; bb < 4; ++bb) {
                const uint32_t sel = bb | ((4 + bb) << 8);
                const uint32_t x01 = __byte_perm(a0, a1, sel), x89 = __byte_perm(a8, a9, sel);
                const int n = 2 * (4 * i + bb);
                mma(o[n], pa, deq(nib_lo(x01, kc), s01, kc), deq(nib_lo(x89, kc), s89, kc));
                mma(o[n + 1], pa, deq(nib_hi(x01, kc), s01, kc), deq(nib_hi(x89, kc), s89, kc));
            }
        }
    }

    // value tile t (of 32) column n is value 32 n + t; this warp has tiles 16 half .. 16 half + 15, so lane j holds
    // values 64 j + 32 e + 16 half + [0, 16) of rows gid, gid + 8 (e: the column's parity)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (!ok_r[i]) continue;
        const int64_t row = ((int64_t)chunk * W + node_r[i]) * H + head_r[i];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float* out = PO + row * D + 64 * j + 32 * e + 16 * half;
#pragma unroll
            for (int t = 0; t < 16; t += 4)
                *reinterpret_cast<float4*>(out + t) =
                    make_float4(o[t][2 * i + e], o[t + 1][2 * i + e], o[t + 2][2 * i + e], o[t + 3][2 * i + e]);
        }
        if (j == 0 && half == 0) {
            PM[row] = m[i];
            PL[row] = l[i];
        }
    }
}

}  // namespace

void attn4_shared(torch::Tensor q, torch::Tensor base, torch::Tensor off, torch::Tensor stream, torch::Tensor items,
                  torch::Tensor po, torch::Tensor pm, torch::Tensor pl, int64_t w, int64_t hk, double scale,
                  int64_t pairs) {
    const int n_items = items.size(0);
    if (!n_items) return;
    const int h = q.size(1), g = h / hk;
    dim3 grid((n_items + pairs - 1) / pairs, hk);
    attn4_kernel<false><<<grid, 64 * pairs, pairs * PAIR_SMEM * sizeof(uint4), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), base.data_ptr<int8_t>(), off.data_ptr<int64_t>(),
        stream.data_ptr<int>(), items.data_ptr<int>(), n_items, nullptr, nullptr, nullptr, nullptr, nullptr,
        po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), w, h, hk, g, (float)scale);
}

void attn4_tail(torch::Tensor q, torch::Tensor base, torch::Tensor off, torch::Tensor stream, torch::Tensor rows,
                torch::Tensor paths, torch::Tensor depths, torch::Tensor kn, torch::Tensor vn, torch::Tensor po,
                torch::Tensor pm, torch::Tensor pl, int64_t w, int64_t hk, double scale, int64_t tails) {
    const int h = q.size(1), g = h / hk;
    dim3 grid(w, hk, tails);
    attn4_kernel<true><<<grid, 64, PAIR_SMEM * sizeof(uint4), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), base.data_ptr<int8_t>(), off.data_ptr<int64_t>(),
        stream.data_ptr<int>(), nullptr, 0, rows.data_ptr<int>(), paths.data_ptr<int>(), depths.data_ptr<int>(),
        kn.data_ptr<int8_t>(), vn.data_ptr<int8_t>(), po.data_ptr<float>(), pm.data_ptr<float>(),
        pl.data_ptr<float>(), w, h, hk, g, (float)scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shared", &attn4_shared);
    m.def("tail", &attn4_tail);
}
