// Tree attention over a 4-bit cache (kvq BITS=4, head dim 256): one warp per 16 (row, head) pairs and 512-key chunk.
//
// The 3090 runs int->float and float->bf16 conversions at 1/8 of its float rate and Triton's kernel stages every
// dequantized tile through shared memory between barriers. Here each lane reads packed codes straight from the
// cache into its mma.sync fragments: a code nibble becomes bf16 by bit placement (0x3F80 | c << 3 = 1 + c/16), two
// packed bf16 FMAs make (c - 7.5)/16 exactly and then times bf16(2 * scale). The head's 256 values are assigned to
// MMA slots so that a lane's share is one contiguous run of a row: 32 bytes of a key row for q.k (the queries
// follow the same order) and one 16-byte scale group of four value rows for p.v (the output is written back in
// order). Shared chunks and the tail (the last committed keys plus each row's own path) run the same code per
// 16-key step, so a full chunk gives identical bits in either and drafted rows equal serial ones.

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

__device__ __forceinline__ uint32_t nib_lo(uint32_t x) {
    if (ABL == 2) return x;
    return ((x << 3) & 0x00780078u) | 0x3F803F80u;
}
__device__ __forceinline__ uint32_t nib_hi(uint32_t x) {
    if (ABL == 2) return x ^ 0x11u;
    return ((x >> 1) & 0x00780078u) | 0x3F803F80u;
}

__device__ __forceinline__ uint32_t deq(uint32_t raw, uint32_t s) {
    if (ABL == 1 || ABL == 2) return raw ^ s;
    uint32_t t, o;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(t) : "r"(raw), "r"(0x3F803F80u), "r"(0xBFBCBFBCu));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(o) : "r"(t), "r"(s), "r"(0x80008000u));
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

__device__ __forceinline__ uint4 ld16(const int8_t* p) {
    if (ABL == 4) {
        const uint32_t v = (uint32_t)(uintptr_t)p;
        return make_uint4(v, v * 3u, v * 5u, v * 7u);
    }
    return *reinterpret_cast<const uint4*>(p);
}
__device__ __forceinline__ uint32_t ld4(const int8_t* p) {
    if (ABL == 4) return ((uint32_t)(uintptr_t)p & 0x03FF03FFu) | 0x3C003C00u;
    return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t ld2(const int8_t* p) {
    if (ABL == 4) return ((uint32_t)(uintptr_t)p & 0x03FFu) | 0x3C00u;
    return *reinterpret_cast<const unsigned short*>(p);
}

template <bool TAIL>
__global__ void attn4_kernel(
    const __nv_bfloat16* __restrict__ Q, const int8_t* __restrict__ base, const int64_t* __restrict__ OFF,
    const int* __restrict__ STREAM, const int* __restrict__ ITEMS, int n_items, const int* __restrict__ ROWS,
    const int* __restrict__ PATHS, const int* __restrict__ DEPTHS, const int8_t* __restrict__ KN,
    const int8_t* __restrict__ VN, float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL, int W,
    int H, int HK, int G, float scale) {
    extern __shared__ uint4 smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, j = lane & 3;
    const int hk = blockIdx.y;
    int s, chunk, p, steps, depth = 0, node_t = 0, first = 0, start = 0, rows = 0;
    if (!TAIL) {
        const int item = blockIdx.x * (blockDim.x >> 5) + warp;
        if (item >= n_items) return;
        s = ITEMS[item * 3];
        first = ITEMS[item * 3 + 1];
        chunk = ITEMS[item * 3 + 2];
        start = STREAM[s * 4];
        rows = STREAM[s * 4 + 1];
        p = STREAM[s * 4 + 2];
        if ((chunk + 1) * CH > p) return;                   // a plan padded for a longer context (a graph's)
        steps = CH / 16;
    } else {
        node_t = blockIdx.x;
        s = ROWS[node_t];
        p = STREAM[s * 4 + 2];
        const int nch = STREAM[s * 4 + 3];
        chunk = p / CH + blockIdx.z;
        if (chunk >= nch) return;
        depth = DEPTHS[node_t];
        const int last = min(p + depth - chunk * CH, CH);  // past it every key is masked: steps there are no-ops
        steps = (last + 15) / 16;
    }
    const int64_t koff = OFF[s * 2], voff = OFF[s * 2 + 1];

    // the warp's 16 rows: (node, head) and whether real
    int node_r[2], head_r[2];
    bool ok_r[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = gid + 8 * i;
        if (!TAIL) {
            const int pair = first + r;
            ok_r[i] = pair < rows * G;
            node_r[i] = start + pair / G;
            head_r[i] = hk * G + pair % G;
        } else {
            ok_r[i] = r < G;
            node_r[i] = node_t;
            head_r[i] = hk * G + r;
        }
    }

    // queries as A fragments in the q.k slot order, kept per lane in shared memory
    uint4* qf = smem + warp * 16 * 32;
#pragma unroll
    for (int st = 0; st < 16; ++st) {
        const int d0 = 64 * j + 8 * (st >> 1) + 4 * (st & 1);
        uint32_t lo[2], hi[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint2 v = make_uint2(0u, 0u);
            if (ok_r[i]) v = *reinterpret_cast<const uint2*>(Q + ((int64_t)node_r[i] * H + head_r[i]) * D + d0);
            lo[i] = __byte_perm(v.x, v.y, 0x5410);           // elements d0, d0 + 2
            hi[i] = __byte_perm(v.x, v.y, 0x7632);           // d0 + 1, d0 + 3
        }
        qf[st * 32 + lane] = make_uint4(lo[0], lo[1], hi[0], hi[1]);
    }

    float o[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const float LOG2E = 1.4426950408889634f;

    // a key's cache row, or its path row in the window (the tail), or none: then a safe row and zeros
    auto row_of = [&](int key, int64_t off, const int8_t* nodes, bool& valid) -> const int8_t* {
        if (TAIL && key >= p) {
            const int slot = min(key - p, depth - 1);       // path slots past the depth are unset
            valid = key - p < depth;
            return nodes + ((int64_t)PATHS[node_t * MAXD + slot] * HK + hk) * R;
        }
        valid = key < p;
        return base + off + ((int64_t)(valid ? key : 0) * HK + hk) * R;
    };
    // q.k: this lane's key rows are key0 + gid and key0 + 8 + gid, bytes [32 j, 32 j + 32) and 2 scales
    uint4 kw[2][2];
    uint32_t ks[2];
    auto load_k = [&](int key0) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            bool valid;
            const int8_t* row = row_of(key0 + 8 * nt + gid, koff, KN, valid);
            const uint4 a = ld16(row + 32 * j);
            const uint4 b = ld16(row + 32 * j + 16);
            const uint32_t c = ld4(row + 128 + 4 * j);
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
            kw[nt][0] = valid ? a : z;
            kw[nt][1] = valid ? b : z;
            ks[nt] = valid ? c : 0u;
        }
    };
    load_k(chunk * CH);
    auto qk = [&](float (&acc)[2][4]) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
            uint32_t sk[2][2];
    #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint32_t a = scale_bf16(ks[nt]), b = scale_bf16(ks[nt] >> 16);
                sk[nt][0] = a | (a << 16);
                sk[nt][1] = b | (b << 16);
            }
    #pragma unroll
            for (int u = 0; u < 8; ++u) {
    #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 a = qf[(2 * u + h) * 32 + lane];
    #pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const uint4& wv = kw[nt][u >> 2];
                        const uint32_t w = (u & 3) == 0 ? wv.x : (u & 3) == 1 ? wv.y : (u & 3) == 2 ? wv.z : wv.w;
                        const uint32_t x = __byte_perm(w, 0u, h ? 0x3322 : 0x1100);
                        const uint32_t sc = sk[nt][u >> 2];
                        mma(acc[nt], a, deq(nib_lo(x), sc), deq(nib_hi(x), sc));
                    }
                }
            }
    };
    float acc[2][4], acc2[2][4];
    qk(acc);                                                    // q.k of step 0
    load_k(chunk * CH + min(1, steps - 1) * 16);

    for (int step = 0; step < steps; ++step) {
        const int key0 = chunk * CH + step * 16;
        // p.v rows, loaded now and used after q.k: keys 2j, 2j+1, 2j+8, 2j+9 of the step, bytes [16 gid, 16 gid + 16)
        uint4 vw[4];
        uint32_t vraw[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            bool valid;
            const int8_t* row = row_of(key0 + 2 * j + (c & 1) + 8 * (c >> 1), voff, VN, valid);
            const uint4 a = ld16(row + 16 * gid);
            const uint32_t sc = ld2(row + 128 + 2 * gid);
            vw[c] = valid ? a : make_uint4(0u, 0u, 0u, 0u);
            vraw[c] = valid ? sc : 0u;
        }
        // the next step's q.k goes to the tensor cores before this step's softmax (its last one is unused)
        qk(acc2);
        load_k(chunk * CH + min(step + 2, steps - 1) * 16);
        // online softmax over these 16 keys (C columns 2j, 2j + 1 of each 8-key tile)
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
                    const float v = cvalid[nt][e] ? acc[nt][2 * i + e] * scale : -INFINITY;
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
                float pair = 0.f;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float pv = (active && cvalid[nt][e]) ? ex2((sv[nt * 2 + e] - next) * LOG2E) : 0.f;
                    pr[nt][2 * i + e] = pv;
                    pair += pv;
                }
                sum += pair;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            l[i] = l[i] * alpha[i] + sum;
            m[i] = next;
        }
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            o[n][0] *= alpha[0];
            o[n][1] *= alpha[0];
            o[n][2] *= alpha[1];
            o[n][3] *= alpha[1];
        }
        // p.v
        const uint4 pa = make_uint4(cvt2(pr[0][0], pr[0][1]), cvt2(pr[0][2], pr[0][3]), cvt2(pr[1][0], pr[1][1]),
                                    cvt2(pr[1][2], pr[1][3]));
        uint32_t vs[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) vs[c] = scale_bf16(vraw[c]);
        const uint32_t s01 = vs[0] | (vs[1] << 16), s89 = vs[2] | (vs[3] << 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t a0 = i == 0 ? vw[0].x : i == 1 ? vw[0].y : i == 2 ? vw[0].z : vw[0].w;
            const uint32_t a1 = i == 0 ? vw[1].x : i == 1 ? vw[1].y : i == 2 ? vw[1].z : vw[1].w;
            const uint32_t a8 = i == 0 ? vw[2].x : i == 1 ? vw[2].y : i == 2 ? vw[2].z : vw[2].w;
            const uint32_t a9 = i == 0 ? vw[3].x : i == 1 ? vw[3].y : i == 2 ? vw[3].z : vw[3].w;
#pragma unroll
            for (int bb = 0; bb < 4; ++bb) {
                const uint32_t sel = bb | ((4 + bb) << 8);
                const uint32_t x01 = __byte_perm(a0, a1, sel), x89 = __byte_perm(a8, a9, sel);
                const int n = 2 * (4 * i + bb);
                mma(o[n], pa, deq(nib_lo(x01), s01), deq(nib_lo(x89), s89));
                mma(o[n + 1], pa, deq(nib_hi(x01), s01), deq(nib_hi(x89), s89));
            }
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[nt][c] = acc2[nt][c];
    }

    // output column n of 8-value tile t is value 32 n + t: lane j holds values [64 j, 64 j + 64) of rows gid, gid + 8
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (!ok_r[i]) continue;
        const int64_t row = ((int64_t)chunk * W + node_r[i]) * H + head_r[i];
        float* out = PO + row * D + 64 * j;
#pragma unroll
        for (int half = 0; half < 2; ++half)
#pragma unroll
            for (int t = 0; t < 32; t += 4)
                *reinterpret_cast<float4*>(out + 32 * half + t) =
                    make_float4(o[t][2 * i + half], o[t + 1][2 * i + half], o[t + 2][2 * i + half],
                                o[t + 3][2 * i + half]);
        if (j == 0) {
            PM[row] = m[i];
            PL[row] = l[i];
        }
    }
}

}  // namespace

void attn4_shared(torch::Tensor q, torch::Tensor base, torch::Tensor off, torch::Tensor stream, torch::Tensor items,
                  torch::Tensor po, torch::Tensor pm, torch::Tensor pl, int64_t w, int64_t hk, double scale,
                  int64_t warps) {
    const int n_items = items.size(0);
    if (!n_items) return;
    const int h = q.size(1), g = h / hk;
    dim3 grid((n_items + warps - 1) / warps, hk);
    attn4_kernel<false><<<grid, 32 * warps, warps * 16 * 32 * sizeof(uint4), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), base.data_ptr<int8_t>(), off.data_ptr<int64_t>(),
        stream.data_ptr<int>(), items.data_ptr<int>(), n_items, nullptr, nullptr, nullptr, nullptr, nullptr,
        po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), w, h, hk, g, (float)scale);
}

void attn4_tail(torch::Tensor q, torch::Tensor base, torch::Tensor off, torch::Tensor stream, torch::Tensor rows,
                torch::Tensor paths, torch::Tensor depths, torch::Tensor kn, torch::Tensor vn, torch::Tensor po,
                torch::Tensor pm, torch::Tensor pl, int64_t w, int64_t hk, double scale, int64_t tails) {
    const int h = q.size(1), g = h / hk;
    dim3 grid(w, hk, tails);
    attn4_kernel<true><<<grid, 32, 16 * 32 * sizeof(uint4), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), base.data_ptr<int8_t>(), off.data_ptr<int64_t>(),
        stream.data_ptr<int>(), nullptr, 0, rows.data_ptr<int>(), paths.data_ptr<int>(), depths.data_ptr<int>(),
        kn.data_ptr<int8_t>(), vn.data_ptr<int8_t>(), po.data_ptr<float>(), pm.data_ptr<float>(),
        pl.data_ptr<float>(), w, h, hk, g, (float)scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shared", &attn4_shared);
    m.def("tail", &attn4_tail);
}
