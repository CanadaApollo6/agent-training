// Tree attention over a 4-bit cache (kvq BITS=4, head dim 256): a block of four warps per 512-key chunk and up to
// 48 (row, head) pairs, MT tiles of 16.
//
// The 3090 runs int->float and float->bf16 conversions at 1/8 of its float rate and Triton's kernel stages every
// dequantized tile through shared memory between barriers. Here each lane reads packed codes straight from the
// cache into its mma.sync fragments: a code nibble becomes bf16 by bit placement (0x3F80 | c << 3 = 1 + c/16), two
// packed bf16 FMAs make (c - 7.5)/16 exactly and then times bf16(2 * scale). The head's 256 values are assigned to
// MMA slots so that a lane's share is one contiguous run of a row: 32 bytes of a key row for q.k (the queries
// follow the same order) and 4 bytes of one scale group of four value rows for p.v (the output is written back in
// order). A step is 32 keys: each warp scores 8 of them for every tile, the block agrees on each row's maximum and
// sum through shared memory and each warp accumulates a quarter of the values. A key or value is loaded and
// decoded once for all the block's rows, and each row's arithmetic is the same whatever the tiling. Shared chunks
// and the tail (the last committed keys plus each row's own path) run the same code per step, so a full chunk gives
// identical bits in either and drafted rows equal serial ones.

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

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// shared memory (4-byte words): q fragments (MT tiles x 16 k-steps x 32 lanes x 4), row maxima and sums of each
// warp's keys (2 x 4 warps x 16 MT), probabilities as p.v A fragment halves (MT x 4 warps x 32 lanes x 2)
template <int MT>
constexpr int smem_words() { return MT * 16 * 32 * 4 + 2 * 4 * 16 * MT + MT * 4 * 32 * 2; }

template <bool TAIL, int MT>
__global__ void __launch_bounds__(128, MT == 3 ? 3 : 4) attn4_kernel(const __nv_bfloat16* __restrict__ Q,
        const int8_t* __restrict__ base, const int64_t* __restrict__ OFF, const int* __restrict__ STREAM,
        const int* __restrict__ ITEMS, int n_items, const int* __restrict__ ROWS, const int* __restrict__ PATHS,
        const int* __restrict__ DEPTHS, const int8_t* __restrict__ KN, const int8_t* __restrict__ VN,
        float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL, int W, int H, int HK, int G,
        float scale) {
    extern __shared__ uint4 smem[];
    uint4* qf = smem;
    float* rmax = reinterpret_cast<float*>(smem + MT * 16 * 32);            // [warp][16 MT]
    float* rsum = rmax + 4 * 16 * MT;
    uint2* pbuf = reinterpret_cast<uint2*>(rsum + 4 * 16 * MT);           // [MT][warp][lane]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, j = lane & 3;
    const int hk = TAIL ? blockIdx.y : blockIdx.x, bx = TAIL ? blockIdx.x : blockIdx.y;
    int s, chunk, p, steps, depth = 0, node_t = 0, first = 0, start = 0, rows = 0;
    if (!TAIL) {
        // the plan's 16-row tiles: a block takes MT of them from a multiple of MT. Blocks go to items 0, MT, 2 MT ..
        // first (the ones that work when a chunk's tiles are a multiple of MT), so the scheduler spreads them
        const int nrow = (n_items + MT - 1) / MT, item = (bx % nrow) * MT + bx / nrow;
        if (item >= n_items) return;
        s = ITEMS[item * 3];
        first = ITEMS[item * 3 + 1];
        if (first % (16 * MT)) return;
        chunk = ITEMS[item * 3 + 2];
        start = STREAM[s * 4];
        rows = STREAM[s * 4 + 1];
        p = STREAM[s * 4 + 2];
        if ((chunk + 1) * CH > p) return;                       // a plan padded for a longer context (a graph's)
        steps = CH / 32;
    } else {
        node_t = bx;
        s = ROWS[node_t];
        p = STREAM[s * 4 + 2];
        const int nch = STREAM[s * 4 + 3];
        chunk = p / CH + blockIdx.z;
        if (chunk >= nch) return;
        depth = DEPTHS[node_t];
        const int last = min(p + depth - chunk * CH, CH);      // past it every key is masked: steps there are no-ops
        steps = (last + 31) / 32;
    }
    const int64_t koff = OFF[s * 2], voff = OFF[s * 2 + 1];

    // row r of the block (0 .. 16 MT): its (node, head) and whether real
    auto row_info = [&](int r, int& node, int& head) -> bool {
        if (!TAIL) {
            const int pr = first + r;
            node = start + pr / G;
            head = hk * G + pr % G;
            return pr < rows * G;
        }
        node = node_t;
        head = hk * G + r;
        return r < G;
    };

    // queries as A fragments in the q.k slot order
    for (int idx = threadIdx.x; idx < MT * 16 * 32; idx += 128) {
        const int mt = idx >> 9, st = (idx >> 5) & 15, ln = idx & 31, g = ln >> 2, jj = ln & 3;
        const int d0 = 64 * jj + 8 * (st >> 1) + 4 * (st & 1);
        uint32_t lo[2], hi[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int node, head;
            uint2 v = make_uint2(0u, 0u);
            if (row_info(16 * mt + g + 8 * i, node, head))
                v = *reinterpret_cast<const uint2*>(Q + ((int64_t)node * H + head) * D + d0);
            lo[i] = __byte_perm(v.x, v.y, 0x5410);               // elements d0, d0 + 2
            hi[i] = __byte_perm(v.x, v.y, 0x7632);               // d0 + 1, d0 + 3
        }
        qf[idx] = make_uint4(lo[0], lo[1], hi[0], hi[1]);
    }
    __syncthreads();

    float o[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < 8; ++n) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
    float m[MT][2], l[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) m[mt][0] = m[mt][1] = -INFINITY, l[mt][0] = l[mt][1] = 0.f;
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
    // q.k: this warp's key row is key0 + 8 warp + gid, bytes [32 j, 32 j + 32) and 2 scales
    uint4 kw[2];
    uint32_t ks;
    auto load_k = [&](int key0) {
        bool valid;
        const int8_t* row = row_of(key0 + 8 * warp + gid, koff, KN, valid);
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
        const int key0 = chunk * CH + step * 32;
        // p.v rows, loaded now and used after q.k: of each 16 keys, 2j, 2j+1, 2j+8, 2j+9, bytes 16 gid + 4 warp ..
        uint32_t vw[2][4], vraw[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                bool valid;
                const int8_t* row = row_of(key0 + 16 * h + 2 * j + (c & 1) + 8 * (c >> 1), voff, VN, valid);
                const uint32_t a = ld4(row + 16 * gid + 4 * warp);
                const uint32_t sc = ld2(row + 128 + 2 * gid);
                vw[h][c] = valid ? a : 0u;
                vraw[h][c] = valid ? sc : 0u;
            }
        float acc[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
        {
            const uint32_t a = scale_bf16(ks), b = scale_bf16(ks >> 16);
            const uint32_t sk[2] = {a | (a << 16), b | (b << 16)};
#pragma unroll
            for (int u = 0; u < 8; ++u)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t x = __byte_perm(word(kw[u >> 2], u & 3), 0u, h ? 0x3322 : 0x1100);
                    const uint32_t b0 = deq(nib_lo(x, kc), sk[u >> 2], kc), b1 = deq(nib_hi(x, kc), sk[u >> 2], kc);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) mma(acc[mt], qf[(mt * 16 + 2 * u + h) * 32 + lane], b0, b1);
                }
        }
        load_k(chunk * CH + min(step + 1, steps - 1) * 32);     // the next step's key, during softmax and p.v
        // this warp's keys: key0 + 8 warp + 2j + e (C columns 2j, 2j+1)
        bool cvalid[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int key = key0 + 8 * warp + 2 * j + e;
            cvalid[e] = key < p || (TAIL && key - p < depth);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                float mx = -INFINITY;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    acc[mt][2 * i + e] = cvalid[e] ? acc[mt][2 * i + e] * scale : -INFINITY;
                    mx = fmaxf(mx, acc[mt][2 * i + e]);
                }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                if (j == 0) rmax[warp * 16 * MT + 16 * mt + gid + 8 * i] = mx;
            }
        __syncthreads();
        // every warp forms the same new maximum; probabilities of its own keys, their sums
        float alpha[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            uint32_t pw[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = 16 * mt + gid + 8 * i;
                const float mx = fmaxf(fmaxf(rmax[r], rmax[16 * MT + r]),
                                       fmaxf(rmax[2 * 16 * MT + r], rmax[3 * 16 * MT + r]));
                const bool active = mx != -INFINITY;
                const float next = active ? fmaxf(m[mt][i], mx) : m[mt][i];
                alpha[mt][i] = active ? (m[mt][i] == -INFINITY ? 0.f : ex2((m[mt][i] - next) * LOG2E)) : 1.f;
                m[mt][i] = next;
                float pv[2];
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    pv[e] = (active && cvalid[e]) ? ex2((acc[mt][2 * i + e] - next) * LOG2E) : 0.f;
                float sum = pv[0] + pv[1];
                sum += __shfl_xor_sync(0xffffffffu, sum, 1);
                sum += __shfl_xor_sync(0xffffffffu, sum, 2);
                if (j == 0) rsum[warp * 16 * MT + r] = sum;
                pw[i] = cvt2(pv[0], pv[1]);
            }
            pbuf[(mt * 4 + warp) * 32 + lane] = make_uint2(pw[0], pw[1]);
        }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = 16 * mt + gid + 8 * i;
                const float sum = (rsum[r] + rsum[16 * MT + r]) + (rsum[2 * 16 * MT + r] + rsum[3 * 16 * MT + r]);
                l[mt][i] = l[mt][i] * alpha[mt][i] + sum;
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    o[mt][n][2 * i] *= alpha[mt][i];
                    o[mt][n][2 * i + 1] *= alpha[mt][i];
                }
            }
        // p.v over this warp's 8 value tiles, the step's 32 keys as two k-steps
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 pa[MT];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint2 x = pbuf[(mt * 4 + 2 * h) * 32 + lane], y = pbuf[(mt * 4 + 2 * h + 1) * 32 + lane];
                pa[mt] = make_uint4(x.x, x.y, y.x, y.y);
            }
            const uint32_t s01 = scale_bf16(vraw[h][0]) | (scale_bf16(vraw[h][1]) << 16);
            const uint32_t s89 = scale_bf16(vraw[h][2]) | (scale_bf16(vraw[h][3]) << 16);
#pragma unroll
            for (int bb = 0; bb < 4; ++bb) {
                const uint32_t sel = bb | ((4 + bb) << 8);
                const uint32_t x01 = __byte_perm(vw[h][0], vw[h][1], sel), x89 = __byte_perm(vw[h][2], vw[h][3], sel);
                const uint32_t l0 = deq(nib_lo(x01, kc), s01, kc), l1 = deq(nib_lo(x89, kc), s89, kc);
                const uint32_t h0 = deq(nib_hi(x01, kc), s01, kc), h1 = deq(nib_hi(x89, kc), s89, kc);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma(o[mt][2 * bb], pa[mt], l0, l1);
                    mma(o[mt][2 * bb + 1], pa[mt], h0, h1);
                }
            }
        }
    }

    // value tile t (of 32) column n is value 32 n + t; this warp has tiles 8 warp .. 8 warp + 7, so lane j holds
    // values 64 j + 32 e + 8 warp + [0, 8) of rows gid, gid + 8 of each tile (e: the column's parity)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int node, head;
            if (!row_info(16 * mt + gid + 8 * i, node, head)) continue;
            const int64_t row = ((int64_t)chunk * W + node) * H + head;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float* out = PO + row * D + 64 * j + 32 * e + 8 * warp;
                const int c = 2 * i + e;
                *reinterpret_cast<float4*>(out) = make_float4(o[mt][0][c], o[mt][1][c], o[mt][2][c], o[mt][3][c]);
                *reinterpret_cast<float4*>(out + 4) =
                    make_float4(o[mt][4][c], o[mt][5][c], o[mt][6][c], o[mt][7][c]);
            }
            if (j == 0 && warp == 0) {
                PM[row] = m[mt][i];
                PL[row] = l[mt][i];
            }
        }
}

template <bool TAIL, int MT>
void launch(dim3 grid, torch::Tensor q, torch::Tensor base, torch::Tensor off, torch::Tensor stream,
            const int* items, int n_items, const int* rows, const int* paths, const int* depths, const int8_t* kn,
            const int8_t* vn, torch::Tensor po, torch::Tensor pm, torch::Tensor pl, int w, int hk, float scale) {
    const int h = q.size(1), g = h / hk, bytes = smem_words<MT>() * 4;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(attn4_kernel<TAIL, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        attr = true;
    }
    attn4_kernel<TAIL, MT><<<grid, 128, bytes, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), base.data_ptr<int8_t>(), off.data_ptr<int64_t>(),
        stream.data_ptr<int>(), items, n_items, rows, paths, depths, kn, vn, po.data_ptr<float>(), pm.data_ptr<float>(),
        pl.data_ptr<float>(), w, h, hk, g, scale);
}

}  // namespace

// items: the plan's (stream, first row, chunk) 16-row tiles; a block takes mt of them from each multiple of mt
void attn4_shared(torch::Tensor q, torch::Tensor base, torch::Tensor off, torch::Tensor stream, torch::Tensor items,
                  torch::Tensor po, torch::Tensor pm, torch::Tensor pl, int64_t w, int64_t hk, double scale,
                  int64_t mt) {
    const int n = items.size(0);
    if (!n) return;
    const dim3 grid(hk, (n + mt - 1) / mt * mt);                         // kv heads inner: working blocks first
    const int* it = items.data_ptr<int>();
    if (mt == 1) launch<false, 1>(grid, q, base, off, stream, it, n, nullptr, nullptr, nullptr, nullptr, nullptr, po, pm, pl, w, hk, scale);
    else if (mt == 2) launch<false, 2>(grid, q, base, off, stream, it, n, nullptr, nullptr, nullptr, nullptr, nullptr, po, pm, pl, w, hk, scale);
    else launch<false, 3>(grid, q, base, off, stream, it, n, nullptr, nullptr, nullptr, nullptr, nullptr, po, pm, pl, w, hk, scale);
}

void attn4_tail(torch::Tensor q, torch::Tensor base, torch::Tensor off, torch::Tensor stream, torch::Tensor rows,
                torch::Tensor paths, torch::Tensor depths, torch::Tensor kn, torch::Tensor vn, torch::Tensor po,
                torch::Tensor pm, torch::Tensor pl, int64_t w, int64_t hk, double scale, int64_t tails) {
    launch<true, 1>(dim3(w, hk, tails), q, base, off, stream, nullptr, 0, rows.data_ptr<int>(), paths.data_ptr<int>(),
                    depths.data_ptr<int>(), kn.data_ptr<int8_t>(), vn.data_ptr<int8_t>(), po, pm, pl, w, hk, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shared", &attn4_shared);
    m.def("tail", &attn4_tail);
}
