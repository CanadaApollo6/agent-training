"""Does the tensor core give the same bits for C = A B as for (B^T A^T)^T? (bf16 m16n8k16, fp32 accumulate)

TensorFold's decode kernel put tokens in the mma's A (16 rows) and weights in B, wasting 15 of 16 rows at batch 1.
Swapping them halves the mmas, and keeps the decode form's exact bits only if the answer here is yes.
On the 3090 (sm_86): 0 mismatches in 30.7M outputs, with nearly every output rounded.
Usage: python mma_operand_symmetry.py
"""

import torch
from torch.utils.cpp_extension import load_inline
cuda = r'''
#include <cuda_bf16.h>
__device__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// X [T][16 tok][KS*16] bf16, W [T][16 feat][KS*16] bf16. out1[t][tok][feat] via A=X (tok rows), B=W (feat 0..7 and 8..15 as two n-tiles)
// out2[t][tok][feat] via A=W (feat rows), B=X (tok 0..7 and 8..15 as two n-tiles). Chains KS k-steps through C.
__global__ void k(const unsigned* X, const unsigned* W, float* o1, float* o2, int KS) {
  const int lane = threadIdx.x, gq = lane >> 2, t = lane & 3, T = blockIdx.x;
  const int K2 = KS * 8;  // uint32 pairs per row
  const unsigned* x = X + (size_t)T * 16 * K2; const unsigned* w = W + (size_t)T * 16 * K2;
  float c1[2][4] = {}, c2[2][4] = {};
  for (int ks = 0; ks < KS; ++ks) {
    const int p0 = ks * 8 + t, p1 = ks * 8 + t + 4;
    // A=X
    for (int j = 0; j < 2; ++j)
      mma(c1[j], x[gq * K2 + p0], x[(gq + 8) * K2 + p0], x[gq * K2 + p1], x[(gq + 8) * K2 + p1],
          w[(8 * j + gq) * K2 + p0], w[(8 * j + gq) * K2 + p1]);
    for (int j = 0; j < 2; ++j)
      mma(c2[j], w[gq * K2 + p0], w[(gq + 8) * K2 + p0], w[gq * K2 + p1], w[(gq + 8) * K2 + p1],
          x[(8 * j + gq) * K2 + p0], x[(8 * j + gq) * K2 + p1]);
  }
  float* O1 = o1 + (size_t)T * 256; float* O2 = o2 + (size_t)T * 256;
  for (int j = 0; j < 2; ++j) {
    // c1: rows tok gq/gq+8, cols feat 8j+2t,+1
    O1[gq * 16 + 8 * j + 2 * t] = c1[j][0]; O1[gq * 16 + 8 * j + 2 * t + 1] = c1[j][1];
    O1[(gq + 8) * 16 + 8 * j + 2 * t] = c1[j][2]; O1[(gq + 8) * 16 + 8 * j + 2 * t + 1] = c1[j][3];
    // c2: rows feat gq/gq+8, cols tok 8j+2t,+1
    O2[(8 * j + 2 * t) * 16 + gq] = c2[j][0]; O2[(8 * j + 2 * t + 1) * 16 + gq] = c2[j][1];
    O2[(8 * j + 2 * t) * 16 + gq + 8] = c2[j][2]; O2[(8 * j + 2 * t + 1) * 16 + gq + 8] = c2[j][3];
  }
}
void run(torch::Tensor X, torch::Tensor W, torch::Tensor o1, torch::Tensor o2, int64_t KS) {
  k<<<X.size(0), 32>>>((const unsigned*)X.data_ptr(), (const unsigned*)W.data_ptr(), o1.data_ptr<float>(), o2.data_ptr<float>(), KS);
}
'''
m = load_inline("mmasym", cpp_sources="void run(torch::Tensor X, torch::Tensor W, torch::Tensor o1, torch::Tensor o2, int64_t KS);",
                cuda_sources=cuda, functions=["run"], extra_cuda_cflags=["-arch=sm_%d%d" % torch.cuda.get_device_capability(), "-O3"])
T, KS = 20000, 4
for trial, (xs, ws) in enumerate([(1, 1), (100, 1), (1e-3, 1)]):
    g = torch.Generator(device="cuda").manual_seed(trial)
    # x: wide-exponent bf16 values; w: integer levels 0..15 as bf16 (like nib2), plus a generic-bf16 variant
    x = (torch.randn(T, 16, KS * 16, device="cuda", generator=g) * torch.exp(torch.randn(T, 16, KS * 16, device="cuda", generator=g) * 3) * xs).to(torch.bfloat16)
    for kind in ("levels", "generic"):
        if kind == "levels":
            w = torch.randint(0, 16, (T, 16, KS * 16), device="cuda", generator=g).to(torch.bfloat16)
        else:
            w = (torch.randn(T, 16, KS * 16, device="cuda", generator=g) * torch.exp(torch.randn(T, 16, KS * 16, device="cuda", generator=g) * 3)).to(torch.bfloat16)
        o1 = torch.empty(T, 16, 16, device="cuda"); o2 = torch.empty_like(o1)
        m.run(x.contiguous(), w.contiguous(), o1, o2, KS)
        torch.cuda.synchronize()
        ref = torch.einsum("tik,tjk->tij", x.double(), w.double())
        same = (o1.view(torch.int32) == o2.view(torch.int32)) | (o1.isnan() & o2.isnan())
        print(f"x scale {xs:g} w {kind}: mismatches {(~same).sum().item()} of {same.numel()}; "
              f"inexact vs fp64 {(o1.double() != ref).float().mean().item():.3f}")
