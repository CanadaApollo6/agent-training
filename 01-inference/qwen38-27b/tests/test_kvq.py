"""8-bit KV cache checks: quantization error, and prefill attention reading packed rows vs bf16 rows."""
import os
os.environ["TENSORFOLD_KV_BITS"] = "8"
import torch
from tensorfold.cuda.kernels import kvq
from tensorfold.cuda.kernels.prefill_attention import attention

torch.manual_seed(0)
dev = "cuda"
H, HK, D = 24, 4, 256
for n, outlier in ((1, 1), (7, 40), (4096, 1), (4096, 40)):
    x = torch.randn(n, HK, D, device=dev, dtype=torch.bfloat16) * 3
    x[:, :, 5] *= outlier                              # an outlier channel, as real keys have
    rows = torch.empty(n, HK, kvq.width(D), device=dev, dtype=torch.int8)
    kvq.store(rows, x)
    y = kvq.unpack(rows, D)
    rel = ((y.float() - x.float()).norm() / x.float().norm()).item()
    print(f"round trip n={n} outlier x{outlier}: relative error {rel:.4f}")
    assert rel < 0.02

def ref(q, k, v, p0):
    W = q.shape[0]; T = p0 + W
    kk = k[:T].float().repeat_interleave(H // HK, 1); vv = v[:T].float().repeat_interleave(H // HK, 1)
    s = torch.einsum("whd,thd->hwt", q.float(), kk) * D ** -0.5
    mask = torch.arange(T, device=dev)[None, :] > (p0 + torch.arange(W, device=dev))[:, None]
    s.masked_fill_(mask[None], float("-inf"))
    return torch.einsum("hwt,thd->whd", s.softmax(-1), vv)

for p0, W in ((0, 100), (300, 64), (5000, 777)):
    T = p0 + W
    k = torch.randn(T, HK, D, device=dev, dtype=torch.bfloat16); k[:, :, 3] *= 20
    v = torch.randn(T, HK, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(W, H, D, device=dev, dtype=torch.bfloat16)
    kq = torch.empty(T, HK, kvq.width(D), device=dev, dtype=torch.int8); vq = torch.empty_like(kq)
    kvq.store(kq, k); kvq.store(vq, v)
    out8 = attention(q, kq, vq, p0, scale=D ** -0.5).float()
    want8 = ref(q, kvq.unpack(kq, D), kvq.unpack(vq, D), p0)   # same math on the dequantized cache
    want16 = ref(q, k, v, p0)                                 # the bf16 cache
    kern = ((out8 - want8).norm() / want8.norm()).item()
    quant = ((out8 - want16).norm() / want16.norm()).item()
    print(f"prefill p0={p0} W={W}: kernel vs dequant ref {kern:.4f}, 8-bit vs bf16 cache {quant:.4f}")
    assert kern < 0.01 and quant < 0.05      # quant: random data with a x20 key channel; real text is gated by KL
print("ok")
