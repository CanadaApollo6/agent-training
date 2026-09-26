"""Part 2: your code. Fill in the two functions, then run

    uv run 01-inference/token-walkthrough/check.py

The checker feeds each function real vectors and real weights from Qwen3.5-0.8B, and compares your output with the
model's own module.

Tools you'll need (x and w are torch tensors):
    x @ W.T              matrix-vector product: W is [out, in], x is [..., in] -> result is [..., out]
    x * y, x + 1         elementwise, like numpy
    x.pow(2)             square every element
    x.mean(-1, keepdim=True)   average over the last dimension (the 1024 numbers), keeping it as a column
    torch.rsqrt(x)       1 / sqrt(x)
    torch.sigmoid(x)     1 / (1 + e^-x)
    x.float(), x.to(torch.bfloat16)   convert precision
"""
import torch


def rmsnorm(x, weight, eps=1e-6):
    """Rescale each 1024-number vector so its root-mean-square is 1, then scale each position by a learned amount.

    x:      [..., 1024] bf16, a vector from the residual stream
    weight: [1024] bf16, learned. Qwen3.5 stores it "zero-centered": the scale actually applied is (1 + weight).
    Do the arithmetic in fp32 (.float()) and return bf16, as the model does.

    Steps: rms = sqrt(mean of x squared + eps); out = x / rms * (1 + weight)
    """
    raise NotImplementedError


def mlp(x, w_gate, w_up, w_down):
    """The feed-forward block: widen to 3584 numbers, gate, and narrow back to 1024.

    x:      [..., 1024] bf16
    w_gate: [3584, 1024]   w_up: [3584, 1024]   w_down: [1024, 3584]

    Steps: gate = x through w_gate; up = x through w_up;
           h = silu(gate) * up, where silu(z) = z * sigmoid(z);
           out = h through w_down
    Stay in bf16 throughout (no .float()), as the model does.
    """
    raise NotImplementedError
