"""RMSNorm on a vector small enough to check by hand. Change the numbers in x and run it again.

    uv run 01-inference/token-walkthrough/try_rmsnorm.py
"""
import torch

x = torch.tensor([3.0, 4.0])    # a "vector" of 2 numbers instead of 1024

squared = x.pow(2)              # square every number:        [9, 16]
mean_sq = squared.mean()        # average them:               12.5
scale = torch.rsqrt(mean_sq)    # 1 / square root of that:    1 / 3.536 = 0.2828
out = x * scale                 # multiply every number by it

print("x        ", x)
print("squared  ", squared)
print("mean_sq  ", mean_sq)
print("scale    ", scale)
print("out      ", out)
print("rms(out) ", out.pow(2).mean().sqrt(), "<- always 1: that's the point of RMSNorm")
