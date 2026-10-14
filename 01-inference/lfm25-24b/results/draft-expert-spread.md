# How much a multi-token check costs (2026-10-14)

`ref.py --routes` on the MLX-4bit weights, teacher-forced on a 443-token chat whose answer is a Python script
(`gguf_bytes.py`). For each window of n consecutive answer tokens: distinct experts per MoE layer, averaged over the
38 MoE layers. Bytes per check = always-on 0.66 GB + 0.20 GB per distinct expert slot (QAD-Q4_0 sizes).

| tokens checked | distinct experts / layer | if routing were random | GB read | cost vs 1 token |
|---|---|---|---|---|
| 1 | 4.00 | 4.00 | 1.47 | 1.00x |
| 2 | 6.82 | 7.75 | 2.04 | 1.39x |
| 3 | 9.19 | 11.27 | 2.51 | 1.71x |
| 4 | 11.24 | 14.56 | 2.93 | 2.00x |
| 8 | 17.93 | 25.81 | 4.28 | 2.92x |
| 16 | 25.52 | 41.21 | 5.81 | 3.96x |

Neighbouring tokens share some experts (11.2 vs 14.6 at 4), but a 4-token check still reads twice the bytes of one
token. A drafted round that checks 4 rows pays off only if it yields more than 2 tokens on average. On Ornith the
always-on path dominated, so checks were nearly free. Here drafting is a modest lever; the serial path comes first.
