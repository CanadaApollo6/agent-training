| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| llama ?B Q3_K - Medium         |  14.82 GiB |    33.12 B | CUDA       |  99 |   1 |          pp2048 |        779.43 ± 1.80 |
| llama ?B Q3_K - Medium         |  14.82 GiB |    33.12 B | CUDA       |  99 |   1 |           tg128 |         27.20 ± 0.03 |
| llama ?B Q3_K - Medium         |  14.82 GiB |    33.12 B | CUDA       |  99 |   1 | pp2048 @ d16384 |        559.80 ± 0.34 |
| llama ?B Q3_K - Medium         |  14.82 GiB |    33.12 B | CUDA       |  99 |   1 |  tg128 @ d16384 |         23.07 ± 0.01 |
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| llama ?B IQ4_XS - 4.25 bpw     |  17.35 GiB |    33.12 B | CUDA       |  99 |   1 |          pp2048 |        920.21 ± 1.55 |
| llama ?B IQ4_XS - 4.25 bpw     |  17.35 GiB |    33.12 B | CUDA       |  99 |   1 |           tg128 |         25.61 ± 0.05 |
| llama ?B IQ4_XS - 4.25 bpw     |  17.35 GiB |    33.12 B | CUDA       |  99 |   1 | pp2048 @ d16384 |        629.79 ± 0.29 |
| llama ?B IQ4_XS - 4.25 bpw     |  17.35 GiB |    33.12 B | CUDA       |  99 |   1 |  tg128 @ d16384 |         21.77 ± 0.01 |
