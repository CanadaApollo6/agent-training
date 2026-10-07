Subject: FFT beta access, custom model caching for RL, and sandbox terminations

Hi Prime Intellect support,

I'm Riel St. Amand. Our team is building RL post-training on Prime, and I have two requests and one issue to report.

1. FFT beta access. Could you enable the full fine-tuning (FFT) hosted training beta for our team? Right now `prime train gpus` returns "No GPU types available. Contact support," and `prime train models --fft-only` returns an empty list (prime CLI v0.9.3).

2. Caching a custom model for RL. The docs say RL runs only start from models cached on the cluster, and RL from /volume models is rejected. We have a full fine-tune of Ornith 1.5 35B-A3B (architecture Qwen3_5MoeForConditionalGeneration, bf16, about 70 GB). We will publish it as a private Hugging Face repo, <HF repo to be added>, and can grant read access as needed. Could you cache it so we can run RL on it?

Context: we plan GRPO-style RL with prime-rl, using verifiers' prime_agent harness on Terminal-Bench-style tasks in Prime Sandboxes. We already ran a successful 3-step prime-rl smoke test on Ornith 9B with Prime Sandboxes, on outside compute.

3. Sandbox terminations. During our runs, many Prime Sandboxes were terminated mid-run with only "The sandbox has been terminated" and no reason (no OOM message). We also see occasional 502 Bad Gateway on exec. In one 87-task x 2 evaluation, 78 runs were lost on the first attempt and 37 on retry, concentrated on heavier tasks (compiling, training, rendering). What causes terminations without a reason? Are there limits we should configure (memory, CPU, disk, region, creation rate)? I can send sandbox IDs: <sandbox IDs>.

Thank you for your help.

Best,
Riel St. Amand
