# DeepSWE env

`../run_eval.sh deepswe ...` runs from here. It's a separate uv project because DeepSWE v1.1 needs Harbor's
separate-verifier support, which is in verifiers main (37e459a0) but not in the dev153 pin the TB2 runs use.

The hub package (`primeintellect/deep-swe`) points every task at `prime/prime/swe-bench-202605:<tag>`. That's Prime's
team registry, and other accounts get HTTP 403. The same images are public at `public.ecr.aws/d3j8x8q7/...`. However,
the Prime runtime only auto-builds VM images from Docker Hub refs. So we copied the images once into our own registry
and made them private:

    uvx prime images push-bulk --manifest images.jsonl --private     # ~1-2.4 GB each, a few minutes

`deep-swe-upstream/` is the env vf-eval loads (`deep-swe-upstream`). It has the same taskset and verifiers, with
images rewritten to `$DEEPSWE_IMAGE_NS` (default `prime/riel-stamand`). Prime allows 10 private images per account, so `../deepswe_tasks.txt` is 10 tasks (seed 0 of the
earlier seed-0 sample of 25). To swap tasks, delete an image (`uvx prime images delete <ref> --yes`), then add the
new one to `images.jsonl` and push.

## Terminal-Bench 4

`terminal-bench-4/` pins the generic Harbor taskset to `terminal-bench/terminal-bench@4.0.0` on Harbor Hub (66 tasks,
separate verifier sandboxes, 8 h agent timeouts that we override with our own caps). 11 tasks are multi-container and
3 need GPUs, so `../tb4_tasks.txt` samples 20 of the other 52 (seed 0). Their solver and verifier images are on Docker
Hub, which Prime auto-builds from. `tb4_images.jsonl` pre-builds them so the first rollouts don't wait:

    uvx prime images push-bulk --manifest tb4_images.jsonl

## Grading all changes, not just commits

Upstream DeepSWE grades only committed work (`pre_artifacts.sh` runs `git diff <base> HEAD`), and each instruction
says to commit. In the first baselines (2026-09-30) 1 run in 55 even mentioned `git commit`, so every patch was empty
and every score 0: the verifier graded the untouched repo (all existing tests passed, no new ones). Since then
`deep_swe_upstream` grades every change the agent made, committed or not (tracked edits plus new files under 1 MB,
dependency and build caches left out), and records on each trace:

- `committed_bytes` and `worktree_bytes`: the committed diff and the full diff.
- `committed_all`: 1 when the commit already held everything.

Official score = reward if `committed_all` is 1, else 0. Runs before the change are labelled `-a1`; reruns `-wt1`.
