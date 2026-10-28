Fix six observed Nanbeige parity failures at 3aeb8e97 over upstream 31ce9eb3.
Keep the accepted fd409155 EXL3 arithmetic, precision and lane ownership unchanged.
Target CUDA sm_86 on the shared RTX 3090, rank 1, original tokenizer and weights read-only.
The server profile, shared HF renderer, XML parser and stop window own the fixes.
Use the existing profile flag, normalization validation, schema decoder and stop hook.
HF and the scored pod generate expected fixtures. Test the actual JSON and SSE handlers.
Reproduce before production edits. Keep the Unicode stop correction independently cherry-pickable.
Rerun P2/P2b host and slice gates. Compare broad host results with a fresh 31ce9eb3 archive.
All device work goes through llama_gpu_run.py with a measured peak estimate and 2048 MiB reserve.
Full-model gates need at least 19.5 GB free. No other job or process may be modified.
One writer owns the coupled server changes. No delegation, issues, PRs or pushes are authorized.
ATC is unavailable locally. The user's explicit sole-writer branch assignment owns this scope.
