#!/bin/bash
# Builds vendor/verifiers: verifiers main at VF_COMMIT plus patches/verifiers-prime-agent-rust-0.10.0.patch, which is
# upstream PR #2776 (prime-agent harness: install the Rust release shape, open as of 2026-10-09) with its pin moved
# from 0.9.9-beta.45 to the v0.10.0 release and that release's SHA256SUMS rows. Upstream verifiers still installs the
# TypeScript prime-agent 0.9.5. Drop this once #2776 (or a 0.10.x pin) lands: point pyproject back at git.
set -euo pipefail
cd "$(dirname "$0")"
VF_COMMIT=50ccf5a88a671fcda1204d2e18931cced5d230d9
rm -rf vendor/verifiers && mkdir -p vendor
git clone -q https://github.com/PrimeIntellect-ai/verifiers vendor/verifiers
git -C vendor/verifiers checkout -q $VF_COMMIT
git -C vendor/verifiers apply "$PWD/patches/verifiers-prime-agent-rust-0.10.0.patch"
grep -q 'RUST_VERSION = "0.10.0"' vendor/verifiers/verifiers/v1/harnesses/prime_agent/harness.py
echo "vendor/verifiers: $VF_COMMIT + prime-agent 0.10.0 (Rust)"
