#!/usr/bin/env bash
# B60 (BMG/INTEL_XE2) matmul tile-clamp regression test.
# On ANV, the large-tile matmul dispatch selected for ubatch > 64 deadlocks the compute engine (job timeout, device lost).
# The INTEL_XE2 clamp in matmul_tile_selector forces the medium tile, which executes correctly.
# Requires: a BMG Vulkan device and llama-bench.
set -u
BENCH=${BENCH:-$(dirname "$0")/../build/bin/llama-bench}

if [[ ! -x "$BENCH" ]]; then
    echo "SKIP: no llama-bench binary at $BENCH (set BENCH)"
    exit 0
fi

VK_DEV=$("$BENCH" --list-devices 2>/dev/null | grep -i "Vulkan" | grep -i "BMG" | grep -oE "Vulkan[0-9]+" | head -1)
if [[ -z "$VK_DEV" ]]; then
    echo "SKIP: no BMG (Intel Xe2) Vulkan device"
    exit 0
fi

MODEL=${MODEL:?set MODEL to a .gguf path}

# control: ubatch 64 has always selected the medium tile
timeout 180 "$BENCH" -dev "$VK_DEV" -m "$MODEL" -ngl 99 -p 128 -n 0 -b 512 -ub 64 -r 1 >/dev/null 2>&1 \
    || { echo "FAIL: ub64 control did not complete"; exit 1; }

# regression: ubatch 512 previously wedged; with the clamp it must complete
timeout 180 "$BENCH" -dev "$VK_DEV" -m "$MODEL" -ngl 99 -p 512 -n 0 -b 512 -ub 512 -r 1 >/dev/null 2>&1 \
    || { echo "FAIL: ub512 wedged or timed out (large-tile path taken)"; exit 1; }

echo "PASS: ub64 control and clamped ub512 both completed on $VK_DEV"
