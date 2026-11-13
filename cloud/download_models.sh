#!/usr/bin/env bash
# Download the five evaluated checkpoints into /workspace/hf (same HF cache layout as the eval worker).
# Xet transfer hung on RunPod (2026-10-26); plain HTTP gives ~140 MB/s per stream. Models fetched in parallel.
set -uo pipefail
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 PATH=/workspace/venv-tools/bin:$PATH
MODELS=${MODELS:-"Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B openai/gpt-oss-120b"}
for m in $MODELS; do
  (
    # gpt-oss: skip the duplicate original/ and metal/ folders, keep the HF MXFP4 safetensors only.
    if [[ "$m" == openai/gpt-oss-* ]]; then
      hf download "$m" --exclude "original/*" --exclude "metal/*" --max-workers 16 > /dev/null
    else
      hf download "$m" --max-workers 8 > /dev/null
    fi
    echo "done $m $(date +%T) rc=$?"
  ) &
done
wait
du -sh "$HF_HOME"/hub/* 2>/dev/null
