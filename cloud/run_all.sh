#!/usr/bin/env bash
# Pod orchestrator: downloads models, then runs generation jobs (one per GPU), then hidden-state extraction + HF bench.
# Usage on pod: cd /workspace/claude && nohup bash cloud/run_all.sh > /workspace/logs/run_all.log 2>&1 &
set -uo pipefail
cd /workspace/claude
export HF_HOME=/workspace/hf VLLM_WORKER_MULTIPROC_METHOD=spawn
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p /workspace/logs runs/gen runs/hidden runs/bench
rm -f /workspace/logs/gpu*.busy

# 1) models (gpt-oss first: biggest and slowest generation job)
MODELS="openai/gpt-oss-120b deepseek-ai/DeepSeek-R1-0528-Qwen3-8B allenai/Olmo-3-7B-Think Skywork/Skywork-OR1-Math-7B Qwen/Qwen3.5-4B"
bash cloud/download_models.sh > /workspace/logs/download.log 2>&1

# 2) generation, at most one job per GPU
JOBS=("openai/gpt-oss-120b low,medium,high 6" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B default 4"
      "allenai/Olmo-3-7B-Think default 4" "Skywork/Skywork-OR1-Math-7B default 4" "Qwen/Qwen3.5-4B default 4")
for job in "${JOBS[@]}"; do
  set -- $job
  # take the first free GPU
  gpu=""
  while [ -z "$gpu" ]; do
    for g in $(seq 0 $((NGPU - 1))); do [ -f "/workspace/logs/gpu$g.busy" ] || { gpu=$g; break; }; done
    [ -z "$gpu" ] && sleep 30
  done
  touch "/workspace/logs/gpu$gpu.busy"
  ( CUDA_VISIBLE_DEVICES=$gpu /workspace/venv-vllm/bin/python cloud/gen_solve.py --model "$1" --efforts "$2" --n "$3" \
      --outdir runs/gen > "/workspace/logs/gen_$(echo $1 | tr / _).log" 2>&1; rm -f "/workspace/logs/gpu$gpu.busy" ) &
done
wait

# 3) hidden states + HF runtime bench (runtime-mirror venv)
for m in $MODELS; do
  eff=default; [[ "$m" == openai/* ]] && eff=low,medium,high
  /workspace/venv-rt/bin/python cloud/extract_hidden.py --model "$m" --efforts "$eff" --out runs/hidden \
    > "/workspace/logs/hidden_$(echo $m | tr / _).log" 2>&1
  e1=${eff%%,*}
  /workspace/venv-rt/bin/python cloud/bench_hf.py --model "$m" --effort "$e1" --batches 32,64 --new-tokens 2048 > "runs/bench/$(echo $m | tr / _).jsonl" 2>/workspace/logs/bench_$(echo $m | tr / _).err
done
echo ALL_DONE
