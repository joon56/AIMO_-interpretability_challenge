#!/usr/bin/env bash
# Stop the original single Qwen run right after its first chunk (problems 0-127) is saved;
# problems 128-321 are handled by shard jobs gen_qwen_B/C in the scheduler queue.
until grep -a -q '^default chunk 0' /workspace/logs/gen_Qwen_Qwen3.5-4B.log; do sleep 20; done
pkill -f '[g]en_solve.py --model Qwen'
sleep 15
for p in $(pgrep -f 'VLLM::[E]ngineCore'); do
  [ "$(ps -o ppid= -p "$p" | tr -d ' ')" = 1 ] && kill "$p" && echo "killed orphan $p"
done
echo "qwen handoff done $(date -u +%H:%M)"
