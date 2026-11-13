#!/usr/bin/env bash
# Bootstrap an extra pod: envs, selected model weights, then the job scheduler with the given queue file.
# Usage on pod: MODELS="openai/gpt-oss-120b allenai/Olmo-3-7B-Think" bash cloud/bootstrap_pod.sh cloud/jobs_pod2.jsonl
set -uo pipefail
cd /workspace/claude
mkdir -p /workspace/logs /workspace/jobs
bash cloud/setup_pod.sh > /workspace/logs/setup.log 2>&1
bash cloud/download_models.sh > /workspace/logs/download.log 2>&1
cp "$1" /workspace/jobs/queue.jsonl
setsid nohup python3 cloud/scheduler.py > /workspace/logs/scheduler.log 2>&1 < /dev/null &
echo "bootstrap done $(date -u +%H:%M)"
