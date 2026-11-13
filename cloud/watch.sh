#!/usr/bin/env bash
# Background watcher: every 20 min pull results (backup), exit as soon as the pod reports a new milestone
# (scheduler job finished, or a generation chunk written). Re-launch after handling the event.
# Usage: bash cloud/watch.sh [max_hours]
cd "$(dirname "$0")/.."
max=$(( ${1:-6} * 3 ))
sig() { bash cloud/pod.sh run "grep -c ' done ' /workspace/logs/scheduler.log; cat /workspace/logs/gen_*.log /workspace/logs/job_*gen_*.log 2>/dev/null | grep -a -c -E '^(default|low|medium|high) chunk'" 2>/dev/null | tr '\n' ' '; }
base=$(sig)
echo "start $(date +%H:%M) baseline: $base"
for i in $(seq 1 "$max"); do
  sleep 1200
  bash cloud/pod.sh pull > /dev/null 2>&1
  now=$(sig)
  if [ -n "$now" ] && [ "$now" != "$base" ]; then
    echo "event $(date +%H:%M): $base -> $now"
    bash cloud/pod.sh run "tail -4 /workspace/logs/scheduler.log; grep -a -h -E '^(default|low|medium|high) chunk' /workspace/logs/gen_*.log /workspace/logs/job_*gen_*.log"
    exit 0
  fi
done
echo "no event within ${1:-6}h"
