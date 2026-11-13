#!/usr/bin/env bash
# Pull from all pods every 10 min until the final outputs exist (or 3 h), so nothing is lost when the balance ends.
cd "$(dirname "$0")/.."
done1=""; done2=""; done3=""
for i in $(seq 1 18); do
  [ -z "$done1" ] && { bash cloud/pod.sh pull runs/gen runs/progress runs/behavior > /dev/null 2>&1
    bash cloud/pod.sh run "grep -q 'done prog_deepseek rc=0' /workspace/logs/scheduler.log" 2>/dev/null && done1=1; }
  [ -z "$done2" ] && { POD_FILE=cloud/POD2 bash cloud/pod.sh pull runs/gen runs/progress runs/behavior > /dev/null 2>&1
    POD_FILE=cloud/POD2 bash cloud/pod.sh run "grep -q 'done p2_prog_olmo_AB rc=0' /workspace/logs/scheduler.log && grep -q 'done p2_beh_gptoss_high' /workspace/logs/scheduler.log" 2>/dev/null && done2=1; }
  [ -z "$done3" ] && { PULL_DIR=runs/pod3 POD_FILE=cloud/POD3 bash cloud/pod.sh pull runs/gen runs/progress > /dev/null 2>&1
    POD_FILE=cloud/POD3 bash cloud/pod.sh run "grep -q 'done p3_prog_olmo_B3 rc=0' /workspace/logs/scheduler.log" 2>/dev/null && done3=1; }
  echo "$(date +%H:%M) pod1=${done1:-0} pod2=${done2:-0} pod3=${done3:-0}"
  [ -n "$done1" ] && [ -n "$done2" ] && [ -n "$done3" ] && { echo ALL_DONE; exit 0; }
  sleep 600
done
echo "timeout (some pods unfinished)"
