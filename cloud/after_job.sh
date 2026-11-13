#!/usr/bin/env bash
# Append a job line to the scheduler queue once all listed jobs (comma-separated ids) have finished OK.
# Usage: setsid nohup bash cloud/after_job.sh <id1,id2,...> '<json job line>' > log 2>&1 < /dev/null &
for id in ${1//,/ }; do
  until grep -q "done $id rc=0" /workspace/logs/scheduler.log; do
    grep -q "done $id rc=[^0]" /workspace/logs/scheduler.log && { echo "dependency $id failed"; exit 1; }
    sleep 20
  done
done
echo "$2" >> /workspace/jobs/queue.jsonl
echo "queued after $1 at $(date -u +%H:%M)"
