"""Tiny GPU job scheduler for the pod. Jobs are appended to /workspace/jobs/queue.jsonl as
{"id": str, "ngpu": int, "cmd": str, "short": bool (optional)}. A GPU is free when nvidia-smi shows no compute process on it and no
scheduler job holds it. FIFO, but a 1-GPU job may backfill while the head job waits for more GPUs.
State: /workspace/jobs/state.json ; logs: /workspace/logs/job_<id>.log
Run: nohup python3 cloud/scheduler.py > /workspace/logs/scheduler.log 2>&1 &
"""

import json
import os
import subprocess
import time
from pathlib import Path

JOBS = Path("/workspace/jobs")
QUEUE, STATE = JOBS / "queue.jsonl", JOBS / "state.json"
JOBS.mkdir(parents=True, exist_ok=True)
QUEUE.touch()
ENV = {**os.environ, "HF_HOME": "/workspace/hf", "TOKENIZERS_PARALLELISM": "false"}


def busy_gpus() -> tuple[set[int], int]:
    uuids = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.split()
    table = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip().splitlines()
    idx = {u.strip(): int(i) for i, u in (line.split(",") for line in table)}
    return {idx[u] for u in uuids if u in idx}, len(table)


state = json.loads(STATE.read_text()) if STATE.exists() else {"done": {}, "running": {}}
procs = {}
while True:
    for jid, p in list(procs.items()):
        if p.poll() is not None:
            state["done"][jid] = {"rc": p.returncode, "end": time.strftime("%H:%M:%S"), **state["running"].pop(jid)}
            del procs[jid]
            print(f"{time.strftime('%H:%M:%S')} done {jid} rc={p.returncode}", flush=True)
    jobs = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]
    pending = [j for j in jobs if j["id"] not in state["done"] and j["id"] not in state["running"]]
    used, n = busy_gpus()
    held = {g for r in state["running"].values() for g in r["gpus"]}
    free = [g for g in range(n) if g not in used and g not in held]
    head_waiting = False
    for j in pending:
        if j["ngpu"] > len(free):
            head_waiting = True  # FIFO head blocked; only short jobs may backfill behind it
            continue
        if head_waiting and not j.get("short"):
            continue
        gpus, free = free[: j["ngpu"]], free[j["ngpu"]:]
        log = open(f"/workspace/logs/job_{j['id']}.log", "a")
        env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
        procs[j["id"]] = subprocess.Popen(["bash", "-c", j["cmd"]], cwd="/workspace/claude", env=env,
                                          stdout=log, stderr=subprocess.STDOUT)
        state["running"][j["id"]] = {"gpus": gpus, "start": time.strftime("%H:%M:%S")}
        print(f"{time.strftime('%H:%M:%S')} start {j['id']} on {gpus}", flush=True)
    STATE.write_text(json.dumps(state, indent=1))
    time.sleep(20)
