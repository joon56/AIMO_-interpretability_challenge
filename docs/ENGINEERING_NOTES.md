# Engineering notes

Measurements, failure modes and infrastructure lessons collected while building the solution (26-27 Oct 2026). The dated
raw entries are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (ids `I*`, `B*`, `E3`).

- [1. Evaluation environment](#1-evaluation-environment)
- [2. The five checkpoints](#2-the-five-checkpoints)
- [3. Decoding throughput with plain Hugging Face `generate`](#3-decoding-throughput-with-plain-hugging-face-generate)
- [4. Failure modes and fixes](#4-failure-modes-and-fixes)
- [5. GPU data generation: compute, cost and scheduling](#5-gpu-data-generation-compute-cost-and-scheduling)
- [6. Safety rails inside the submission](#6-safety-rails-inside-the-submission)

## 1. Evaluation environment

| item | value |
|---|---|
| runtime image | `ghcr.io/aimo-interp/aimo-codabench-runtime:torch2.12.1-cuda13.2-cudnn9-transformers5.13.0-sklearn1.8.0-kernels0.15.2` |
| libraries | torch 2.12.1 (CUDA 13.2), transformers 5.13.0, accelerate 1.13.0, scikit-learn 1.8.0, joblib 1.5.3, pandas 3.0.3; **no vLLM** |
| hardware | one RTX PRO 6000 Blackwell, 96 GB, shared (about 30 GB free with gpt-oss-120b loaded) |
| network / weights | offline; checkpoints in `/app/data/huggingface`, `local_files_only=True` |
| process model | one fresh process per case (no state survives), about 170 s per case on the test run, 4,200 s per validation run |
| per-case fixed cost | import 2.6 s; load 2.3 s (small models) / 6-7 s (gpt-oss-120b); prompt forward 0.2-0.4 s (gpt-oss 0.6-3.5 s) |
| validation sets | Main 21 cases (gpt-oss low 5 / medium 3 / high 3, Skywork 3, DeepSeek 3, Qwen 2, Olmo 2), Small 14 cases; balanced per model |
| constant baselines | always-`True` scores 0.476 on Main (10 of 21 cases robust) |

Consequence: a probe-only decision needs about 10-25 s per case, leaving roughly 110-130 s for budgeted sampling.

## 2. The five checkpoints

(`runs/model_specs.jsonl`, produced by `cloud/model_specs.py`)

| model | architecture | layers | hidden | attention | context | weights |
|---|---|---:|---:|---|---:|---:|
| Qwen/Qwen3.5-4B | hybrid, `Qwen3_5ForConditionalGeneration` | 32 | 2560 | 24 linear-attention + 8 full; 16 heads / 4 KV | 262k | 9.3 GB bf16 |
| Skywork/Skywork-OR1-Math-7B | `Qwen2ForCausalLM` | 28 | 3584 | GQA, 28 heads / 4 KV | 131k | 15.2 GB bf16 |
| allenai/Olmo-3-7B-Think | `Olmo3ForCausalLM` | 32 | 4096 | 24 sliding-window (4096) + 8 full; **32 KV heads** | 65k | 14.6 GB bf16 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | `Qwen3ForCausalLM` | 36 | 4096 | GQA, 32 heads / 8 KV | 131k | 16.4 GB bf16 |
| openai/gpt-oss-120b | `GptOssForCausalLM`, MoE 128 experts, top-4 | 36 | 2880 | 18 sliding-window (128) + 18 full | 131k | 65.2 GB MXFP4 |

## 3. Decoding throughput with plain Hugging Face `generate`

Sampled decoding (T = 1, top-k 40, top-p 0.95), 2,048 new tokens, runtime-mirror environment on an RTX PRO 6000
(`cloud/bench_hf.py`, `runs/pod/runs/bench`). These numbers fixed the sample counts and token caps in `SAMPLING`.

| model | batch 32: ms/step, tokens/s, peak GB | batch 64: ms/step, tokens/s, peak GB |
|---|---|---|
| Qwen3.5-4B | 26.6, 1,203, 11.2 | 52.8, 1,213, 14.6 |
| Skywork-OR1-Math-7B | 18.1, 1,768, 18.2 | 23.5, 2,726, 22.1 |
| Olmo-3-7B-Think | 61.4, 521, 48.5 | 118.4, 541, 83.3 |
| DeepSeek-R1-0528-Qwen3-8B | 27.2, 1,178, 25.0 | 40.4, 1,585, 34.6 |
| gpt-oss-120b | 50.9, 628, 64.6 | 78.9, 812, 68.4 |

* Qwen3.5-4B scales *linearly* with batch size because transformers falls back to a torch implementation of the linear
  attention (the fast kernels are not part of the image): more samples cost more time.
* Olmo-3 is attention-bound (32 KV heads, no GQA): 16 samples is all that fits the budget; batch 64 needs 83 GB.
* Cold start in the pod measurement (disk cold): import 2.2-4.6 s, load 3-9 s.

## 4. Failure modes and fixes

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | DeepSeek-R1-0528-Qwen3-8B prompts decode without any spaces; the first vLLM run produced garbage that looked valid | `AutoTokenizer` (a LlamaTokenizer) strips **all** spaces on encode/decode in transformers 5.13 *and* 5.19 | `aimo_robust.modeling.load_tokenizer` tries `AutoTokenizer` and `PreTrainedTokenizerFast`, keeps the first that round-trips a probe string; the vLLM generator tokenizes with it and passes token ids (`skip_tokenizer_init`) |
| 2 | gpt-oss-120b OOM when loaded with transformers | without the `kernels` package (range `[0.15.2, 0.16)`) the MXFP4 weights are silently dequantized to bf16, which cannot fit; `kernels` 0.17.2 does so too | pin `kernels>=0.15.2,<0.16` (the evaluation image has 0.15.2) |
| 3 | gpt-oss-120b in vLLM 0.32 on one 96 GB GPU: OOM during MXFP4 -> Marlin repack | sm_120 has no FlashInfer MXFP4 MoE path | `--tensor-parallel-size 2` |
| 4 | vLLM tp=2 crashed selecting an attention backend | it auto-picked FLASHINFER, which is not installed | `attention_backend=TRITON_ATTN`, plus `ninja` on `PATH` for the JIT |
| 5 | Qwen3.5-4B vLLM failed at start | the multimodal processor needs a tokenizer; `skip_tokenizer_init` breaks it | skip tokenizer init only when `AutoTokenizer` fails the round trip (it does not for Qwen) |
| 6 | Qwen 60.6% and DeepSeek 69.4% of samples hit the 32k cap | these models think far longer than the cap | lengths count at the cap (lower bound); the within-problem spread of the black-box estimator uses finished samples only; the 64k resample was out of budget |
| 7 | Olmo-3 and DeepSeek generation crawled in vLLM | Olmo's 32 KV heads leave only 146k tokens of KV cache; long-context DeepSeek decoding is KV-read-bound (about 0.5-1k tokens/s/GPU), not preemption | sharding, smaller n, a 144-problem subset for the two slow models |
| 8 | non-finite values in the stored gpt-oss medium/high progress states; `NaN` `len_prog` for Olmo | final-layer hidden states overflow fp16 when stored; Olmo's progress states cover only its first 72 problems | `train_progress.py` skips layers with non-finite states and handles missing values explicitly (AUC, medians, fusion fall back to the prompt probe) |
| 9 | Hugging Face downloads hung at 0 B/s on the pod | the xet transfer backend | `HF_HUB_DISABLE_XET=1`, parallel plain-HTTP downloads (about 140 MB/s per stream) |
| 10 | SSH uploads to the pod froze above about 1.4 KB; post-quantum key exchange reset | an MTU black hole on the upstream path | `KexAlgorithms=curve25519-sha256` and uploads as base64 chunks of at most 1,000 bytes, one SSH session each, reassembled and SHA-256-verified (`cloud/pod.sh`) |
| 11 | vLLM workers died at spawn | missing `if __name__ == "__main__"` guard (spawn start method) | guard in `cloud/gen_solve.py` |
| 12 | `pkill -f pattern` killed its own remote shell; `nohup` kept SSH open | the pattern appears in the command line | bracket trick `[g]en_solve.py`, `setsid nohup ... < /dev/null &` |
| 13 | `pip install` refused (PEP 668) on the pod image | system Python is externally managed | `uv venv` per environment (`cloud/setup_pod.sh`) |
| 14 | Windows: Hugging Face snapshot symlink errors; TLS errors behind an intercepting proxy | no symlink privilege; custom root certificate | `local_dir` downloads; `truststore.inject_into_ssl()` |
| 15 | a 12-problem DeepSeek chunk skewed the first probe calibration | it came from the garbled-prompt run (#1) | moved aside (`gen_partial/`) until the full data existed |

## 5. GPU data generation: compute, cost and scheduling

* **Hardware.** RunPod pods with 4 x RTX PRO 6000 Blackwell Server (96 GB each, driver 595.91, CUDA 13.2), 400 GB volume,
  256 cores, 1.5 TB RAM: the same GPU class as the evaluation worker, so timings transfer. Two long-lived pods plus one short
  pod for the last Olmo shard. About $8-9 per pod-hour; the work consumed roughly $80 of prepaid credit (a $60 deposit and a
  $20 top-up) before the balance hit zero and the pods stopped on their own at about 03:00 KST on 27 Oct.
* **Generation throughput (vLLM 0.32.0).** gpt-oss-120b with tp=2: low 10.3k tokens/s, medium 6.5k tokens/s; Skywork 2.0k
  tokens/s per GPU (6.9 M tokens in the first chunk, 2% truncated, median 11.8k tokens); DeepSeek and Olmo were KV-bound.
* **Budget triage (I10).** With $19 left the full DeepSeek shards (one third done, unsaved) and a second Olmo shard were killed;
  DeepSeek was re-run on the 144-problem subset with n = 2 and behavior collection for Qwen/DeepSeek/Olmo was dropped; the
  remaining jobs (progress states, gpt-oss behavior) were queued behind each other so that they finished before the balance did.
* **Scheduler (`cloud/scheduler.py`).** A tiny GPU job queue: jobs are JSON lines `{"id", "ngpu", "cmd", "short"}` appended to
  `/workspace/jobs/queue.jsonl`; a GPU is free when `nvidia-smi` shows no compute process on it and no scheduler job holds it;
  FIFO with 1-GPU backfill while a multi-GPU head job waits; state in `state.json`, one log per job. `after_job.sh` chains a job
  after its dependencies, `watch.sh` pulls results every 20 minutes and exits on a new milestone, `collect_final.sh` pulls from
  every pod until the final outputs exist so nothing is lost when the balance ends.
* **Reproducing the data.** `cloud/run_all.sh` (single pod) or `cloud/bootstrap_pod.sh cloud/jobs_*.jsonl` (scheduler) rebuild
  `runs/pod/runs/{gen,hidden,progress,behavior,bench}`; the exact job lists that were run are `cloud/jobs_*.jsonl`. The Korean
  RunPod walkthrough is [cloud_gpu_manual.ko.md](cloud_gpu_manual.ko.md).

## 6. Safety rails inside the submission

* `are_robust` never raises: any exception inside one case yields `True` (an invalid prediction is scored as wrong, a constant
  guess is never worse), and the traceback goes to stderr.
* Sampling is optional per artifact and fail-soft: if it fails, the probe-only decision for that case is used.
* OOM during sampling halves K and retries while time remains; a wall-clock `StoppingCriteria` bounds generation; the process
  budget is `AIMO_BUDGET_S` (default 140 s) from import time.
* One `[aimo] <model> <effort> <features> logit=... t=...s` line per case on stderr -- features, decision and timing only, never
  problem text -- so that evaluation logs can be diagnosed without exposing validation data (`experiments/parse_eval_log.py`).
* A bundle check (`experiments/check_bundle.py`) imports `solution`, verifies the `are_robust` signature, the fallback for an
  unknown model and which configs carry a full / probe-only artifact. The official `run_local.py` ingestion harness was used
  once, on CPU with Qwen3.5-0.8B and a fake artifact, to validate packaging (coverage 1.0, unknown-model fallback works).
