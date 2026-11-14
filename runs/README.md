# Results and intermediate artifacts

Everything here was produced by the scripts in `experiments/` and `cloud/`. Problems are referenced by `pid` only (see
[../data/README.md](../data/README.md)); the official-train problem text that the original CSV/JSONL files contained was
replaced by its `pid` before publishing, so the released `report/*.csv` files are for inspection (the scripts regenerate
them locally, with text, from the data you download).

```
runs/
├── model_specs.jsonl              architecture / dtype / context / revision of the five checkpoints (cloud/model_specs.py)
├── v1/
│   ├── report/                    probe outputs and decision tables (CSV, see below)
│   ├── art_fused/                 v1 artifacts: probes + progress probes + fusion + logistic decision   (7 x .npz)
│   ├── art_probeonly/             v1p artifacts: prompt-state length probe only, no sampling           (7 x .npz)
│   └── {train_probe,train_progress,fuse}.log   console output of the three training steps
├── v2/art/                        v2 artifacts = v1 art_fused with the threshold moved to the pool median (7 x .npz)
├── v3/art/                        v3 artifacts (nominated) = v2 with the capability-aware thresholds       (7 x .npz)
├── v3b/art/                       v3 black-box ablation: decision on len_beh from the same samples         (7 x .npz)
├── v3p/art/                       v3 probe-only ablation: prompt-state z_len at its capability threshold   (7 x .npz)
└── pod/runs/
    ├── gen/                       vLLM samples: answers, lengths, finish reasons (one jsonl per config and shard)
    ├── behavior/                  the submission's own sampler run on official train cases (5 configs, 74 cases)
    └── bench/                     HF-generate cold-start and throughput on the evaluation GPU class
```

## Not in git (regenerate with `cloud/`)

| path | size | produced by | needed for |
|---|---:|---|---|
| `runs/pod/runs/hidden/*.npz` | 977 MB | `cloud/extract_hidden.py` (all layers, last-token and problem-span-mean states of every pool prompt) | `experiments/train_probe.py` |
| `runs/pod/runs/progress/*.npz` | 130 MB | `cloud/extract_progress.py` (hidden states of partial traces at the deployed token cap) | `experiments/train_progress.py` |
| `runs/pod/runs/progress_alt/*.npz` | 68 MB | same, gpt-oss at cap 1792 (an alternative that was not deployed) | -- |

The trained artifacts above are enough to build and run all submission bundles; the large files are only needed to
retrain the probes. If you want them, regenerate them with the GPU scripts (see the README, "Reproduce").

## File formats

**`gen/<config>[.partN].jsonl`** -- one line per problem:
`{"pid", "model", "effort", "prompt_len", "samples": [{"ans": "220" | null, "len": 11773, "finish": "stop" | "length"}, ...]}`.
`len` = generated tokens, `finish` = `length` when the sample hit the 32k cap. `.partN` files are shards (`--start/--end`).

**`behavior/<config>.jsonl`** -- one line per official train case, the exact output of `aimo_robust.sampler` at the deployed
token cap: `{"model", "effort", "pid", "k", "finish", "agree", "agree_all", "modal", "lens", "fin", "ans", "forced",
"seconds", "k_cfg", "max_new"}`. DeepSeek and Olmo were not collected (budget).

**`bench/<config>.jsonl`** -- `{"stage": "cold", "import_s", "load_s", "mem_gb", "tokenizer"}` and
`{"stage": "gen", "batch", "new_tokens", "seconds", "tok_s", "step_ms", "peak_gb"}` lines.

**`v1/report/*.csv`**

| file | rows | columns |
|---|---|---|
| `pool_probe_oof.csv` | pool problem x config x target | `model_id, reasoning_effort, pid, target (solve\|len), oof` (raw out-of-fold probe output) `, y` (measured target) |
| `pool_prog_oof.csv` | pool problem x config | `..., pid, len_mix_oof` (out-of-fold progress estimate) `, mlog` (measured mean log length) |
| `train_rows_z.csv` | official labelled row | `model_id, reasoning_effort, pid, is_robust, max_drop, y, grp, emp_rate, emp_len, emp_len_rel, z_solve, z_len` |
| `train_rows_prog.csv` | official labelled row | `model_id, reasoning_effort, pid, len_prog` |
| `train_rows_fused.csv` | official labelled row | all of the above plus `len_fused` |

`y` = 1 for robust; `grp` = template group; `emp_*` = measured solve rate / mean log length / length relative to the
config's pool median (the "oracle" columns, they use the generated samples of the same problem); `z_*`, `len_prog`,
`len_fused` = out-of-fold hidden-state features, centered on each config's pool median.

**`*.npz` artifacts** -- one file per config, plain arrays (no pickle). Schema in [../docs/METHOD.md](../docs/METHOD.md#artifact-format).
Load with `numpy.load(path, allow_pickle=False)`.

## Provenance of the artifact sets

| set | built by | decision |
|---|---|---|
| `v1/art_probeonly` (v1p) | `experiments/train_probe.py` | logistic on `z_len` (balanced weights): robust iff `z_len <= -0.079` |
| `v1/art_fused` (v1) | `train_probe.py` -> `train_progress.py` -> `fuse.py` | logistic on `len_fused`: robust iff `len_fused <= -0.157` |
| `v2/art` (v2) | `experiments/make_v2_artifacts.py` on `v1/art_fused` | robust iff `len_fused <= 0` |
| `v3/art` (v3, nominated) | `experiments/capability_threshold.py --art runs/v2/art --rho 0.7` | robust iff `len_fused <= theta_c`; probes and fusion weights are v2's; keys `cap_rho`, `cap_theta`, `cap_theta_z` added ([METHOD.md](../docs/METHOD.md#the-v3-rule-capability-aware-threshold)) |
| `v3b/art` (black-box ablation) | `experiments/make_ablation.py --art runs/v3/art --mode blackbox` | progress probe and fusion removed; robust iff the censored log-normal length estimate `len_beh <= 0`; a sampling failure returns `True` |
| `v3p/art` (probe-only ablation) | `experiments/make_ablation.py --art runs/v3/art --mode probeonly` | no sampling; robust iff `z_len <= theta_z` |

`capability_threshold.py` reproduces `v3/art` from `v2/art` bit for bit (checked 14 Nov 2026 with the files published here:
266 of 266 arrays identical), and `make_ablation.py` reproduces `v3b/art` and `v3p/art` from `v3/art` (196 and 175 arrays identical). The report CSVs under `v1/report/` carry `pid` instead of problem text, which is all the scripts
need besides the rebuilt problem pool (`experiments/download_data.py --build`).
