# Data

**This repository ships no problem text.** The problems come from two public sources whose licenses differ from this
repository's, so they are downloaded on demand:

```bash
python experiments/download_data.py --build     # downloads everything below, then builds data/pool/problems*.jsonl
```

The rebuilt `problems.jsonl` / `problems_olmo.jsonl` and all 17 source parquet files were verified to be byte-identical
to the files the experiments ran on (checked 13 Nov 2026).

## Sources

| What | Hub id | License | Used for |
|---|---|---|---|
| Official labelled rows (94 rows, 36 problems, 7 model configs) | `aimo-interp/train-main-v2` | `other` (see dataset card) | the only robustness labels: 80 labelled rows (59 robust / 21 non-robust), 14 unlabelled in the 0.10-0.25 band |
| Competition problems with integer answers | `MathArena/{aime_2023_I,aime_2023_II,aime_2024_I,aime_2024_II,aime_2025,aime_2026,hmmt_feb_2023,hmmt_feb_2024,hmmt_feb_2025,hmmt_feb_2026,hmmt_nov_2025,brumo_2025,cmimc_2025,smt_2025,apex_2025,apex-shortlist}` | CC BY-NC-SA 4.0 | the label-free problem pool the probes are trained on |
| Other organizer samples (optional, `--all`) | `aimo-interp/{val-sample,aimo-interp-challenge-sample-full,augmented-sample-math*}` | see cards | exploratory analysis only (E0 in the experiment log) |

The competition rules allow external data: the organizers retired the rule against extra labeled data (clarification of
19 Sep 2026). Nothing from the hidden validation/test sets is used or stored anywhere in this repository.

## The problem pool (`data/pool/problems.jsonl`, 322 problems)

`experiments/build_pool.py` keeps MathArena problems whose reference answer is an integer, removes duplicates by
normalized text and appends the official train problems that are not already in the pool.

| source | problems | | source | problems |
|---|---:|---|---|---:|
| aime_2025 | 30 | | hmmt_feb_2023 | 16 |
| aime_2026 | 30 | | hmmt_feb_2026 | 16 |
| apex-shortlist | 29 | | aime_2023_I | 15 |
| smt_2025 | 28 | | aime_2023_II | 15 |
| official train (not in MathArena) | 22 | | aime_2024_I | 15 |
| hmmt_nov_2025 | 21 | | aime_2024_II | 15 |
| hmmt_feb_2024 | 17 | | hmmt_feb_2025 | 14 |
| brumo_2025 | 16 | | apex_2025 | 7 |
| cmimc_2025 | 16 | | | |

* 300 problems have a reference answer; the 22 official-train problems have none, so their "truth" is the majority
  vote of finished answers across all generated samples (gpt-oss-high votes count twice).
* `problems_olmo.jsonl` (144 problems) = every official train problem + {aime_2025, aime_2026, hmmt_feb_2025,
  hmmt_feb_2026, brumo_2025, cmimc_2025}. The two slowest models to sample (Olmo-3, DeepSeek-R1) were run on this subset.
* `pid` = first 12 hex characters of `sha1(normalized problem text)`, normalization = collapse whitespace, strip, lowercase
  (`experiments/build_pool.py`). Every released result file refers to problems by `pid` only.
* `pool_manifest.csv` lists `pid, source, in_official_train, reference_answer_known` for all 322 problems (no text,
  no answers).

## Template groups (cross-validation)

Symbolic siblings of one template share a *group*: two problems are linked when `difflib.SequenceMatcher` on the first 200
normalized characters exceeds 0.75, groups are the connected components. All cross-validation in this repository splits by
group (the official rows contain 27 groups over 33 labelled problems), never by row.

## Official labelled rows

`train-main-v2` columns: `model_id, reasoning_effort, problem, is_robust, max_drop`. `is_robust` is true iff
`max_drop <= 0.10` and false iff `max_drop >= 0.25`; the 14 rows in between are null. The dataset card warns that the split
is 74% robust among labelled rows, whereas validation is close to balanced, so a threshold fitted on it need not transfer
-- which is exactly what happened to v1 (see the README, "Why v1 failed").

| configuration | robust | non-robust | unlabelled |
|---|---:|---:|---:|
| Qwen/Qwen3.5-4B | 8 | 4 | 2 |
| Skywork/Skywork-OR1-Math-7B | 9 | 3 | 3 |
| allenai/Olmo-3-7B-Think | 7 | 2 | 1 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 7 | 3 | 0 |
| openai/gpt-oss-120b, high | 13 | 2 | 3 |
| openai/gpt-oss-120b, low | 5 | 2 | 1 |
| openai/gpt-oss-120b, medium | 10 | 5 | 4 |

## Generated samples

For every (config, pool problem) the notes below describe `runs/pod/runs/gen/*.jsonl`, produced by `cloud/gen_solve.py`
with vLLM 0.32.0 using the label-generation sampling settings (temperature 1.0, top-k 40, top-p 0.95) and a 32,768-token cap
(the real labels use 100k). The organizers' solve prompt is the system message. See `runs/README.md` for the file formats.

| config | problems | samples / problem | truncated at 32k | median length (tokens) | generated tokens |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 322 | 4 | 60.6% | 32,768 (finished: 23,028) | 37.1 M |
| Skywork-OR1-Math-7B | 322 | 4 | 1.2% | 12,272 | 17.0 M |
| Olmo-3-7B-Think | 108 | 3-4 | 7.8% | 17,499 | 7.4 M |
| DeepSeek-R1-0528-Qwen3-8B | 144 | 2 | 69.4% | 32,768 (finished: 23,061) | 8.5 M |
| gpt-oss-120b, low | 322 | 6 | 0.0% | 1,319 | 2.8 M |
| gpt-oss-120b, medium | 322 | 6 | 0.0% | 4,194 | 10.2 M |
| gpt-oss-120b, high | 322 | 6 | 20.7% | 13,620 (finished: 9,971) | 31.1 M |
| **total** | | | | | **114.2 M tokens, 9,056 samples** |

Heavy truncation (Qwen, DeepSeek) means that "mean log length" is a *lower bound* of the true value for those problems;
truncated samples count at the cap, and the within-problem spread used by the black-box estimator is computed from finished
samples only.
