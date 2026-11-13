# Submission plan and ledger

Validation on Codabench is a sanity check, not a measurement: Main has 21 cases and Small 14, so one score has a standard
error of about 0.11 / 0.13 and differences below roughly 0.15 are noise.

## Rules we held ourselves to

1. **Never pick a method by validation score.** Selection uses grouped cross-validation on the official labelled rows plus the
   label-free problem pool. Validation is a sanity check, and a last-resort tie-break only for gaps of 0.15 or more.
2. Every submission tests a stated hypothesis and is logged below. No "flip a few predictions" variants: the organizers
   disqualify label-profiling sequences.
3. Pre-flight every bundle before spending quota (organizers' anonymous test queue, which lost its workers on 27 Oct, or a pod):
   coverage 1.0, zero invalid predictions, at most 140 s per case.
4. Keep at least 10 submissions in reserve until the end, because stuck Codabench runs still count.
5. Limits at planning time: 10 per day, 300 in total, 4,200 s per run. Phase end 1 Nov 2026, 11:55 UTC (20:55 KST).

## Plan (26 Oct 2026) -- about 50 submissions, 10 per day

| phase | when (KST) | submissions | purpose |
|---|---|---:|---|
| A sanity | 27 Oct morning | 2 | v1 probe-only, Main and Small |
| B ablation | 27-28 Oct | 6 | per track: probe-only / behavior-only / combined (white-box award evidence) |
| C improve | 28-30 Oct | at most 12 | at most 4 per day, only variants with a CV gain |
| D final | 31 Oct | 6 | two finals per track and a rerun of each (non-determinism check) |
| E reserve | 1 Nov until 20:55 | at least 10 | Codabench failures only, no new methods on the last day |

Nomination (two per participant): the best Main bundle and the best Small bundle, both carrying the white-box probes; the report
shows the ablation with and without them.

## Ledger of the runs that are attributed to a bundle

| id | date (KST) | where | track | bundle (sha256[:16]) | hypothesis / change | val acc | runtime | notes |
|---|---|---|---|---|---|---|---|---|
| q1 | 26 Oct 18:34 | organizers' queue | Main | constant `True` | runtime probe: fresh-process cost on the evaluation worker | 0.476 | 381 s total | 10 of 21 cases robust |
| q2 | 26 Oct 22:27 | organizers' queue | Small | v0rt (38dbee4f) | runtime test of the probe + sampler path, placeholder weights | 0.571 (meaningless) | 800 s total; Qwen 87-93 s, Skywork 103 s per case | Qwen finished 0/32 samples on 5/5 cases at 32 x 3072 tokens |
| b1 | 26 Oct 20:21 | Codabench | Main / Small | baseline-always-true (34ecec56 / 02067b0b) | official constant baselines | 0.48 / 0.50 | | ids 1009005 / 1009009 |
| C1 | 27 Oct 07:57 | Codabench | Main | **v1-main** (b5511621) | full white-box `len_fused`, threshold fitted on the official rows | **0.571** (12/21) | 2,122 s (97 s per case) | id 1011078; 0 sampling failures; predicted robust for only 3 of 21 cases |
| C3 | 27 Oct 08:49 | Codabench | Main | **v1p-main** (514f15ce) | ablation: prompt-state length probe only, no sampling | **0.667** (14/21) | deterministic; estimated 10-25 s per case | id 1011176; threshold -0.079 predicts robust for 9 of 21 on the C1 features |
| -- | 27 Oct 08:56 | (built) | Main / Small | v2-main (337db860) / v2-small (61e197e6) | threshold at the per-model pool median | not recorded here | | v1-small (d25d1936) and v1p-small (f251768f) were built but superseded |

Uploads made after 27 Oct 09:30 KST (v2 and later variants) are not tracked in this ledger.

The submission bundles are in [`../submissions/`](../submissions) with their hashes in `MANIFEST.md`.

## Why v1 failed (0.571) -- a label-free diagnosis

The per-case stderr lines of the v1-main run (features and timing only) show where the 18 "non-robust" predictions came from.
Aggregated per configuration (21 Main validation cases; `len_fused` is centered on each model's pool median):

| configuration | cases | predicted robust | mean `len_fused` |
|---|---:|---:|---:|
| Qwen3.5-4B | 2 | 0 | -0.023 |
| Skywork-OR1-Math-7B | 3 | 1 | -0.720 |
| Olmo-3-7B-Think | 2 | 2 | -0.289 |
| DeepSeek-R1-0528-Qwen3-8B | 3 | 0 | +0.022 |
| gpt-oss-120b, low | 5 | 0 | +0.109 |
| gpt-oss-120b, medium | 3 | 0 | +0.002 |
| gpt-oss-120b, high | 3 | 0 | +0.335 |
| **all** | **21** | **3** | median +0.004 |

1. The centering works: the median of `len_fused` over the validation cases is +0.004, i.e. the pool median is the right
   reference for a balanced set, and 15 of the 21 values lie within +-0.15 of it.
2. The decision threshold does not: v1 fitted the logistic intercept on the 80 official rows, which are skewed easy (74% robust,
   median `len_fused` -0.20, see figure 4 in the README), so it demanded `len_fused <= -0.157`, which is left of 18 of the 21
   validation cases.
3. No bug and no feature failure: sampling worked on every case (0 failures), timing was 88-104 s per case.
4. v2 keeps the slope and drops the intercept (robust iff `len_fused <= 0`), which on the same features predicts robust for
   10 of 21 cases. The change uses no validation labels and follows from the prevalence mismatch that the official dataset card
   warns about. Whether it helps on validation or test is not recorded in this repository.
