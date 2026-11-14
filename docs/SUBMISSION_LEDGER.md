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
5. Limits at planning time: 10 per day, 300 in total, 4,200 s per run. Phase end 1 Nov 2026, 11:55 UTC (20:55 KST); the
   organizers later extended it to 2 Nov 2026, 12:00 UTC (21:00 KST).

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

What happened to the plan: the GPU credit ran out on 27 Oct (see "Why v1 failed"), so phase C became a single label-free
change, the capability-aware threshold of v3 (chosen from training-row evidence, not from validation); phase D was v3 for both
tracks plus its two ablations; the reserve was spent on re-running the probe-only ablation for the report. Fourteen Codabench
uploads are listed below (and two runtime tests on the organizers' temporary queue), far fewer than the 50 planned.

## Ledger of the runs that are attributed to a bundle

| id | date (KST) | where | track | bundle (sha256[:16]) | hypothesis / change | val acc | runtime | notes |
|---|---|---|---|---|---|---|---|---|
| q1 | 26 Oct 18:34 | organizers' queue | Main | constant `True` | runtime probe: fresh-process cost on the evaluation worker | 0.476 | 381 s total | 10 of 21 cases robust |
| q2 | 26 Oct 22:27 | organizers' queue | Small | v0rt (38dbee4f) | runtime test of the probe + sampler path, placeholder weights | 0.571 (meaningless) | 800 s total; Qwen 87-93 s, Skywork 103 s per case | Qwen finished 0/32 samples on 5/5 cases at 32 x 3072 tokens |
| b1 | 26 Oct 20:21 | Codabench | Main / Small | baseline-always-true (34ecec56 / 02067b0b) | official constant baselines | 0.48 / 0.50 | | ids 1009005 / 1009009 |
| C1 | 27 Oct 07:57 | Codabench | Main | **v1-main** (b5511621) | full white-box `len_fused`, threshold fitted on the official rows | **0.571** (12/21) | 2,122 s (97 s per case) | id 1011078; 0 sampling failures; predicted robust for only 3 of 21 cases |
| C3 | 27 Oct 08:49 | Codabench | Main | **v1p-main** (514f15ce) | ablation: prompt-state length probe only, no sampling | **0.667** (14/21) | deterministic; estimated 10-25 s per case | id 1011176; threshold -0.079 predicts robust for 9 of 21 on the C1 features |
| -- | 27 Oct 08:56 | (built) | Main / Small | v2-main (337db860) / v2-small (61e197e6) | threshold at the per-model pool median | see C12 / C13 | | v1-small (d25d1936) and v1p-small (f251768f) were built but superseded and never uploaded |
| C12 | 28 Oct 02:01 | Codabench | Main | v2-main (337db860) | threshold at the pool median | **0.619** (13/21) | | id 1013674; v1 0.571 / v1p 0.667 / v2 0.619, all within the +-0.1 noise |
| C13 | 29 Oct 02:36 | Codabench | Small | v2-small (61e197e6) | first Small run of the full pipeline | **0.643** (9/14) | | id 1016746 |
| C7 | 29 Oct 02:44 | Codabench | Main | **v3-main** (296ed6b0) | v2 features, capability-aware per-configuration threshold ([METHOD.md](METHOD.md#the-v3-rule-capability-aware-threshold)) | **0.619** (13/21) | | id 1016760; same score as v2-main; **nominated for Main** |
| C8 | 29 Oct 02:44 | Codabench | Small | **v3-small** (4a8ed026) | same rule, four small models | **0.643** (9/14) | | id 1016763; uploaded with both tasks ticked, so its Main-task child record shows 0.00 / 21 invalid (a Small-only bundle has no Main answers): not a failure |
| C9 | 29 Oct 02:44 | Codabench | Main | v3b-main (7adda6a7) | white-box ablation: same samples, black-box length estimate `len_beh <= 0`, no hidden states | 0.619 (13/21) | | id 1016761; ties the white-box method on Main validation |
| C10 | 29 Oct 02:45 | Codabench | Small | v3b-small (b53dff7a) | same | 0.571 (8/14) | | id 1016767 |
| C14 | 30 Oct 01:30 | Codabench | Small | v3-small (4a8ed026), Small task only | clean nomination ID and rerun check | **0.643** (9/14) | | id 1019315; same score as 1016763; **nominated for Small** |
| C15 | 1 Nov 23:41 | Codabench | Main | v3p-main (0f19f535) | probe-only ablation with the capability threshold, no sampling | 0.667 (14/21) | | id 1025821; same score as v1p-main under a different threshold |
| C11 | 1 Nov 23:42 | Codabench | Small | v3p-small (b1d29dad) | probe-only ablation | failed, no scores | | id 1025824; server-side "Failure updating submission data" |
| C11 | 2 Nov 00:05 | Codabench | Small | v3p-small (b1d29dad) | probe-only ablation, re-upload | 0.571 (8/14) | | id 1025857 |

The ledger numbers (q, b, C) are upload numbers; the `C` numbers of the [experiment log](EXPERIMENT_LOG.md) are a separate series.
The submission bundles are in [`../submissions/`](../submissions) with their hashes in `MANIFEST.md`.

## Validation summary (21 Main + 14 Small cases)

| method | Main | Small | both |
|---|---:|---:|---:|
| constant "robust" (official baseline) | 0.476 (10/21) | 0.500 (7/14) | 17/35 |
| v1: full white-box, threshold fitted on the official rows | 0.571 (12/21) | not run | -- |
| v1p: prompt probe only, threshold fitted on the official rows | 0.667 (14/21) | not run | -- |
| v2: threshold at the pool median | 0.619 (13/21) | 0.643 (9/14) | 22/35 |
| **v3 (final): capability-aware threshold** | **0.619 (13/21)** | **0.643 (9/14)** | **22/35** |
| v3b: black-box ablation (same samples, no hidden states) | 0.619 (13/21) | 0.571 (8/14) | 21/35 |
| v3p: probe-only ablation (no sampling) | 0.667 (14/21) | 0.571 (8/14) | 22/35 |

One score on 21 (14) cases has a standard error of about 0.11 (0.13), so the validation board cannot rank any of these methods:
every gap is one or two cases. For scale, the maximum of five equally good methods on 21 cases is inflated from 0.64 to about
0.76 (winner's curse). The black-box ablation scores like the white-box method because, at this budget, it degenerates to a
per-model constant for every configuration whose traces never finish, which scores about one half on a set balanced per model.

## Final nomination and the rules that apply to it

* **Nominated** (decided 30 Oct, submitted with the report on 4 Nov): **1016760** (v3-main, Main track only) and **1019315**
  (v3-small, Small track only), White-Box Award ticked. Only the first two IDs of the OpenReview form are evaluated, and each
  ID names its own tracks.
* **160 s per call.** The organizers' clarification of 30 Oct (getting-started issue #6): the final evaluation calls
  `are_robust` once per case and enforces 160 s per call; an overrun counts as a wrong answer. The final score is the better of
  the two nominations. v3 needs 88-104 s per case on the validation server, about 55 s of margin.
* The phase ended on 2 Nov 21:00 KST; nothing was uploaded after that.

## Why v3 and not v1p, v2 or v3p

Selection used a mechanism and the 80 labelled official rows, not the validation board. v3 fits nothing to those rows; they only
score the rules and inform one number, rho = 0.7, which sits slightly below the natural robust rate (0.74) among consistently
solved training cases:

| comparison | official rows (80) | validation |
|---|---|---|
| v3 vs v1p (S1 in the log) | 63 vs 52 correct; 13 vs 2 discordant cases, sign test p = .007 (v1p's threshold was fitted on these rows, so the rows are in-sample for v1p) | Main 13 vs 14 of 21: any split of a one-case gap gives p = 1.0 |
| v3 vs v3p (S2) | 63 vs 56 correct (balanced accuracy .749 vs .689); 9 vs 2 discordant, p = .065 (cases share problems, so this p is optimistic); AUC .804 vs .768, bootstrap interval of the gap [-.07, .12] | Main 13 vs 14, Small 9 vs 8 |

The weak evidence favours v3, none favours v1p or v3p, and the rule (threshold at the 0.7-quantile of the label-free pool among
reliably solved problems) was fixed before these scores were computed. v3 therefore stayed the nomination for both tracks.

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
   warns about. On validation v2 scored 0.619 on Main (13/21, id 1013674) and 0.643 on Small (9/14, id 1016746), one case
   above v1 on Main. v3 then moved the boundary from the pool median to the reliably solved problems (see above and
   [METHOD.md](METHOD.md#the-v3-rule-capability-aware-threshold)).
