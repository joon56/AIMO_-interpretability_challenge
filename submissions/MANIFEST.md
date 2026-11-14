# Submission bundles

SHA-256 (first 16 hex), size in bytes and a note for every bundle in this folder.
See [docs/SUBMISSION_LEDGER.md](../docs/SUBMISSION_LEDGER.md) for what each one tested.

| built (KST) | file | sha256[:16] | bytes | note |
|---|---|---|---|---|
| 2026-10-26 | baseline-always-true-main.zip | 34ecec568b544e08 | 430 | official constant baseline |
| 2026-10-26 | baseline-always-true-small.zip | 02067b0bedc51457 | 526 | official constant baseline |
| 2026-10-27 04:27 | v1-main.zip | b5511621d4be6c9c | 8706408 | v1 FULL: len_fused (per-config OLS of prompt length probe + progress probe; CV AUC .796 bal .734 n=80), sampling+hook |
| 2026-10-27 04:27 | v1-small.zip | d25d19362c0b3d14 | 5430321 | v1 FULL: len_fused (per-config OLS of prompt length probe + progress probe; CV AUC .796 bal .734 n=80), sampling+hook |
| 2026-10-27 04:27 | v1p-main.zip | 514f15ce6c0fd466 | 5807401 | v1 PROBE-ONLY: z_len (CV AUC .752 bal .677 n=80), no sampling |
| 2026-10-27 04:27 | v1p-small.zip | f251768fdec4bf96 | 3623335 | v1 PROBE-ONLY: z_len (CV AUC .752 bal .677 n=80), no sampling |
| 2026-10-27 08:56 | v2-main.zip | 337db860c10d0941 | 8706063 | v2: v1 features, threshold at per-model pool median (len_fused<0); fallback z_len<0 |
| 2026-10-27 08:56 | v2-small.zip | 61e197e6d2d79bdb | 5430117 | v2: v1 features, threshold at per-model pool median (len_fused<0); fallback z_len<0 |
| 2026-10-28 01:22 | v3-main.zip | 296ed6b0ba5bfc3c | 8706989 | **v3, nominated for the Main track**: v2 features (len_fused), per-config capability-aware threshold theta_c = 0.7-quantile of the pool out-of-fold len_fused over reliably-solved problems (label-free); fallback z_len <= theta_z |
| 2026-10-28 01:22 | v3-small.zip | 4a8ed026dd5f7acd | 5430651 | **v3, nominated for the Small track**: same rule, four small models |
| 2026-10-28 01:22 | v3b-main.zip | 7adda6a7f1bc36e0 | 5808914 | ABLATION black-box: same K / token-cap sampling, decision len_beh <= 0 (censored MLE from finish flags and lengths), no hidden-state readout; failure -> True |
| 2026-10-28 01:22 | v3b-small.zip | b53dff7aa080f583 | 3624192 | ABLATION black-box (Small) |
| 2026-10-28 01:22 | v3p-main.zip | 0f19f535440d3869 | 5808165 | ABLATION probe-only: prompt-state z_len <= theta_z (capability threshold), no sampling |
| 2026-10-28 01:22 | v3p-small.zip | b1d29dadb118686f | 3623772 | ABLATION probe-only (Small) |

Not published: the contract-test, runtime-test (`v0rt`) and `v1pre` bundles. They were superseded within hours and carry
placeholder or partial artifacts. All bundles here contain the same code as [`../src`](../src); only the `.npz` artifacts
(thresholds, probe weights) differ. `experiments/mock_predict_test.py` checks the decision path of a bundle without model weights.
