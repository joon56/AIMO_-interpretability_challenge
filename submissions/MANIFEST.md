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
