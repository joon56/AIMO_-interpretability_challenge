<!-- Working notes written before the final numbers existed; the README holds the final results. -->
# Draft — AIMO Interpretability Challenge technical report (working notes, numbers TBD)

## Thesis
Robustness labels (max family drop <= 0.10 vs >= 0.25, base answer consistently correct) are dominated by the
model's per-problem reliability p. Evidence: organizers' sample (other models, surface families): base acc 1.0 ->
92% robust; 0.7-0.9 -> 25-57%. A binomial view: with ~10 samples/family and ~6 families, p = 0.95 vs 0.85 flips the
expected label. So the task reduces to estimating p for a *specific* model on an *unseen* problem, cheaply.

## Method (two signals, one calibrated decision)
1. White-box solve-rate probe. One forward pass over the chat-formatted problem; hidden state at layer L
   (last prompt token or problem-span mean), standardise -> PCA(32) -> ridge to logit solve rate.
   Training targets need no robustness labels: self-generated solve rates (n=4..6 samples, organizers' sampling
   T=1, top-k 40, top-p 0.95, 32k cap) on 322 recent olympiad problems (MathArena 2023-26 + train problems).
2. Budgeted self-consistency (test time). K parallel samples under a wall-clock budget, then thinking-budget
   forcing for unfinished samples. Features: finish rate, modal agreement (finished-only and incl. forced).
3. Decision: class-balanced logistic regression on [probe z, agreement logits, finish] fitted on official train
   rows (80 labelled), leave-template-out CV (symbolic siblings grouped).

## Ablation plan (white-box award)
prior-only / probe-only / behavior-only / combined; OOF AUC + balanced acc on train (template CV), val (queue),
own held-out. White-box component = probe z.

## Engineering findings worth reporting
- DeepSeek-R1-0528-Qwen3-8B: transformers 5.13/5.19 AutoTokenizer strips spaces -> garbled prompts silently.
  Fix: round-trip-checked tokenizer selection.
- Eval worker throughput (fresh process): import 2.6 s, load 2.3-6 s, HF sampled decode 0.7-1.8k tok/s at batch 32.
- gpt-oss-120b in vLLM 0.32 on RTX PRO 6000 (sm_120): Marlin MXFP4 repack OOM at tp=1; tp=2 needs TRITON_ATTN.
- Olmo-3-7B-Think: 32 KV heads -> tiny KV capacity (146k tokens in vLLM) -> slowest to sample.

## Data/code release
Problem pool (MathArena + train), per-sample answers/lengths, probe artifacts (npz), code.

## Evidence log (2026-10-26, numbers to refresh at the end)
- Oracle (full-budget vLLM samples, n=4-6, 32k cap): per-row mean log reasoning length predicts the official label:
  gpt-oss-120b medium AUC 1.00 (n=15; unlabeled band rows fall exactly between classes), Skywork 0.93-1.00,
  gpt-oss low 0.80. Centered per model by its pool median ("relative length"): pooled AUC 0.90-0.91 (n=29-39),
  rule rel<0 -> robust: 86-87% acc. Solve rate alone: 0.68-0.71 (saturates at 4/4).
- Interpretation: the label is a noisy readout of per-model problem difficulty; reasoning length is a sharper,
  continuous difficulty measure than an n=4 solve rate. Unlabeled (0.10<drop<0.25) rows sit at the boundary.
- White-box probe (last prompt token, one layer, PCA32+ridge) predicts the model's own mean log length:
  Spearman 0.70 (Qwen3.5-4B), 0.52 (Skywork) on held-out pool problems; train robust AUC 0.73-0.77 (n=24).
- Behavioral estimate: censored log-normal MAP of mean log length from K budgeted samples; uninformative when
  traces are far longer than the budget (Qwen3.5-4B: 0/32 finished at 3k tokens on all val cases).
- Eval runtime (queue): 87-103 s/case with sampling; 9-25 s probe-only.
- v1 (10/27): decision on one feature len_fused = per-model OLS (fit on the label-free pool) of the prompt-state length
  probe and the progress probe (hidden state at the token cap during budgeted sampling), centered per model.
  Train-row repeated grouped CV (n=80): AUC .796, balanced acc .734 vs prompt probe only .752/.677;
  black-box censored length from the same samples .623 and adds nothing (.812 with both on n=61).
