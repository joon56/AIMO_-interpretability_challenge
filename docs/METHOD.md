# Method in detail

This document specifies the pipeline of [`src/aimo_robust`](../src/aimo_robust) and the training scripts in
[`experiments/`](../experiments). Numbers are the ones in `runs/` and reproduce with the commands in the README.

- [1. Notation and targets](#1-notation-and-targets)
- [2. Prompt-state probe (`z_len`)](#2-prompt-state-probe-z_len)
- [3. Progress probe (`len_prog`)](#3-progress-probe-len_prog)
- [4. Fusion and decision](#4-fusion-and-decision)
- [5. Test-time procedure and safety rails](#5-test-time-procedure-and-safety-rails)
- [6. Black-box comparison features](#6-black-box-comparison-features)
- [7. Artifact format](#artifact-format)
- [8. Chosen hyper-parameters per configuration](#8-chosen-hyper-parameters-per-configuration)
- [9. Pre-registered decision rules and what happened](#9-pre-registered-decision-rules-and-what-happened)

## 1. Notation and targets

A *configuration* $c$ is a model plus a reasoning effort (7 of them: four fixed-effort models and gpt-oss-120b at low /
medium / high). A problem is $x$. For every pool problem and configuration, $n$ completions $y_1..y_n$ are sampled with the
label-generation settings (T = 1, top-k 40, top-p 0.95, 32,768-token cap, organizers' solve prompt). With $\ell_i$ the number
of generated tokens (cap-truncated samples count at the cap):

$$m_c(x)=\frac1n\sum_i \log(1+\ell_i)\qquad\qquad p_c(x)=\frac1n\sum_i \mathbb 1[\text{finished}\wedge \text{answer}=\text{reference}]$$

The *relative difficulty* is $d_c(x)=m_c(x)-\operatorname{median}_{x'\in\text{pool}} m_c(x')$. It is the central quantity:
among the 79 official labelled rows with generated samples, $-d_c$ separates robust from non-robust with **AUC 0.842** (the
raw, uncentered $-m_c$ gets 0.676 and the solve rate $p_c$ 0.627; `runs/v1/train_probe.log`).

The pool median matters because the validation and test sets are balanced per model: "easy for this model" has to be measured
against that model's own typical length, not against other models.

## 2. Prompt-state probe (`z_len`)

*Input.* The chat-formatted prompt that the submission itself builds (`aimo_robust.prompting.build_prompt`): the organizers'
solve instruction as system message, the problem as user message, the generation prompt, `enable_thinking=True`
(`reasoning_effort=low|medium|high` for gpt-oss). One forward pass with `output_hidden_states=True`, no sampling.

*Features.* Layer index $L$ into `hidden_states` (0 = embeddings, depth = last), pooled either over the **last** prompt token
or as the **mean** over the tokens inside the problem span (character offsets).

*Probe.* Standardize with pool statistics (std + 1e-4) $\to$ PCA to 32 components $\to$ ridge regression ($\alpha=10$) onto
$m_c(x)$ (a second probe onto $\operatorname{logit}\frac{k+0.5}{n-k+0.5}$ of the solve rate exists in the artifacts but is not
used by v1/v2: adding it hurt, see the README).

*Selection.* For each configuration: six depths ($\tfrac14,\tfrac13,\tfrac12,\tfrac23,\tfrac34$ of the layers and the last
hidden state) $\times$ two poolings, scored by out-of-fold Spearman $\rho$ over the pool in grouped 5-fold CV (template groups).
The out-of-fold predictions are also used to center the feature: $z_{len}(x)=\hat m_c(x)-\operatorname{median}_{\text{pool}}\hat m_c$.

*Quality.* Spearman between $\hat m_c$ and $m_c$ on held-out pool problems: Qwen 0.69, Skywork 0.52, Olmo 0.53, DeepSeek 0.44,
gpt-oss low 0.39 / medium 0.46 / high 0.63 (figure 2 in the README).

## 3. Progress probe (`len_prog`)

*Idea.* While the K parallel samples decode, the hidden state of a sample that is still running at the token cap $B_c$
carries information about how much longer it will run. Reading that state costs nothing extra.

*Test time.* A forward hook on decoder layer $\ell_p$ stores the last-position hidden state at every decode step; after the
final step each sample $i$ has a state $s_i$. For samples that finished before the cap the observed $\log(1+\ell_i)$ is used.

$$\widetilde v(x)=\frac1K\sum_i\Big[\mathbb 1[\text{finished}_i]\,\log(1+\ell_i)+\mathbb 1[\text{running}_i]\,g_c(s_i)\Big],\qquad
\text{len\_mix}(x)=\widetilde v(x)-\operatorname{median}_{\text{pool}}\widetilde v$$

*Training* (`cloud/extract_progress.py`, `experiments/train_progress.py`). For every stored vLLM sample that ran past $B_c$,
rebuild prompt $+$ the first $B_c-1$ generated tokens, run one forward pass and keep the last-position state at the stored
layers; the target is the problem's $m_c(x)$ (sample-level state, problem-level label). $g_c$ is the same
standardize-PCA(32)-ridge(10) probe; the layer is chosen by the problem-level Spearman of the resulting `len_mix` in grouped
5-fold CV. Layers whose stored states overflow fp16 (the final layer of gpt-oss medium/high) are skipped.

*Quality.* Problem-level Spearman of `len_mix`: gpt-oss low 0.99 (60% of its samples finish within the cap, so the estimate is
largely measured), medium 0.74, high 0.70, Qwen 0.54, Skywork 0.48, DeepSeek 0.36, Olmo 0.26 (only 72 problems, 288 running
samples).

*Why it exists.* The test-time budget only allows a few thousand decoded tokens, while the reasoning traces are far longer.
Share of the stored vLLM samples that would have finished inside the deployed cap:

| config | cap | finished within cap |
|---|---:|---:|
| gpt-oss-120b low | 1536 | 60.2% |
| gpt-oss-120b medium | 1536 | 11.6% |
| gpt-oss-120b high | 1536 | 2.2% |
| Skywork-OR1-Math-7B | 4096 | 5.3% |
| Qwen3.5-4B | 3072 | 0.0% |
| Olmo-3-7B-Think | 2560 | 0.0% |
| DeepSeek-R1-0528-Qwen3-8B | 3072 | 0.0% |

For five of the seven configurations, at most 12% of the samples finish, so any black-box statistic (finish rate, answer
agreement, observed lengths) is almost empty -- but the hidden state of the unfinished samples is not.

## 4. Fusion and decision

*Fusion* (`experiments/fuse.py`). Per configuration, ordinary least squares on the label-free pool (out-of-fold values):

$$m_c(x)\approx a_0+a_1\,z^{raw}_{len}(x)+a_2\,\text{len\_mix}^{raw}(x),\qquad
\text{len\_fused}(x)=a_0+a_1 z^{raw}_{len}+a_2\text{len\_mix}^{raw}-\operatorname{median}_{\text{pool}}(\cdot)$$

where the superscript *raw* means "before subtracting the configuration's pool median". Problems without progress states
(Olmo's second half) fall back to the prompt probe alone. Coefficients are in section 8; held-out Spearman of the fused
estimate against $m_c$: Qwen 0.72, Skywork 0.56, Olmo 0.55, DeepSeek 0.50, gpt-oss low 0.98 / medium 0.74 / high 0.74.

*Decision.* One shared rule for all configurations, on the single feature `len_fused`:

| version | rule | how it was obtained |
|---|---|---|
| v1 | robust iff `len_fused` $\le -0.157$ (logit $=-0.336-2.142\cdot$`len_fused`) | class-balanced logistic regression on the 80 official labelled rows |
| v1p | robust iff $z_{len}\le-0.079$ | same, on $z_{len}$ only, no sampling |
| **v2** | robust iff `len_fused` $\le 0$ (logit $=-2.142\cdot$`len_fused`) | slope of v1, intercept removed: the threshold is each model's pool median |

The official rows are 74% robust, the validation/test populations are balanced per model (the dataset card says so). v1
inherited the intercept of the unbalanced rows and predicted "robust" for 3 of 21 validation cases; v2 removes the
intercept. The v2 change uses no validation label: it follows from the stated prevalence mismatch, and
`experiments/make_v2_artifacts.py` is the complete diff.

## 5. Test-time procedure and safety rails

`src/solution.py::are_robust` is called once per case in a fresh process (about 170 s budget including imports). For each
problem `predict_one` does:

1. load the artifact for `(model, effort)` (fallback: the `default` effort file), the round-trip-checked tokenizer and the model (bf16);
2. one forward pass $\to$ $z_{len}$ (and $z_{solve}$ for the fallback rule);
3. if the artifact has a progress probe: sample $K$ completions in one batched `generate` call with the label sampling settings,
   `max_new_tokens` = the cap the probe was trained at, a wall-clock `StoppingCriteria`, a forward hook on layer $\ell_p$;
   compute `len_mix`, `len_fused`;
4. logit $\to$ `bool`; one `[aimo] ...` line per case on stderr (features and timing only, never problem text).

| config | K | cap $B_c$ | sampling window | notes |
|---|---:|---:|---:|---|
| Qwen3.5-4B | 32 | 3072 | 100 s | torch fallback for linear attention: 26.6 ms/step at batch 32, scales with batch |
| Skywork-OR1-Math-7B | 64 | 4096 | 100 s | 23.5 ms/step at batch 64 |
| Olmo-3-7B-Think | 16 | 2560 | 100 s | 32 KV heads, attention-bound: 61 ms/step at batch 32 |
| DeepSeek-R1-0528-Qwen3-8B | 32 | 3072 | 100 s | 27.2 ms/step at batch 32 |
| gpt-oss-120b (all efforts) | 40 | 1536 | 95 s | 50.9 ms/step at batch 32, 78.9 at 64 |

The sampling window is additionally capped at `BUDGET_S - 25 s` (default `BUDGET_S = 140`, env `AIMO_BUDGET_S`) after process
start. Measured on the evaluation worker class: 88-104 s per case, 2,122 s for the 21 Main validation cases.

*Fail-soft.* Any exception inside sampling keeps the probe-only decision for that case; an out-of-memory error halves $K$ while
time remains; a missing artifact, a failed prompt pass or a non-finite logit returns `True` (the constant guess is never worse
than an invalid prediction). `are_robust` never raises.

## 6. Black-box comparison features

Implemented in `aimo_robust.sampler` and kept in the artifacts for the ablation (they read no hidden state):

* `finish` -- share of the K samples that finished within the cap;
* `agree` / `agree_all` -- modal answer share among finished answers over K (unfinished count as disagreement), with an optional
  forced-answer pass for unfinished samples;
* `len_beh` -- censored log-normal MAP estimate of the problem's mean log length from the observed lengths: finished samples are
  exact observations with known within-problem spread $\sigma_c$, unfinished ones are right-censored at their generated length,
  prior $\mathcal N(\mu_{0,c},\tau_c)$ from the pool (grid 4.00-12.99 in steps of 0.01).

On the 61 official rows that have behavior samples: `len_fused` AUC 0.811 / balanced accuracy 0.736, black-box `len_beh`
0.623 / 0.613, both together 0.812 / 0.744, with `agree_logit` added 0.764. Hidden-state signals subsume the black-box
finish-rate information (experiment A1 in the log).

## Artifact format

One `numpy` `.npz` per configuration (`<model with / -> _>__<effort>.npz`), arrays only, no pickle. Probe blocks use the suffix
`""` for the solve-rate probe and `"_len"` for the length probe; the progress probe uses the prefix `prog_`.

| key | shape | meaning |
|---|---|---|
| `layers`, `pool`, `layers_len`, `pool_len` | `[1]`, str | hidden-state layer index and pooling (`last` or `mean`) |
| `mu`, `sd`, `pca`, `w`, `b` (+ `_len`) | `[d]`, `[d]`, `[32, d]`, `[32]`, scalar | standardization, PCA components, ridge weights; PCA centering is folded into `b` |
| `center`, `center_len` | scalar | pool median of the out-of-fold probe output |
| `prog_layer`, `prog_cap`, `prog_mu`, `prog_sd`, `prog_pca`, `prog_w`, `prog_b`, `prog_center` | | progress probe and its token cap |
| `fuse_a`, `fuse_center` | `[3]`, scalar | $a_0,a_1,a_2$ and the pool median of the fused value |
| `comb_feats`, `comb_w`, `comb_b` | `[1]`, `[1]`, scalar | decision on `len_fused`: `logit = comb_b + comb_w * len_fused` |
| `cal_feats`, `cal_w`, `cal_b` | | fallback decision (v1: on `z_solve, z_len`; v1p: `z_len`; v2: `z_len`, weight -2, bias 0) |
| `beh_mu_center`, `beh_sigma`, `beh_tau` | scalar | pool median / within-problem spread / spread of mean log length (black-box estimator) |

The arithmetic is in `aimo_robust.predict.probe_z`: `z = pca @ ((h - mu) / sd) @ w + b`.

## 8. Chosen hyper-parameters per configuration

From `runs/v2/art` (layer index into `hidden_states`; 0 = embeddings).

| configuration | length probe (layer, pooling) | progress probe (layer, cap) | fusion $a_0, a_1, a_2$ |
|---|---|---|---|
| Qwen3.5-4B | 32, last | 24, 3072 | -4.634, 0.841, 0.612 |
| Skywork-OR1-Math-7B | 21, last | 21, 4096 | -3.883, 0.492, 0.924 |
| Olmo-3-7B-Think | 16, last | 32, 2560 | -5.379, 0.870, 0.683 |
| DeepSeek-R1-0528-Qwen3-8B | 27, last | 36, 3072 | -2.137, 0.535, 0.673 |
| gpt-oss-120b, high | 18, mean | 18, 1536 | -5.828, 0.461, 1.162 |
| gpt-oss-120b, low | 24, mean | 36, 1536 | -0.936, 0.038, 1.106 |
| gpt-oss-120b, medium | 24, mean | 18, 1536 | -1.853, 0.164, 1.060 |

The weight on the progress estimate is largest exactly where traces finish within the cap (gpt-oss low: 1.11 vs 0.04 for the
prompt probe) and the prompt probe carries most of the weight where nothing finishes (Qwen, Olmo, DeepSeek).

## 9. Pre-registered decision rules and what happened

Written down on 26 Oct 2026, 20:00 KST, before any model had been sampled (`docs/EXPERIMENT_LOG.md` has the dated evidence).

| rule | statement | outcome |
|---|---|---|
| D1 | oracle AUC of the empirical solve rate $\ge0.75$ $\Rightarrow$ reliability hypothesis holds; $<0.65$ $\Rightarrow$ pivot to perturbation checks | solve-rate AUC 0.63 (n = 79): the *rate* saturates at $n=4$. The pivot went to a continuous difficulty proxy (reasoning length, oracle AUC 0.84), not to perturbation checks |
| D2 | truncation at 32k $>25\%$ for a model $\Rightarrow$ resample truncated problems at 64k | triggered for Qwen (60.6%) and DeepSeek (69.4%); not done for lack of GPU budget; lengths of truncated samples count at the cap (lower bound) and the black-box spread is estimated from finished samples |
| D3 | keep a probe only if held-out Spearman $\ge0.5$ | met by Qwen, Skywork, Olmo, gpt-oss high; DeepSeek (0.44) and gpt-oss low / medium (0.39 / 0.46) are below, which is why the progress probe was added and the two estimates fused |
| D4 | keep sampling only if it adds $\ge0.02$ OOF AUC | fusion adds 0.044 AUC (0.752 $\to$ 0.796): kept |
| D5 | the validation set is a sanity check; never tune thresholds to it | v1 scored 0.571; the diagnosis (prevalence mismatch) is label-free and led to v2 |
| D6 | nominate the best combined candidate plus a decorrelated hedge (probe-only) | -- |
