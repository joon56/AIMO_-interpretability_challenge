# Technical report

**`AIMO_report.pdf`** is the technical report exactly as it was submitted to the challenge on OpenReview on 4 Nov 2026:
*How Long Will It Think? Hidden-State Length Probes Predict the Robustness of Reasoning Models.* It is the double-blind
version (anonymous author line), 7 pages: a 5-page main text, then limitations, the AI Use Statement, references and an
appendix. SHA-256 (first 16 hex): `801b3dd54e8ca31f`.

| path | what |
|---|---|
| `AIMO_report.pdf` | the submitted PDF |
| `paper/main.tex`, `paper/refs.bib` | LaTeX source |
| `paper/neurips_2026.sty` | the official NeurIPS 2026 style file (`dblblindworkshop` option), kept under its own terms |
| `paper/figures/fig_pipeline.pdf`, `fig_main.pdf` | the two figures of the report |
| `make_figures.py` | regenerates both figures from `runs/` |
| `report_numbers.py` | recomputes every number quoted in the report and writes `numbers.json` |
| `numbers.json` | the numbers of Table 1 and Table 2 (per-configuration probe quality, thresholds, ablation, bootstrap intervals) |

## Rebuild

```bash
python experiments/download_data.py --build                  # the problem pool (no problem text is stored in git)
python report/report_numbers.py                              # recomputes report/numbers.json from runs/ (no fitting on official labels)
python report/make_figures.py --out report/paper/figures     # fig_main and fig_pipeline (PDF + PNG preview)
cd report/paper && tectonic main.tex                         # or pdflatex + bibtex; the PDF is main.pdf
```

Checked on 14 Nov 2026 with the files in this repository: `report_numbers.py` reproduces all 145 values of `numbers.json`,
and `make_figures.py` reproduces the two PNG figures byte for byte (the PDFs differ only in their creation date).

## What the report claims, in one paragraph

The official labels mostly measure how reliably a model solves a problem, and the model's reasoning length relative to its own
typical length measures that more sharply than the solve rate (AUC 0.84 vs 0.63 on the official training rows, from full
traces). The submission predicts that length from hidden states: a prompt probe plus a progress probe on the state of a
budgeted rollout, both trained on a label-free pool of 322 problems, and a capability-aware threshold placed among the
problems a model reliably solves. On the 80 official labelled rows, used for scoring only, it reaches AUC 0.80 and balanced
accuracy 0.75; on the validation board it scores 0.619 (Main, 13/21) and 0.643 (Small, 9/14). The hidden-state readout beats a
black-box length estimate from the same samples on the training rows (AUC 0.82 vs 0.60), but the 35 validation cases cannot
separate the white-box method from its black-box or probe-only ablations. The README of this repository has the full
write-up, including the early versions that the report only summarizes.
