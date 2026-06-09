# Paper

This directory holds the LaTeX source of the paper. The Overleaf project is
the working copy; commit a snapshot here so the repository is self-contained.

## Workflow: originals vs. edited versions

Put the untouched files exported from Overleaf in `paper/original/`:

- `paper/original/main.tex`
- `paper/original/references.bib`

That folder is git-ignored (staging only). The cleaned, corrected versions
that we edit and align with the code are committed at the top of `paper/`:

- `paper/main.tex`        (edited)
- `paper/references.bib`  (edited)
- `sn-jnl.cls` and the `.bst` style file
- `figures/` (the four figures, copied from `../results/figures/`)

So: drop the originals in `paper/original/`, and we produce `paper/main.tex`
and `paper/references.bib` from them. Build with `latexmk -pdf main.tex`.

## Pending alignment with the code

When the clean source is added, the following must be reconciled so the paper
matches the code in this repository (the code is the source of truth):

- Scope is benign versus all attacks, not DoS only. Sections describing a
  DoS-only experiment must be updated.
- Feature count is 22 everywhere (one section currently says 20).
- Feature selection method is mutual information (`mutual_info_classif`).
- Threshold objective is an F-beta score with beta = sqrt((1 - w) / w),
  w = 0.9, optimised by a continuous bounded search. The equation and the
  algorithm listing must both state this.
- No automated hyperparameter search is used; baseline hyperparameters are
  fixed. Any Optuna claim must be removed or corrected.
- HIF ensemble: three members, t = 100, psi in {256, 512, 128}, alpha1 =
  alpha2 = 0.5, scores combined by arithmetic mean.
- HIF is semi-supervised (it uses labelled anomalies for the leaf centroids).
- SVM baseline is LinearSVC with probability calibration, not a kernel SVC.
- LOF is unsupervised; its near-chance ROC AUC is explained as a property of
  its transductive nature.
- Split is stratified 70 / 15 / 15; global seed 42; single run.
- Fix the broken cross-reference to the results comparison subsection.
- Fill the missing F1 and ROC AUC cells in the baseline row of the results
  table (or mark them as not reported).

## Validation against the original HIF (reproducibility subsection)

Our HIF was validated against Marteau's original implementation
(https://github.com/pfmarteau/HIF, GPL-2.0+). This belongs in a short
reproducibility/validation subsection, NOT as a model in the main results
table. Add a small table comparing the per-signal and combined ROC AUC of our
HIF versus the original on the same preprocessed split (produced by
reproduce/compare_with_original.py), and one sentence stating that the two
agree within tree-construction randomness. Cite Marteau's HIF (arXiv:1705.03800)
for the algorithm and note that the original code is GPL and only used for
validation, not redistributed.
