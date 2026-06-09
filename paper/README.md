# Paper

This directory holds the LaTeX source of the paper. The Overleaf project is
the working copy; commit a snapshot here so the repository is self-contained.

## Files to add here

- `main.tex` (the Springer sn-jnl manuscript)
- `references.bib` (the bibliography actually referenced by `main.tex`)
- `sn-jnl.cls` and the `.bst` style file
- `figures/` (the four figures, copied from `../results/figures/`)

Build with `latexmk -pdf main.tex`.

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
