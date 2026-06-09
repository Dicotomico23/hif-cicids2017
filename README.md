# Hybrid Isolation Forest on CICIDS2017

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dicotomico23/hif-cicids2017/blob/main/notebooks/colab_run.ipynb)

Reproducible comparison of a Hybrid Isolation Forest (HIF) against supervised
baselines (Random Forest, Multi-Layer Perceptron, Support Vector Machine) and
an unsupervised baseline (Local Outlier Factor) for network anomaly detection
on the CICIDS2017 dataset.

The task is binary: benign traffic versus anomalous (any attack) traffic. HIF
is tuned for a high-precision operating point; the baselines provide a balanced
reference. This is the companion code for the accompanying paper.

## Run on Colab (recommended)

The full pipeline is heavy for a laptop. Click the badge above, or open
`notebooks/colab_run.ipynb` in Google Colab, to clone the repo, install the
dependencies, download the dataset and run the study on Colab hardware. Use a
High-RAM runtime; a GPU is not needed.

## Quick start (local)

```
git clone https://github.com/Dicotomico23/hif-cicids2017.git
cd hif-cicids2017
pip install -r requirements.txt
pip install -e .

python tests/test_smoke.py          # fast check, no dataset needed
python data/download.py             # fetch the archived dataset
python reproduce/run_comparison.py  # full study
```

`data/download.py` retrieves the dataset and writes the results table and
figures to `results/`. Tune the supervised baselines with Optuna (validation F1; test never used):

```
python reproduce/run_comparison.py --optimize --n_trials 20
```

Run instantly on the committed sample (no download needed):

```
python reproduce/run_comparison.py --data data/sample/cicids2017_sample.csv
```

Run the full dataset on a smaller, class-balanced slice for faster turnaround
(stratified, preserves class proportions):

```
python reproduce/run_comparison.py --fraction 0.25
```

A quick partial run on a subsample:

```
python reproduce/run_comparison.py --nrows 50000
```

## What the pipeline does

1. Load and concatenate the CICIDS2017 CSV files.
2. Binarize labels: 0 = benign, 1 = any attack.
3. Stratified 70 / 15 / 15 train / validation / test split.
4. Select the top 22 features by mutual information.
5. Standardize (scaler fitted on train only).
6. Borderline-SMOTE on the training set.
7. Train HIF (ensemble of three), the supervised baselines and LOF; tune the
   HIF threshold on validation; evaluate once on test.

The split happens before scaling, feature selection and oversampling so that
none of them sees test data. Full details are in `docs/methodology.md`.

## Validation against the original HIF

`reproduce/compare_with_original.py` checks our HIF against Marteau's original
implementation (https://github.com/pfmarteau/HIF) on the same preprocessed
data. Marteau's code is GPL-2.0+, so it is fetched at run time into `.external/`
(git-ignored) and never vendored into this MIT repository.

```
python reproduce/compare_with_original.py --nrows 50000
```

## Repository layout

```
src/hif/             HIF implementation, baselines, preprocessing, metrics
reproduce/           run_comparison.py (main study) and compare_with_original.py
scripts/             dataset packaging helper
tests/               synthetic smoke test
data/                download instructions (no data is committed)
results/             generated table and figures
docs/                methodology and notes
paper/               LaTeX source of the paper (see below)
```

## Outputs

- `results/table5_results.csv` and `.json`: metrics for every model.
- `results/figures/`: radar chart, HIF confusion matrix, balanced-accuracy bar
  chart and precision bar chart.

## Reproducibility

A single global seed (42) controls the splits, feature selection,
Borderline-SMOTE and every model. Dependency versions are pinned in
`requirements.txt`. Results are from one run with this seed.

## Paper

The LaTeX source lives in `paper/`. The Overleaf project is the working copy;
this repository holds the committed snapshot. See `paper/README.md` once the
source is added.

## License

MIT. See `LICENSE`. If you use this work, see `CITATION.cff`.
