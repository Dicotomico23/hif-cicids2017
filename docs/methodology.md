# Methodology

This document explains, in full, what the pipeline does and why. It is the
reference for anyone trying to understand or extend the experiments.

## Goal

Compare a Hybrid Isolation Forest (HIF) against supervised baselines and an
unsupervised baseline for detecting attacks in network traffic, using the
CICIDS2017 benchmark. The task is binary: distinguish benign traffic from
anomalous (attack) traffic.

## Scope: benign versus all attacks

Every attack category in CICIDS2017 (DoS/DDoS, Brute Force, PortScan, Botnet,
Web Attack, Infiltration, Heartbleed) is collapsed into a single positive
class labelled "anomalous". The negative class is benign traffic. The study
does not perform per-attack classification. This matches the original
experiment design: a detector that flags "normal versus not normal".

## Pipeline

The stages run in this exact order. The ordering matters: the split happens
before any fitting so that the scaler, the feature selector and SMOTE never
see test data.

1. Load. Download CICIDS2017 via kagglehub and concatenate every CSV.
2. Clean. Map labels to 0 (BENIGN) and 1 (any attack). Replace infinities
   with NaN and drop rows with missing values.
3. Split. Stratified 70 / 15 / 15 into train / validation / test.
4. Feature selection. Rank features by mutual information, estimated on a
   10,000-row training subsample, and keep the top 22.
5. Scale. Standardize with a `StandardScaler` fitted on the training set only.
6. Balance. Apply Borderline-SMOTE to the training set to oversample the
   minority (attack) class.
7. Train and evaluate. Train all models on the balanced training set, tune the
   HIF threshold on the validation set, and evaluate once on the test set.

## The Hybrid Isolation Forest

Each HIF combines three signals into one anomaly score:

- s, the isolation depth (standard Isolation Forest path length);
- sc, the Euclidean distance to the centroid of the normal samples in the leaf;
- sa, a ratio that brings a sample closer to the score of known anomalies when
  it lands near an anomaly centroid.

The final score is

    shif = a2 * (a1 * s + (1 - a1) * sc) + (1 - a2) * sa

with a1 = a2 = 0.5. A HIF is trained on benign samples to model normal
behaviour, then labelled anomalies are routed into the leaves so each leaf can
store an anomaly centroid. Using labelled anomalies in this way is what makes
the model semi-supervised rather than a plain Isolation Forest.

The implementation follows the Hybrid Isolation Forest formulation by
Pierre-Francois Marteau (https://github.com/pfmarteau/HIF). It was validated
against that reference: `reproduce/compare_with_original.py` runs both
implementations on the same preprocessed split and compares the ROC AUC of the
three signals and of the combined score. After alignment the two agree closely
(combined ROC AUC differs only by tree-construction randomness). Note that
Marteau's code is GPL-2.0+; it is fetched at run time and never vendored into
this MIT repository.

## The ensemble

Three HIF members are trained with the same number of trees (t = 100) and
different subsample sizes (psi in {256, 512, 128}). Their scores are combined
with a simple arithmetic mean. A single decision threshold is then applied to
the averaged score.

## Threshold selection

The operating point favours precision over recall. The threshold is chosen on
the validation set by maximising an F-beta score, with

    beta = sqrt((1 - w) / w),   w = 0.9

The search is a grid over the range of validation scores. A grid is used
rather than a continuous optimiser because the F-beta objective is piecewise
constant in the threshold, which traps continuous solvers in flat regions. A
high w pushes the threshold towards very high precision (few false alarms) at
the cost of recall, which is the behaviour reported for HIF.

The score normalization bounds are fitted on the validation set and reused at
inference time, so the threshold chosen on validation transfers correctly to
the test set.

## Baselines

- Random Forest (`RandomForestClassifier`, 100 trees).
- Neural Network, a Multi-Layer Perceptron (`MLPClassifier`, hidden layers
  128 and 64, early stopping).
- Support Vector Machine, `LinearSVC` wrapped in `CalibratedClassifierCV` to
  obtain probability estimates for the ROC AUC. A linear SVM is used because a
  kernel SVM does not scale to this many rows.
- Local Outlier Factor (`LocalOutlierFactor`), an unsupervised, density-based
  detector. LOF is transductive: it scores the points it is fitted on and does
  not generalise inductively, so its ROC AUC on held-out data is close to
  chance. This is a property of the method and is reported as such.

The supervised baselines (RF, NN, SVM) are tuned with Optuna when the pipeline
is run with `--optimize`: each trial fits a candidate on the balanced training
set and is scored by validation F1 (the test set is never used for tuning). The
search spaces are defined in `src/hif/optimize.py`. Without `--optimize`, the
baselines use the fixed defaults in `src/hif/baselines.py`. LOF and the HIF
ensemble are not tuned (fixed configurations).

## Metrics

Accuracy, balanced accuracy, F1, precision, recall and ROC AUC, plus training
and inference time. The positive class is "anomalous". Figures produced:

- `fig_radar_metrics.png`, a radar chart of the headline metrics;
- `fig_hif_confusion_matrix.png`, the HIF confusion matrix;
- `fig_bar_balanced_acc.png`, balanced accuracy per model;
- `fig_bar_precision.png`, precision per model.

## Reproducibility

A single global seed (42) is applied to numpy, the splits, the feature
selector, Borderline-SMOTE and every scikit-learn estimator. Results are from
a single run with this seed. Pinned dependency versions are in
`requirements.txt`.

## Relationship to the original code

The original experiments lived in a Jupyter notebook that contained several
bugs and could not run end to end. This repository is the corrected,
reproducible version of that work: the same HIF formulation, the same
benign-versus-anomalous framing and the same top-22 mutual-information feature
selection, repackaged into a tested, runnable pipeline with a proper
train/validation/test split, Borderline-SMOTE and the three-member ensemble.
