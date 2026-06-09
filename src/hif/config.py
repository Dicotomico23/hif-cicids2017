"""Central configuration for the HIF comparison pipeline.

All tunable constants live here so a reader can see, in one place, every
choice that affects the reported results. Changing a value here changes the
behaviour of the whole pipeline.
"""

# Reproducibility: a single global seed is applied to numpy, the train/val/test
# splits, the feature selector, Borderline-SMOTE and every scikit-learn model.
RANDOM_STATE = 42

# Kaggle mirror of CICIDS2017 (the same MachineLearning CSV export used in the
# original experiments). Downloaded automatically via kagglehub.
KAGGLE_DATASET = "chethuhn/network-intrusion-dataset"

# Name of the label column as it appears in the CICIDS2017 CSV files. The
# leading space is part of the original column name and is intentional.
LABEL_COL = " Label"

# Feature selection: number of features kept, ranked by mutual information.
TOP_K_FEATURES = 22
# Mutual information is estimated on a random subsample of the training set to
# keep the estimator fast on millions of rows.
MI_SAMPLE_SIZE = 10_000

# Train / validation / test split (stratified). The validation set is used
# only for HIF threshold selection; the test set is evaluated once.
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # as a fraction of the full dataset

# HIF ensemble: three Hybrid Isolation Forests with different subsample sizes.
# t   = number of trees, psi = subsample size per tree.
ENSEMBLE_CONFIGS = [
    {"t": 100, "psi": 256},
    {"t": 100, "psi": 512},
    {"t": 100, "psi": 128},
]

# Threshold selection objective. The operating point favours precision over
# recall by maximising an F-beta score with beta derived from this weight.
PRECISION_WEIGHT = 0.9
