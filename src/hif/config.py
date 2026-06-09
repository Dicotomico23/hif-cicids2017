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

# HIF semi-supervised step: cap on how many labelled anomalies are routed into
# each tree to build the per-leaf anomaly centroids. Routing the full
# SMOTE-balanced anomaly set (hundreds of thousands of rows) through every tree
# is the dominant training cost and a serious memory hazard (each routed vector
# is stored at a leaf in every tree). A representative random subsample yields
# essentially the same centroids at a fraction of the time and memory. Set to
# None to route every anomaly (not recommended on large datasets).
ANOMALY_CENTROID_SAMPLE = 5000

# Console verbosity for the pipeline scripts:
#   0 = quiet   (results table only),
#   1 = default (announce every pipeline step; the historical behaviour),
#   2 = verbose (everything in 1 plus environment/config dump, per-feature
#       mutual-information scores, library warnings and Optuna trial logs).
# Set at runtime with run_comparison.py --verbose; do not edit here.
VERBOSITY = 1


def set_verbosity(level):
    """Set the global console verbosity (0, 1 or 2)."""
    global VERBOSITY
    VERBOSITY = int(level)


def vprint(msg="", level=1):
    """Print ``msg`` only when the current VERBOSITY is at least ``level``."""
    if VERBOSITY >= level:
        print(msg)
