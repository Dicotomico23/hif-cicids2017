"""Data loading and preprocessing for CICIDS2017.

The pipeline, in order:

  1. Download and concatenate every CICIDS2017 CSV file.
  2. Collapse the multiclass label into a binary target: 0 = BENIGN,
     1 = any attack (the "anomalous" class). This is intentional: the study
     evaluates benign-versus-anomalous detection, not per-attack classification.
  3. Replace infinities, drop rows with missing values.
  4. Split into train / validation / test (stratified) BEFORE any scaling or
     feature selection, so nothing is fitted on data it should not see.
  5. Rank features by mutual information on a training subsample and keep the
     top K.
  6. Standardize (scaler fitted on train only).
  7. Oversample the minority class in the training set with Borderline-SMOTE.

Steps 4 to 7 are deliberately ordered to prevent data leakage into the test set.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    KAGGLE_DATASET,
    LABEL_COL,
    MI_SAMPLE_SIZE,
    RANDOM_STATE,
    TEST_SIZE,
    TOP_K_FEATURES,
    VAL_SIZE,
)


def load_dataset(nrows=None, dataset=KAGGLE_DATASET):
    """Download CICIDS2017 via kagglehub and return the concatenated DataFrame.

    Args:
        nrows: if set, a random subsample of this many rows is returned. Useful
               for quick smoke runs; leave as None for the full study.
    """
    import os

    import kagglehub

    path = kagglehub.dataset_download(dataset)
    print("Dataset path: %s" % path)

    frames = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".csv"):
                fp = os.path.join(root, name)
                print("  Loading %s" % fp)
                frames.append(pd.read_csv(fp, low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    if nrows is not None and nrows < len(df):
        df = df.sample(n=nrows, random_state=RANDOM_STATE).reset_index(drop=True)
    print("Raw shape: %s" % str(df.shape))
    return df


def clean(df):
    """Binarize the label, drop infinities/NaNs, and return (df, numeric_cols)."""
    df[LABEL_COL] = df[LABEL_COL].apply(
        lambda x: 0 if str(x).strip() == "BENIGN" else 1
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if LABEL_COL in numeric_cols:
        numeric_cols.remove(LABEL_COL)

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    counts = df[LABEL_COL].value_counts()
    total = len(df)
    print("Shape after cleaning: %s" % str(df.shape))
    print("  BENIGN : %8d  (%.1f%%)" % (counts[0], counts[0] / total * 100))
    print("  ATTACK : %8d  (%.1f%%)" % (counts[1], counts[1] / total * 100))
    return df, numeric_cols


def split(df, numeric_cols):
    """Stratified train / val / test split (70 / 15 / 15 by default)."""
    X_all = df[numeric_cols].values
    y_all = df[LABEL_COL].values

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_all,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=VAL_SIZE / (1.0 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_tmp,
    )
    print("  Train : %8d rows" % X_train.shape[0])
    print("  Val   : %8d rows" % X_val.shape[0])
    print("  Test  : %8d rows" % X_test.shape[0])
    return X_train, X_val, X_test, y_train, y_val, y_test


def select_features(X_train, y_train, numeric_cols, k=TOP_K_FEATURES):
    """Rank features by mutual information on a subsample; return (idx, names)."""
    sample_size = min(MI_SAMPLE_SIZE, len(X_train))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_train), sample_size, replace=False)

    mi = mutual_info_classif(
        X_train[sample_idx], y_train[sample_idx],
        discrete_features=False, random_state=RANDOM_STATE,
    )
    ranking = (
        pd.DataFrame({"Feature": numeric_cols, "MI": mi})
        .sort_values("MI", ascending=False)
    )
    top = ranking.head(k)["Feature"].tolist()
    print("Top %d features by mutual information:" % k)
    for i, feat in enumerate(top, 1):
        print("  %2d. %s" % (i, feat.strip()))

    col_idx = [numeric_cols.index(feat) for feat in top]
    return col_idx, top


def scale(X_train, X_val, X_test):
    """Standardize all splits with a scaler fitted on the training set only."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def balance(X_train, y_train):
    """Oversample the minority (attack) class with Borderline-SMOTE."""
    print("  Before SMOTE: %s" % str(np.bincount(y_train)))
    smote = BorderlineSMOTE(random_state=RANDOM_STATE, n_jobs=-1)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print("  After  SMOTE: %s" % str(np.bincount(y_bal)))
    return X_bal, y_bal
