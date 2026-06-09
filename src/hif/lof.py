"""Local Outlier Factor baseline.

LOF is an unsupervised, density-based detector. It is transductive: it scores
the points it is fitted on rather than generalising to unseen data. We fit it
on the training and test rows together and then read off the test-set portion.
Because LOF does not learn an inductive decision boundary, its ROC AUC on
held-out data is close to chance; this is a property of the method, not a bug,
and is reported as such.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def run_lof(X_train, X_test, y_test, n_neighbors=20):
    """Fit LOF transductively and return (predictions, scores) for the test set.

    Args:
        X_train: scaled training features (before SMOTE).
        X_test:  scaled test features.
        y_test:  test labels, used only to set the contamination fraction.

    Returns:
        (lof_pred, lof_scores) aligned with X_test. Higher score = more anomalous.
    """
    contamination = float(np.mean(y_test))
    contamination = max(0.001, min(contamination, 0.5))  # clip to a valid range

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1
    )

    X_lof = np.vstack([X_train, X_test])
    lof_raw = lof.fit_predict(X_lof)
    pred_all = np.where(lof_raw == -1, 1, 0)
    scores_all = -lof.negative_outlier_factor_  # higher = more anomalous

    n_train = len(X_train)
    return pred_all[n_train:], scores_all[n_train:]
