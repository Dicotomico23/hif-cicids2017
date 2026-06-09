"""Ensemble of three Hybrid Isolation Forests.

The three members differ only in their subsample size (psi). Their anomaly
scores are combined with a simple arithmetic mean before a single threshold
is applied. Averaging makes the detector more stable than any individual
configuration.
"""

import time

import numpy as np

from .config import ENSEMBLE_CONFIGS, PRECISION_WEIGHT, vprint
from .forest import HybridIsolationForest, fbeta_threshold


class HIFEnsemble:
    """Three HIF members with averaged scores and one shared threshold."""

    def __init__(self, configs=None):
        self.configs = configs if configs is not None else ENSEMBLE_CONFIGS
        self.members = [HybridIsolationForest(**cfg) for cfg in self.configs]
        self.threshold = 0.5

    def fit(self, X_normal, X_anomalies=None, y_anomalies=None, verbose=True):
        if verbose:
            vprint("  Training %d HIF members ..." % len(self.members))
        for i, hif in enumerate(self.members):
            t0 = time.time()
            hif.fit(X_normal, X_anomalies, y_anomalies)
            if verbose:
                vprint("    Member %d done in %.1fs" % (i + 1, time.time() - t0))
        return self

    def score_samples(self, X):
        """Average the anomaly scores across all ensemble members."""
        return np.mean([hif.score_samples(X) for hif in self.members], axis=0)

    def optimize_threshold(self, X_val, y_val, precision_weight=PRECISION_WEIGHT):
        # Fix each member's normalization on the validation set so the chosen
        # threshold transfers correctly to the test set.
        for hif in self.members:
            hif.calibrate(X_val)
        scores = self.score_samples(X_val)
        self.threshold = fbeta_threshold(scores, y_val, precision_weight)
        return self.threshold

    def predict(self, X):
        return (self.score_samples(X) > self.threshold).astype(int)
