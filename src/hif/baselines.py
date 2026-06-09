"""Supervised baseline classifiers used for comparison.

Random Forest, a Multi-Layer Perceptron and a linear SVM. The SVM uses
LinearSVC (much faster than a kernel SVC on hundreds of thousands of rows)
wrapped in CalibratedClassifierCV so it can produce probability estimates
for the ROC AUC. All hyperparameters are fixed and documented; no automated
search is performed.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE


def build_baselines():
    """Return a dict {name: estimator} of the supervised baselines."""
    rf = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=200,
        random_state=RANDOM_STATE,
        early_stopping=True,
    )
    svm = CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE, max_iter=2000), n_jobs=-1
    )
    return {"RF": rf, "NN": mlp, "SVM": svm}
