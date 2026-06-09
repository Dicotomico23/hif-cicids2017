"""Fast smoke test on synthetic data (no dataset download required).

Verifies that the HIF ensemble, the baselines, LOF, the metric suite and the
figure generators all run end to end and produce outputs of the right shape.
Run with:  python -m pytest -q   or   python tests/test_smoke.py
"""

import os
import sys
import tempfile

import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

import pandas as pd  # noqa: E402

from hif.baselines import build_baselines  # noqa: E402
from hif.ensemble import HIFEnsemble  # noqa: E402
from hif.lof import run_lof  # noqa: E402
from hif.metrics import evaluate, plot_radar  # noqa: E402


def _synthetic(n_normal=400, n_attack=100, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(n_normal, dim))
    attack = rng.normal(3.0, 1.0, size=(n_attack, dim))
    X = np.vstack([normal, attack])
    y = np.concatenate([np.zeros(n_normal, int), np.ones(n_attack, int)])
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def test_pipeline_runs():
    X_train, y_train = _synthetic(seed=1)
    X_val, y_val = _synthetic(seed=2)
    X_test, y_test = _synthetic(seed=3)

    X_normal = X_train[y_train == 0]
    X_anom = X_train[y_train == 1]

    ensemble = HIFEnsemble(configs=[{"t": 20, "psi": 64}, {"t": 20, "psi": 32}])
    ensemble.fit(X_normal, X_anom, y_train[y_train == 1], verbose=False)
    thr = ensemble.optimize_threshold(X_val, y_val)
    pred = (ensemble.score_samples(X_test) > thr).astype(int)
    assert pred.shape == y_test.shape

    for name, clf in build_baselines().items():
        clf.fit(X_train, y_train)
        assert clf.predict(X_test).shape == y_test.shape

    lof_pred, lof_scores = run_lof(X_train, X_test, y_test)
    assert lof_pred.shape == y_test.shape

    row = evaluate(y_test, pred, ensemble.score_samples(X_test), label="HIF")
    assert 0.0 <= row["Precision"] <= 1.0

    with tempfile.TemporaryDirectory() as d:
        df = pd.DataFrame([row]).set_index("Model")
        out = os.path.join(d, "radar.png")
        plot_radar(df, out)
        assert os.path.exists(out)

    print("smoke test passed")


if __name__ == "__main__":
    test_pipeline_runs()
