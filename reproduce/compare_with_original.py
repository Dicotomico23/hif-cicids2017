"""Compare our HIF implementation against Marteau's original HIF.

This validates our implementation against the reference it is based on:
Pierre-Francois Marteau's Hybrid Isolation Forest (https://github.com/pfmarteau/HIF).

LICENSING: Marteau's HIF is GPL-2.0+, which is incompatible with this repo's
MIT license. We therefore do NOT vendor (copy) his code. Instead this script
fetches it at run time into .external/HIF (git-ignored, never committed and
never redistributed by us). If you prefer, clone it yourself and pass
--marteau-path. By running this script you obtain Marteau's GPL code directly
from his repository, under its own license.

Both implementations are run on exactly the same preprocessed split (same
seed, features, scaling, SMOTE) and the same (ntrees, sample_size). We report,
for each, the ROC AUC of the three HIF signals and of the combined HIF2 score
(arithmetic mean of the per-signal min-max-normalized values, alpha1=alpha2=0.5).

Usage:
    python reproduce/compare_with_original.py --nrows 50000
    python reproduce/compare_with_original.py --marteau-path /path/to/HIF
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from hif import preprocessing as P  # noqa: E402
from hif.config import ENSEMBLE_CONFIGS, RANDOM_STATE  # noqa: E402
from hif.forest import HybridIsolationForest  # noqa: E402

MARTEAU_URL = "https://github.com/pfmarteau/HIF.git"


def fetch_marteau(path=None):
    """Return a path to Marteau's HIF, cloning it into .external/HIF if needed."""
    if path:
        return path
    dest = os.path.join(_ROOT, ".external", "HIF")
    if not os.path.isdir(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print("Fetching Marteau's GPL HIF into %s ..." % dest)
        subprocess.check_call(["git", "clone", "--depth", "1", MARTEAU_URL, dest])
    return dest


def load_marteau(path):
    """Import Marteau's hif.py under a private name, shimming its missing deps."""
    # The original does `from version import __version__`; provide a stub.
    if "version" not in sys.modules:
        stub = types.ModuleType("version")
        stub.__version__ = "0"
        sys.modules["version"] = stub
    spec = importlib.util.spec_from_file_location(
        "marteau_hif", os.path.join(path, "hif.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _auc(y_true, score):
    try:
        return roc_auc_score(y_true, score)
    except ValueError:
        return float("nan")


def _minmax(z):
    lo, hi = z.min(), z.max()
    return (z - lo) / (hi - lo) if hi > lo else np.zeros_like(z)


def _hif2(iso, sc, sa, a1=0.5, a2=0.5):
    s, c, a = _minmax(iso), _minmax(sc), _minmax(sa)
    return a2 * (a1 * s + (1 - a1) * c) + (1 - a2) * a


def our_signals(X_normal, X_anomalies, X_test, t, psi):
    hif = HybridIsolationForest(t=t, psi=psi)
    hif.fit(X_normal, X_anomalies, np.ones(len(X_anomalies)))
    iso, dx, ratio = hif._signals(X_test)
    return iso, dx, ratio


def marteau_signals(module, X_normal, X_anomalies, X_test, t, psi):
    hf = module.hiForest(X_normal, ntrees=t, sample_size=psi)
    for x in X_anomalies:
        hf.addAnomaly(x, 1)
    hf.computeAnomalyCentroid()
    n = len(X_test)
    s0 = np.zeros(n)
    sc = np.zeros(n)
    sa = np.zeros(n)
    for i, x in enumerate(X_test):
        score, _labs, mean_dist, mean_dist_ratio = hf.computeAggScore(x)
        s0[i], sc[i], sa[i] = score, mean_dist, mean_dist_ratio
    return s0, sc, sa


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nrows", type=int, default=50000,
                    help="subsample size (the comparison is per-tree pure Python "
                         "in the original, so keep this modest)")
    ap.add_argument("--t", type=int, default=100, help="number of trees")
    ap.add_argument("--psi", type=int, default=ENSEMBLE_CONFIGS[0]["psi"],
                    help="subsample size per tree")
    ap.add_argument("--test-cap", type=int, default=20000,
                    help="cap test rows scored (the original scorer is slow)")
    ap.add_argument("--marteau-path", default=None,
                    help="path to an existing clone of pfmarteau/HIF")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    np.random.seed(RANDOM_STATE)

    df = P.load_dataset(nrows=args.nrows)
    df, cols = P.clean(df)
    Xtr, Xval, Xte, ytr, yval, yte = P.split(df, cols)
    idx, _ = P.select_features(Xtr, ytr, cols)
    Xtr, Xte = Xtr[:, idx], Xte[:, idx]
    Xtr, _Xval, Xte, _ = P.scale(Xtr, Xval[:, idx], Xte)
    Xb, yb = P.balance(Xtr, ytr)
    X_normal = Xb[yb == 0]
    X_anom = Xb[yb == 1]

    if args.test_cap and len(Xte) > args.test_cap:
        sel = np.random.RandomState(RANDOM_STATE).choice(
            len(Xte), args.test_cap, replace=False)
        Xte, yte = Xte[sel], yte[sel]

    print("\nTrain normal=%d, anomalies=%d, test=%d (pos rate %.3f)"
          % (len(X_normal), len(X_anom), len(Xte), yte.mean()))
    print("Config: t=%d, psi=%d" % (args.t, args.psi))

    print("\nScoring our implementation ...")
    o_iso, o_dx, o_ratio = our_signals(X_normal, X_anom, Xte, args.t, args.psi)

    print("Fetching and scoring Marteau's original ...")
    module = load_marteau(fetch_marteau(args.marteau_path))
    m_s0, m_sc, m_sa = marteau_signals(module, X_normal, X_anom, Xte, args.t, args.psi)

    rows = [
        {"signal": "isolation", "ours": _auc(yte, o_iso), "marteau": _auc(yte, m_s0)},
        {"signal": "centroid_dist", "ours": _auc(yte, o_dx), "marteau": _auc(yte, m_sc)},
        {"signal": "ratio", "ours": _auc(yte, o_ratio), "marteau": _auc(yte, m_sa)},
        {"signal": "combined_hif2",
         "ours": _auc(yte, _hif2(o_iso, o_dx, o_ratio)),
         "marteau": _auc(yte, _hif2(m_s0, m_sc, m_sa))},
    ]
    table = pd.DataFrame(rows).set_index("signal")
    print("\nROC AUC comparison (higher is better):")
    print(table.to_string(float_format="{:.4f}".format))

    os.makedirs(args.output, exist_ok=True)
    out = os.path.join(args.output, "comparison_original.csv")
    table.reset_index().to_csv(out, index=False)
    print("\nSaved %s" % out)


if __name__ == "__main__":
    main()
