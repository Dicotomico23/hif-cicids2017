"""End-to-end reproduction of the HIF comparison study on CICIDS2017.

Runs the full pipeline (load, clean, split, feature selection, scaling,
Borderline-SMOTE, HIF ensemble training, threshold selection, baseline
training, evaluation) and writes the results table and figures.

Usage:
    python reproduce/run_comparison.py
    python reproduce/run_comparison.py --output results --seed 42
    python reproduce/run_comparison.py --nrows 50000   # quick partial run

The default run reproduces the numbers and figures reported in the paper.
The --nrows flag takes a random subsample for a faster, partial check; it
does not reproduce the paper's exact numbers.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# Allow running without installing the package (python reproduce/run_comparison.py).
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, os.path.abspath(_SRC))

from hif import preprocessing
from hif.baselines import build_baselines
from hif.config import PRECISION_WEIGHT, RANDOM_STATE
from hif.ensemble import HIFEnsemble
from hif.lof import run_lof
from hif.metrics import (
    evaluate,
    plot_balanced_accuracy,
    plot_confusion,
    plot_precision,
    plot_radar,
)

warnings.filterwarnings("ignore")


def _section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _safe_configs(n_normal):
    """Cap each member's psi to the available benign training rows.

    On the full dataset this leaves the configured values untouched. On a small
    --nrows subsample it prevents psi from exceeding the training set size.
    """
    from hif.config import ENSEMBLE_CONFIGS

    configs = []
    for cfg in ENSEMBLE_CONFIGS:
        psi = min(cfg["psi"], max(2, n_normal // 2))
        configs.append({"t": cfg["t"], "psi": psi})
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results", help="output directory")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="random seed")
    parser.add_argument("--nrows", type=int, default=None,
                        help="random raw-row cap before cleaning (quick dev run)")
    parser.add_argument("--fraction", type=float, default=None,
                        help="keep a stratified fraction in (0,1] of the cleaned "
                             "data (preserves class balance; faster runs)")
    parser.add_argument("--data", default=None,
                        help="path to a CSV file or directory to use as the "
                             "dataset (e.g. data/sample/cicids2017_sample.csv)")
    parser.add_argument("--optimize", action="store_true",
                        help="tune the supervised baselines with Optuna")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Optuna trials per baseline when --optimize is set")
    parser.add_argument("--parallel_trials", type=int, default=1,
                        help="concurrent Optuna trials for the MLP (CPU-only; "
                             ">1 speeds up NN tuning but makes TPE "
                             "non-deterministic). RF/SVM ignore this.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    _section("1. Load CICIDS2017")
    df = preprocessing.load_dataset(nrows=args.nrows, path=args.data)

    _section("2. Clean and binarize labels (0 = BENIGN, 1 = attack)")
    df, numeric_cols = preprocessing.clean(df)
    if args.fraction:
        df = preprocessing.stratified_sample(df, args.fraction)

    _section("3. Stratified 70/15/15 split")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split(df, numeric_cols)

    _section("4. Feature selection (mutual information)")
    col_idx, top_features = preprocessing.select_features(X_train, y_train, numeric_cols)
    X_train = X_train[:, col_idx]
    X_val = X_val[:, col_idx]
    X_test = X_test[:, col_idx]

    _section("5. Standard scaling (fit on train only)")
    X_train, X_val, X_test, _ = preprocessing.scale(X_train, X_val, X_test)

    _section("6. Borderline-SMOTE on the training set")
    X_train_bal, y_train_bal = preprocessing.balance(X_train, y_train)
    X_normal = X_train_bal[y_train_bal == 0]
    X_anomalies = X_train_bal[y_train_bal == 1]
    y_anomalies = y_train_bal[y_train_bal == 1]

    results = []

    _section("7. HIF ensemble (3 members)")
    ensemble = HIFEnsemble(configs=_safe_configs(len(X_normal)))
    t0 = time.time()
    ensemble.fit(X_normal, X_anomalies, y_anomalies)
    hif_train_time = time.time() - t0
    print("  HIF training time: %.1fs" % hif_train_time)

    print("  Optimizing threshold on the validation set ...")
    opt_thr = ensemble.optimize_threshold(X_val, y_val, precision_weight=PRECISION_WEIGHT)
    print("  Optimal threshold: %.4f" % opt_thr)

    t0 = time.time()
    hif_scores = ensemble.score_samples(X_test)
    hif_pred = (hif_scores > opt_thr).astype(int)
    hif_eval_time = time.time() - t0
    print("  Inference time: %.1fs" % hif_eval_time)

    row = evaluate(y_test, hif_pred, hif_scores, label="HIF")
    row["Train_Time_s"] = hif_train_time
    row["Eval_Time_s"] = hif_eval_time
    results.append(row)

    mode = "Optuna-tuned" if args.optimize else "fixed hyperparameters"
    _section("8. Supervised baselines (RF, NN, SVM) -- %s" % mode)
    default_baselines = build_baselines()
    for name in ["RF", "NN", "SVM"]:
        print("\n  Training %s ..." % name)
        t0 = time.time()
        if args.optimize:
            from hif.optimize import tune_supervised
            # Only the MLP benefits from concurrent trials; RF/SVM already use
            # all cores within a single trial, so keep their trials serial.
            trial_jobs = args.parallel_trials if name == "NN" else 1
            clf, best_params = tune_supervised(
                name, X_train_bal, y_train_bal, X_val, y_val,
                n_trials=args.n_trials, trial_jobs=trial_jobs)
            print("    best params: %s" % best_params)
        else:
            clf = default_baselines[name]
            clf.fit(X_train_bal, y_train_bal)
        train_t = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(X_test)
        try:
            y_score = clf.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_score = None
        eval_t = time.time() - t0

        print("  Train: %.1fs  |  Inference: %.1fs" % (train_t, eval_t))
        row = evaluate(y_test, y_pred, y_score, label=name)
        row["Train_Time_s"] = train_t
        row["Eval_Time_s"] = eval_t
        results.append(row)

    _section("9. LOF (transductive)")
    t0 = time.time()
    lof_pred, lof_scores = run_lof(X_train, X_test, y_test)
    lof_t = time.time() - t0
    print("  Fit + predict time: %.1fs" % lof_t)
    row = evaluate(y_test, lof_pred, lof_scores, label="LOF")
    row["Train_Time_s"] = lof_t
    row["Eval_Time_s"] = 0.0
    results.append(row)

    _section("10. Results table")
    results_df = pd.DataFrame(results).set_index("Model")
    cols = ["Accuracy", "Balanced_Acc", "F1", "Precision", "Recall", "ROC_AUC",
            "Train_Time_s", "Eval_Time_s"]
    print(results_df[cols].to_string(float_format="{:.4f}".format))

    results_df.reset_index().to_csv(
        os.path.join(args.output, "table5_results.csv"), index=False
    )
    with open(os.path.join(args.output, "table5_results.json"), "w") as fh:
        json.dump(results_df.reset_index().to_dict(orient="records"), fh, indent=2)

    _section("11. Figures")
    plot_radar(results_df, os.path.join(fig_dir, "fig_radar_metrics.png"))
    plot_confusion(y_test, hif_pred, os.path.join(fig_dir, "fig_hif_confusion_matrix.png"))
    plot_balanced_accuracy(results_df, os.path.join(fig_dir, "fig_bar_balanced_acc.png"))
    plot_precision(results_df, os.path.join(fig_dir, "fig_bar_precision.png"))
    print("  Saved figures to %s" % fig_dir)

    print("\nDone. Results written to %s" % args.output)


if __name__ == "__main__":
    main()
