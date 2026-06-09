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
from hif.config import PRECISION_WEIGHT, RANDOM_STATE, set_verbosity, vprint
from hif.ensemble import HIFEnsemble
from hif.lof import run_lof
from hif.metrics import (
    evaluate,
    plot_balanced_accuracy,
    plot_confusion,
    plot_precision,
    plot_radar,
)


def _section(title):
    vprint("\n" + "=" * 60)
    vprint(title)
    vprint("=" * 60)


def _print_environment(args):
    """Verbose (-vv) startup banner: environment and resolved configuration."""
    import platform

    import sklearn
    from hif.config import (ENSEMBLE_CONFIGS, TEST_SIZE, TOP_K_FEATURES,
                            VAL_SIZE)
    from hif.optimize import SEARCH_SPACES

    print("\n" + "=" * 60)
    print("0. Environment and configuration")
    print("=" * 60)
    print("  Python      : %s" % platform.python_version())
    print("  numpy       : %s" % np.__version__)
    print("  pandas      : %s" % pd.__version__)
    print("  scikit-learn: %s" % sklearn.__version__)
    try:
        import imblearn
        print("  imbalanced-learn: %s" % imblearn.__version__)
    except Exception:
        pass
    print("  seed              : %d" % args.seed)
    print("  top-K features    : %d (mutual information)" % TOP_K_FEATURES)
    print("  split             : train %.0f%% / val %.0f%% / test %.0f%%"
          % ((1 - TEST_SIZE - VAL_SIZE) * 100, VAL_SIZE * 100, TEST_SIZE * 100))
    print("  precision weight  : %.2f" % PRECISION_WEIGHT)
    print("  HIF ensemble      : %s" % ENSEMBLE_CONFIGS)
    if args.optimize:
        print("  Optuna trials     : %d per baseline" % args.n_trials)
        for name, space in SEARCH_SPACES.items():
            print("    %-3s search space: %s" % (name, space))


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
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="console detail: 0 = results table only, "
                             "1 = announce every step (default), 2 = also dump "
                             "environment/config, per-feature MI scores, library "
                             "warnings and Optuna trial logs")
    parser.add_argument("--resume", action="store_true",
                        help="reuse checkpoints in --checkpoint-dir and skip the "
                             "phases already completed (preprocessing and each "
                             "finished model). Lets a killed run continue.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="where phase checkpoints are written/read "
                             "(default: <output>/checkpoints). Point it outside "
                             "the repo to survive a re-clone.")
    args = parser.parse_args()

    set_verbosity(args.verbose)
    # Library warnings are noise at the normal levels; show them only at -vv.
    # PYTHONWARNINGS is also set so the filter reaches joblib/loky worker
    # processes (e.g. the SVM calibration folds), which do not inherit the
    # parent's in-process warnings filter.
    if args.verbose < 2:
        warnings.filterwarnings("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
    else:
        warnings.simplefilter("default")
        _print_environment(args)

    np.random.seed(args.seed)
    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ckpt_dir = args.checkpoint_dir or os.path.join(args.output, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    prep_path = os.path.join(ckpt_dir, "prep.npz")
    results_path = os.path.join(ckpt_dir, "results.json")
    hif_scores_path = os.path.join(ckpt_dir, "hif_scores.npy")
    hif_meta_path = os.path.join(ckpt_dir, "hif_meta.json")
    meta_path = os.path.join(ckpt_dir, "run_meta.json")

    run_meta = {"fraction": args.fraction, "seed": args.seed,
                "data": args.data, "nrows": args.nrows}
    if args.resume and os.path.isfile(meta_path):
        with open(meta_path) as fh:
            saved_meta = json.load(fh)
        if saved_meta != run_meta:
            print("\n" + "!" * 70)
            print("WARNING: --resume but the run parameters differ from the saved")
            print("checkpoint. Saved: %s" % saved_meta)
            print("Now:   %s" % run_meta)
            print("Delete %s to start clean if this is unintended." % ckpt_dir)
            print("!" * 70 + "\n")

    # ---- Preprocessing (phases 1-6): load from checkpoint or compute ----
    if args.resume and os.path.isfile(prep_path):
        _section("1-6. Preprocessing (resumed from checkpoint)")
        d = np.load(prep_path, allow_pickle=False)
        X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
        y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]
        X_train_bal, y_train_bal = d["X_train_bal"], d["y_train_bal"]
        top_features = list(d["top_features"])
        vprint("  Loaded preprocessed data from %s" % prep_path)
    else:
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

        np.savez(prep_path, X_train=X_train, X_val=X_val, X_test=X_test,
                 y_train=y_train, y_val=y_val, y_test=y_test,
                 X_train_bal=X_train_bal, y_train_bal=y_train_bal,
                 top_features=np.array(top_features))
        with open(meta_path, "w") as fh:
            json.dump(run_meta, fh, indent=2)
        vprint("  Checkpoint saved: %s" % prep_path)

    X_normal = X_train_bal[y_train_bal == 0]
    X_anomalies = X_train_bal[y_train_bal == 1]
    y_anomalies = y_train_bal[y_train_bal == 1]

    # ---- Completed-model results (for --resume) ----
    done = {}
    if args.resume and os.path.isfile(results_path):
        with open(results_path) as fh:
            for r in json.load(fh):
                done[r["Model"]] = r
        if done:
            vprint("Resuming: %d model result(s) already done: %s"
                   % (len(done), ", ".join(done)))

    def _save_results():
        order = ["HIF", "RF", "NN", "SVM", "LOF"]
        rows = [done[m] for m in order if m in done]
        with open(results_path, "w") as fh:
            json.dump(rows, fh, indent=2)

    # ---- Phase 7: HIF ensemble ----
    _section("7. HIF ensemble (3 members)")
    if "HIF" in done and os.path.isfile(hif_scores_path) and os.path.isfile(hif_meta_path):
        with open(hif_meta_path) as fh:
            opt_thr = json.load(fh)["threshold"]
        hif_scores = np.load(hif_scores_path)
        hif_pred = (hif_scores > opt_thr).astype(int)
        vprint("  (resumed) HIF already done; loaded scores and threshold %.4f" % opt_thr)
    else:
        ensemble = HIFEnsemble(configs=_safe_configs(len(X_normal)))
        t0 = time.time()
        ensemble.fit(X_normal, X_anomalies, y_anomalies)
        hif_train_time = time.time() - t0
        vprint("  HIF training time: %.1fs" % hif_train_time)

        vprint("  Optimizing threshold on the validation set ...")
        opt_thr = ensemble.optimize_threshold(X_val, y_val, precision_weight=PRECISION_WEIGHT)
        vprint("  Optimal threshold: %.4f" % opt_thr)

        t0 = time.time()
        hif_scores = ensemble.score_samples(X_test)
        hif_pred = (hif_scores > opt_thr).astype(int)
        hif_eval_time = time.time() - t0
        vprint("  Inference time: %.1fs" % hif_eval_time)

        row = evaluate(y_test, hif_pred, hif_scores, label="HIF")
        row["Train_Time_s"] = hif_train_time
        row["Eval_Time_s"] = hif_eval_time
        done["HIF"] = row
        np.save(hif_scores_path, hif_scores)
        with open(hif_meta_path, "w") as fh:
            json.dump({"threshold": opt_thr}, fh)
        _save_results()

    # ---- Phase 8: supervised baselines (one checkpoint per model) ----
    mode = "Optuna-tuned" if args.optimize else "fixed hyperparameters"
    _section("8. Supervised baselines (RF, NN, SVM) -- %s" % mode)
    default_baselines = build_baselines()
    for name in ["RF", "NN", "SVM"]:
        if name in done:
            vprint("  (resumed) %s already done" % name)
            continue
        vprint("\n  Training %s ..." % name)
        t0 = time.time()
        if args.optimize:
            from hif.optimize import tune_supervised
            # Only the MLP benefits from concurrent trials; RF/SVM already use
            # all cores within a single trial, so keep their trials serial.
            trial_jobs = args.parallel_trials if name == "NN" else 1
            clf, best_params = tune_supervised(
                name, X_train_bal, y_train_bal, X_val, y_val,
                n_trials=args.n_trials, trial_jobs=trial_jobs)
            vprint("    best params: %s" % best_params)
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

        vprint("  Train: %.1fs  |  Inference: %.1fs" % (train_t, eval_t))
        row = evaluate(y_test, y_pred, y_score, label=name)
        row["Train_Time_s"] = train_t
        row["Eval_Time_s"] = eval_t
        done[name] = row
        _save_results()

    # ---- Phase 9: LOF ----
    _section("9. LOF (transductive)")
    if "LOF" in done:
        vprint("  (resumed) LOF already done")
    else:
        t0 = time.time()
        lof_pred, lof_scores = run_lof(X_train, X_test, y_test)
        lof_t = time.time() - t0
        vprint("  Fit + predict time: %.1fs" % lof_t)
        row = evaluate(y_test, lof_pred, lof_scores, label="LOF")
        row["Train_Time_s"] = lof_t
        row["Eval_Time_s"] = 0.0
        done["LOF"] = row
        _save_results()

    # ---- Phase 10: results table ----
    _section("10. Results table")
    order = ["HIF", "RF", "NN", "SVM", "LOF"]
    results = [done[m] for m in order if m in done]
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
    vprint("  Saved figures to %s" % fig_dir)

    print("\nDone. Results written to %s" % args.output)


if __name__ == "__main__":
    main()
