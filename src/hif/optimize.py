"""Optuna hyperparameter optimization for the supervised baselines.

Each supervised baseline (RF, NN, SVM) is tuned independently by maximizing the
F1-score on the validation set. The model is fitted on the (SMOTE-balanced)
training set and scored on the held-out validation set; the test set is never
used for tuning. LOF (unsupervised) and the HIF ensemble are not tuned here.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE

# Hidden-layer presets for the MLP (Optuna categoricals must be simple values).
_MLP_LAYERS = {"128-64": (128, 64), "256-128": (256, 128), "64-32": (64, 32)}


def _rf(trial):
    return RandomForestClassifier(
        n_estimators=trial.suggest_categorical("n_estimators", [100, 200, 300]),
        max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        n_jobs=-1, random_state=RANDOM_STATE,
    )


def _mlp(trial):
    layers = _MLP_LAYERS[trial.suggest_categorical("hidden_layers", list(_MLP_LAYERS))]
    return MLPClassifier(
        hidden_layer_sizes=layers,
        alpha=trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        max_iter=200, early_stopping=True, random_state=RANDOM_STATE,
    )


def _svm(trial):
    base = LinearSVC(
        C=trial.suggest_float("C", 1e-2, 1e2, log=True),
        max_iter=2000, random_state=RANDOM_STATE,
    )
    return CalibratedClassifierCV(base, n_jobs=-1)


_FACTORIES = {"RF": _rf, "NN": _mlp, "SVM": _svm}

# Human-readable search spaces, for documentation/reporting.
SEARCH_SPACES = {
    "RF": "n_estimators in {100,200,300}; max_depth in {None,10,20,30}; "
          "min_samples_split in [2,10]; max_features in {sqrt,log2,None}",
    "NN": "hidden_layer_sizes in {(128,64),(256,128),(64,32)}; "
          "alpha in [1e-5,1e-2] (log); learning_rate_init in [1e-4,1e-2] (log)",
    "SVM": "C in [1e-2,1e2] (log), LinearSVC + probability calibration",
}


def tune_supervised(name, X_train, y_train, X_val, y_val, n_trials=20,
                    seed=RANDOM_STATE, trial_jobs=1):
    """Tune one supervised baseline with Optuna; return (fitted_model, best_params).

    The objective is the validation F1-score. The returned model is refitted on
    the training set with the best hyperparameters.

    The whole pipeline is CPU-only (no GPU benefit). RF parallelizes internally
    and the SVM parallelizes its calibration folds, so their trials are best run
    one at a time. MLP has no inner parallelism, so on a multi-core CPU you can
    speed it up by running its Optuna trials concurrently via ``trial_jobs``.
    Note: ``trial_jobs > 1`` makes the TPE search non-deterministic (concurrent
    trials cannot see each other's results), so the default is 1 for exact
    reproducibility.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    factory = _FACTORIES[name]

    def objective(trial):
        model = factory(trial)
        model.fit(X_train, y_train)
        return f1_score(y_val, model.predict(X_val), zero_division=0)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=trial_jobs,
                   show_progress_bar=False)

    best_model = factory(study.best_trial)  # FrozenTrial replays stored params
    best_model.fit(X_train, y_train)
    return best_model, study.best_params
