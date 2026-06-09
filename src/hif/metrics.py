"""Evaluation metrics and the figures reported in the paper."""

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Metrics shown on the radar chart.
RADAR_METRICS = ["Accuracy", "F1", "Precision", "Recall", "ROC_AUC"]


def evaluate(y_true, y_pred, y_score=None, label=""):
    """Compute the metric suite for one model and return it as a dict."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    bal = balanced_accuracy_score(y_true, y_pred)
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    print("  %s" % label)
    print("    Accuracy      : %.4f" % acc)
    print("    Balanced acc  : %.4f" % bal)
    print("    F1            : %.4f" % f1)
    print("    Precision     : %.4f" % prec)
    print("    Recall        : %.4f" % rec)
    print("    ROC AUC       : %.4f" % auc)
    return {
        "Model": label, "Accuracy": acc, "Balanced_Acc": bal, "F1": f1,
        "Precision": prec, "Recall": rec, "ROC_AUC": auc,
    }


def plot_radar(results_df, out_path):
    """Radar chart comparing the headline metrics across models."""
    model_list = results_df.index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(RADAR_METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = matplotlib.colormaps["tab10"].colors

    for i, model in enumerate(model_list):
        vals = results_df.loc[model, RADAR_METRICS].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.5, label=model, color=colors[i])
        ax.fill(angles, vals, alpha=0.07, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_METRICS, size=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison", size=13, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, out_path):
    """Confusion matrix for a single model (used for HIF)."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("HIF ensemble: confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_balanced_accuracy(results_df, out_path):
    """Bar chart of balanced accuracy across models."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = matplotlib.colormaps["tab10"].colors
    bars = ax.bar(
        results_df.index, results_df["Balanced_Acc"],
        color=[colors[i] for i in range(len(results_df))],
        edgecolor="black", linewidth=0.6,
    )
    for bar, val in zip(bars, results_df["Balanced_Acc"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                "%.4f" % val, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Balanced accuracy by model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precision(results_df, out_path):
    """Bar chart of precision across models."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = matplotlib.colormaps["tab10"].colors
    bars = ax.bar(
        results_df.index, results_df["Precision"],
        color=[colors[i] for i in range(len(results_df))],
        edgecolor="black", linewidth=0.6,
    )
    for bar, val in zip(bars, results_df["Precision"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                "%.4f" % val, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision")
    ax.set_title("Precision by model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
