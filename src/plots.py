"""Shared Matplotlib style and model charts (used by notebook 04 and the dashboard).

Colors: blue = no disease / lowers risk, orange = disease / raises risk (same as
the EDA charts). Each model has its own hue, distinct from those two.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NO_DISEASE, DISEASE = "#2a78d6", "#eb6834"
MODEL_COLORS = {
    "Logistic Regression": "#6250d6",  # violet
    "Decision Tree": "#e87ba4",        # magenta
    "Random Forest": "#1baf7a",        # aqua
}
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1", "roc_auc": "ROC-AUC",
}

# Plain-language names for the encoded feature columns
FEATURE_LABELS = {
    "age": "Age",
    "sex": "Male sex",
    "cp_typical_angina": "Chest pain: typical angina",
    "cp_atypical_angina": "Chest pain: atypical angina",
    "cp_non_anginal_pain": "Chest pain: non-anginal",
    "cp_asymptomatic": "Chest pain: none (asymptomatic)",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "High fasting blood sugar",
    "restecg_normal": "Resting ECG: normal",
    "restecg_st_t_abnormality": "Resting ECG: ST-T abnormality",
    "restecg_lv_hypertrophy": "Resting ECG: LV hypertrophy",
    "thalach": "Max heart rate (exercise)",
    "exang": "Angina during exercise",
    "oldpeak": "ST depression (exercise)",
    "slope_upsloping": "ST slope: upsloping",
    "slope_flat": "ST slope: flat",
    "slope_downsloping": "ST slope: downsloping",
    "ca": "Blocked vessels (fluoroscopy)",
    "thal_normal": "Thallium test: normal",
    "thal_fixed_defect": "Thallium test: fixed defect",
    "thal_reversible_defect": "Thallium test: reversible defect",
}

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def _clean_y_axis(ax) -> None:
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def model_comparison(test: pd.DataFrame):
    """Grouped horizontal bars: one group per metric, one bar per model."""
    metrics = list(METRIC_LABELS)
    models = list(test.index)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    h = 0.8 / len(models)
    y = np.arange(len(metrics))
    for i, model in enumerate(models):
        vals = test.loc[model, metrics].values
        pos = y + (i - (len(models) - 1) / 2) * h
        ax.barh(pos, vals, height=h, color=MODEL_COLORS[model], label=model,
                edgecolor=SURFACE, linewidth=2)
        for p, v in zip(pos, vals):
            ax.text(v + 0.008, p, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [METRIC_LABELS[m] for m in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Score on held-out test set (75 patients, threshold 0.5)")
    _clean_y_axis(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1)
    ax.set_title("Test set scores", fontsize=13, pad=30)
    return fig


def roc_curves(roc: dict[str, pd.DataFrame], auc: pd.Series):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(0.62, 0.55, "random guess", color=MUTED, fontsize=8, rotation=38)
    for model, curve in roc.items():
        ax.plot(curve["fpr"], curve["tpr"], color=MODEL_COLORS[model], linewidth=2,
                label=f"{model} (AUC {auc[model]:.2f})")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False-positive rate (healthy patients flagged)")
    ax.set_ylabel("True-positive rate (recall)")
    ax.legend(loc="lower right")
    ax.set_title("ROC curves, test set", fontsize=13)
    return fig


def confusion_matrices(matrices: dict[str, np.ndarray]):
    """Side-by-side confusion matrices, each titled by its dict key."""
    blues = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#86b6ef", "#1c5cab"])
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.6 * len(matrices), 4.2))
    axes = np.atleast_1d(axes)
    names = [["Correctly cleared", "False alarm"], ["Missed case", "Caught case"]]
    for ax, (title, cm) in zip(axes, matrices.items()):
        ax.imshow(cm, cmap=blues, vmin=0, vmax=cm.sum() / 2)
        for i in range(2):
            for j in range(2):
                dark = cm[i, j] > cm.sum() / 4
                ax.text(j, i, f"{cm[i, j]}\n{names[i][j]}", ha="center", va="center",
                        color="#ffffff" if dark else INK, fontsize=10,
                        fontweight="bold" if (i, j) == (1, 0) else "normal")
        ax.set_xticks([0, 1], ["No disease", "Disease"])
        ax.set_yticks([0, 1], ["No disease", "Disease"], rotation=90, va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, fontweight="normal", color=INK)
    fig.tight_layout()
    return fig


def logistic_coefficients(coefs: pd.Series):
    """Diverging bars: orange raises predicted risk, blue lowers it."""
    coefs = coefs.sort_values()
    labels = [FEATURE_LABELS.get(c, c) for c in coefs.index]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(labels, coefs.values, height=0.65,
            color=[DISEASE if v > 0 else NO_DISEASE for v in coefs.values])
    ax.axvline(0, color=AXIS, linewidth=1)
    for y, v in enumerate(coefs.values):
        if abs(v) >= 0.3:
            ax.text(v + (0.02 if v > 0 else -0.02), y, f"{v:+.2f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=8, color=INK)
    lim = np.abs(coefs.values).max() * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Coefficient (change in log-odds per standard deviation)")
    _clean_y_axis(ax)
    ax.legend(handles=[Patch(color=DISEASE, label="Raises predicted risk"),
                       Patch(color=NO_DISEASE, label="Lowers predicted risk")],
              loc="lower right")
    ax.set_title("Logistic regression coefficients", fontsize=13)
    return fig


def permutation_importance_chart(imp: pd.DataFrame, top: int = 10):
    """Top features by drop in test ROC-AUC when shuffled (with ±1 std whiskers)."""
    imp = imp.head(top).iloc[::-1]
    labels = [FEATURE_LABELS.get(c, c) for c in imp.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, imp["importance"], height=0.6, color=MODEL_COLORS["Random Forest"])
    ax.errorbar(imp["importance"], range(len(imp)), xerr=imp["std"], fmt="none",
                ecolor=INK_2, elinewidth=1, capsize=3)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    _clean_y_axis(ax)
    ax.set_title(f"Random forest permutation importance, top {top}", fontsize=13)
    return fig
