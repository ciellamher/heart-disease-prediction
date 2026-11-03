"""Results dashboard for the heart-disease models.

    streamlit run app.py

Trains the models from the raw data on startup (a few seconds), so it needs
no generated files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from src import plots
from src.modeling import (
    BEST_MODEL, TARGET_RECALL, evaluate, forest_permutation_importance,
    logistic_coefficients, prepare_data, scores,
)

FIG_DIR = Path(__file__).parent / "reports" / "figures"

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
plots.apply_style()


@st.cache_resource(show_spinner="Training models…")
def load():
    X, y = prepare_data()
    results = evaluate(X, y)
    coefs = logistic_coefficients(results, X.columns)
    imp = forest_permutation_importance(results, X, y)
    return X, y, results, coefs, imp


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


X, y, results, coefs, imp = load()
y_test = results.y_test
best_prob = results.test_probs[BEST_MODEL]
tuned = results.test_at_threshold

# --- Header ------------------------------------------------------------------
st.title("Heart Disease Prediction")
st.markdown(
    f"Interpretable models that predict heart disease from routine clinical measurements. "
    f"UCI Cleveland dataset: {len(y)} patients, {X.shape[1]} features."
)
st.warning(
    "**Not medical advice.** This is a learning project on a small 1988 hospital dataset. "
    "It is not validated for clinical use and must not be used to diagnose anyone.",
    icon="⚠️",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", BEST_MODEL)
c2.metric("ROC-AUC (test)", f"{tuned['roc_auc']:.2f}")
c3.metric(f"Recall at threshold {results.threshold}", f"{tuned['recall']:.0%}",
          help="Share of patients with disease the model flags.")
c4.metric("Precision at that threshold", f"{tuned['precision']:.0%}",
          help="Share of flagged patients who really have disease.")

tab_results, tab_threshold, tab_drivers, tab_data = st.tabs(
    ["Model results", "Recall vs. false alarms", "What drives risk", "About the data"])

# --- Model results -------------------------------------------------------------
with tab_results:
    st.markdown(
        "Three baselines trained on 222 patients and scored once on 75 held-out patients "
        "(default threshold 0.5). **Recall matters most:** missing a sick patient is worse "
        "than a false alarm."
    )
    table = results.test.rename(columns=plots.METRIC_LABELS)
    st.dataframe(table.style.format("{:.2f}").highlight_max(axis=0, color="#e4e0fa"),
                 use_container_width=True)
    left, right = st.columns(2)
    with left:
        show(plots.model_comparison(results.test))
    with right:
        show(plots.roc_curves(results.roc, results.test["roc_auc"]))
    with st.expander("Cross-validation on the training set (5 folds, mean)"):
        cv = results.cv[[f"{m}_mean" for m in plots.METRIC_LABELS]]
        cv.columns = list(plots.METRIC_LABELS.values())
        st.dataframe(cv.style.format("{:.3f}"), use_container_width=True)

# --- Threshold explorer --------------------------------------------------------
with tab_threshold:
    st.markdown(
        f"The model outputs a probability. The **threshold** decides when a patient is flagged. "
        f"Lower it to catch more cases at the cost of more false alarms. "
        f"The recall-focused setting ({results.threshold}) was chosen on training data only, "
        f"targeting {TARGET_RECALL:.0%} recall."
    )
    threshold = st.slider("Decision threshold", 0.05, 0.95, float(results.threshold), 0.01)
    pred = (best_prob >= threshold).astype(int)
    s = scores(y_test, pred, best_prob)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recall", f"{s['recall']:.0%}")
    k2.metric("Precision", f"{s['precision']:.0%}")
    k3.metric("Missed cases", int(cm[1, 0]), help=f"of {int(y_test.sum())} patients with disease")
    k4.metric("False alarms", int(cm[0, 1]), help=f"of {int((y_test == 0).sum())} healthy patients")
    left, _ = st.columns([2, 3])
    with left:
        show(plots.confusion_matrices({f"{BEST_MODEL}, threshold {threshold:.2f}": cm}))

# --- Drivers -------------------------------------------------------------------
with tab_drivers:
    st.markdown("""
**In plain language**, the models lean most on:

1. **Blocked vessels on fluoroscopy.** More narrowed vessels on the dye X-ray test means higher risk. This is the strongest factor in both models.
2. **Exercise stress test.** Low max heart rate, ST depression (a dip in the ECG during exercise) and chest pain brought on by exercise all point to a heart that struggles under load.
3. **Thallium scan.** A normal scan lowers risk. A *reversible defect* (blood flow drops during exercise) raises it.
4. **No chest pain ("asymptomatic").** Patients with no chest pain had *more* disease. They were referred for testing anyway, so this often means silent disease.
5. **Male sex** raises predicted risk.

These are associations in one hospital's data, not causes.
""")
    left, right = st.columns(2)
    with left:
        show(plots.logistic_coefficients(coefs))
    with right:
        show(plots.permutation_importance_chart(imp))

# --- Data ----------------------------------------------------------------------
with tab_data:
    st.markdown(
        "[UCI Heart Disease, Cleveland subset](https://archive.ics.uci.edu/dataset/45/heart+disease): "
        "303 Cleveland Clinic patients who underwent coronary angiography (published 1988). 6 rows with missing values "
        "were dropped → 297 patients, 46% with heart disease."
    )
    st.image(str(FIG_DIR / "03_risk_by_group.png"))
    st.caption("Exploratory analysis: share of patients with heart disease by group (see notebooks/03_eda.ipynb).")
