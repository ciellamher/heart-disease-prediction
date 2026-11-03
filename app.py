"""Results dashboard for the heart-disease models.

    streamlit run app.py

Trains the models from the raw data on startup (a few seconds), so it needs
no generated files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix

from src import plots
from src.modeling import (
    BEST_MODEL, TARGET_RECALL, evaluate, forest_permutation_importance,
    logistic_coefficients, prepare_data, scores,
)

FIG_DIR = Path(__file__).parent / "reports" / "figures"

st.set_page_config(page_title="Heart disease prediction", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 1180px; padding-top: 2.5rem; }
      h1 { font-size: 1.9rem !important; font-weight: 600 !important; letter-spacing: -0.01em; }
      [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 500; }
      [data-testid="stMetricLabel"] p { color: #52514e; }
      [data-testid="stCaptionContainer"] { color: #52514e; }
    </style>
    """,
    unsafe_allow_html=True,
)
plots.apply_style()


@st.cache_resource(show_spinner="Training models...")
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
tuned_cm = confusion_matrix(y_test, (best_prob >= results.threshold).astype(int), labels=[0, 1])
n_sick = int(y_test.sum())

st.title("Heart disease prediction")
st.caption(
    f"UCI Cleveland data, {len(y)} patients. Logistic regression, decision tree and random forest, "
    f"compared on a held-out test set of {len(y_test)}. Student project, not for medical use."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC", f"{tuned['roc_auc']:.2f}")
c2.metric("Recall", f"{tuned['recall']:.0%}")
c3.metric("Precision", f"{tuned['precision']:.0%}")
c4.metric("Missed cases", f"{int(tuned_cm[1, 0])} of {n_sick}")
st.caption(f"Logistic regression on the test set, flagging patients above a {results.threshold} probability.")

st.write("")
tab_results, tab_threshold, tab_features, tab_data = st.tabs(
    ["Results", "Threshold", "Features", "Data"])

with tab_results:
    st.write(
        f"Each model was trained on {len(y) - len(y_test)} patients and scored once on the other "
        f"{len(y_test)}, using a 0.5 threshold. Logistic regression did best on recall, the metric "
        "that matters most here, and is also the easiest to explain."
    )
    table = results.test.rename(columns=plots.METRIC_LABELS)
    st.dataframe(table.style.format("{:.2f}"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        show(plots.model_comparison(results.test))
    with right:
        show(plots.roc_curves(results.roc, results.test["roc_auc"]))
    with st.expander("Cross-validation, training set (5 folds)"):
        cv = results.cv[[f"{m}_mean" for m in plots.METRIC_LABELS]]
        cv.columns = list(plots.METRIC_LABELS.values())
        st.dataframe(cv.style.format("{:.3f}"), use_container_width=True)

with tab_threshold:
    st.write(
        "The model gives each patient a probability, and anyone above the threshold is flagged. "
        "A lower threshold misses fewer sick patients but flags more healthy ones. "
        f"The default of {results.threshold} was chosen on the training data to reach about "
        f"{TARGET_RECALL:.0%} recall."
    )
    threshold = st.slider("Threshold", 0.05, 0.95, float(results.threshold), 0.01)
    pred = (best_prob >= threshold).astype(int)
    s = scores(y_test, pred, best_prob)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recall", f"{s['recall']:.0%}")
    k2.metric("Precision", f"{s['precision']:.0%}")
    k3.metric("Missed cases", f"{int(cm[1, 0])} of {n_sick}")
    k4.metric("False alarms", f"{int(cm[0, 1])} of {int((y_test == 0).sum())}")
    left, _ = st.columns([2, 3])
    with left:
        show(plots.confusion_matrices({f"Threshold {threshold:.2f}": cm}))

with tab_features:
    st.write(
        "Both models lean on the same few measurements. The strongest is the number of major "
        "vessels narrowed on fluoroscopy. Next come the exercise stress test (lower peak heart rate, "
        "ST depression, chest pain during exercise) and the thallium scan, where a normal result "
        "lowers the predicted risk."
    )
    st.write(
        "Two results are less obvious. Patients with no chest pain had more disease than those "
        "with it, probably because they were referred for testing for other reasons. And cholesterol "
        "barely mattered on its own but carries some weight once the other factors are included. "
        "None of this shows cause and effect."
    )
    left, right = st.columns(2)
    with left:
        show(plots.logistic_coefficients(coefs))
    with right:
        show(plots.permutation_importance_chart(imp))

with tab_data:
    st.write(
        "The data is the Cleveland subset of the "
        "[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease): "
        "303 patients who had coronary angiography, published in 1988. Six rows with missing "
        "test results were dropped, leaving 297, of whom 46% had heart disease."
    )
    st.image(str(FIG_DIR / "03_risk_by_group.png"))
    st.caption(
        "This is a portfolio project on a small, decades-old dataset. It has not been validated "
        "clinically and should not be used to diagnose anyone."
    )
