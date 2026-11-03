"""Train and evaluate the baseline models.

Run from the project root to print the results and write them to reports/:

    python -m src.modeling
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.cleaning import clean, encode
from src.data_loading import PROJECT_ROOT, load_raw

REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.25
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
TARGET_RECALL = 0.90  # catch at least 9 in 10 patients with disease
BEST_MODEL = "Logistic Regression"


def prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Raw file -> cleaned -> one-hot encoded features X and target y.

    Built from the raw file (not data/processed/) so it works on a fresh clone.
    """
    encoded = encode(clean(load_raw()))
    X = encoded.drop(columns=["patient_id", "target"])
    y = encoded["target"]
    return X, y


def build_models() -> dict[str, ClassifierMixin]:
    """The three baselines, with light settings to limit overfitting on 297 rows."""
    return {
        # Scaling puts every coefficient on the same footing (per standard deviation).
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        # A shallow tree stays readable and doesn't memorise the training set.
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE),
    }


def split(X: pd.DataFrame, y: pd.Series):
    """Stratified 75/25 split, so both parts keep the ~46% disease rate."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def scores(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


@dataclass
class Results:
    cv: pd.DataFrame                      # 5-fold CV on the training set: mean & std per metric
    test: pd.DataFrame                    # held-out test set, threshold 0.5
    roc: dict[str, pd.DataFrame]          # test-set ROC curve per model
    confusion: dict[str, np.ndarray]      # test-set confusion matrix per model
    test_probs: dict[str, np.ndarray]     # test-set P(disease) per model
    y_test: pd.Series
    threshold: float                      # recall-focused threshold for BEST_MODEL, chosen on training data
    test_at_threshold: dict[str, float]   # BEST_MODEL test scores at that threshold
    fitted: dict[str, ClassifierMixin] = field(repr=False)


def evaluate(X: pd.DataFrame, y: pd.Series) -> Results:
    """Cross-validate on the training set, then fit and score on the test set.

    The CV numbers are used to compare models (5 estimates, less noisy);
    the test set is touched once, for the final reported numbers.
    """
    X_train, X_test, y_train, y_test = split(X, y)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_rows, test_rows = [], []
    roc, confusion, probs, fitted = {}, {}, {}, {}
    for name, model in build_models().items():
        cv = cross_validate(model, X_train, y_train, cv=folds, scoring=METRICS)
        cv_rows.append({"model": name,
                        **{f"{m}_mean": cv[f"test_{m}"].mean() for m in METRICS},
                        **{f"{m}_std": cv[f"test_{m}"].std() for m in METRICS}})

        model = clone(model).fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        test_rows.append({"model": name, **scores(y_test, pred, prob)})
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc[name] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        confusion[name] = confusion_matrix(y_test, pred)
        probs[name] = prob
        fitted[name] = model

    threshold = choose_threshold(build_models()[BEST_MODEL], X_train, y_train, folds)
    best_prob = probs[BEST_MODEL]
    at_threshold = scores(y_test, (best_prob >= threshold).astype(int), best_prob)

    return Results(
        threshold=threshold, test_at_threshold=at_threshold,
        cv=pd.DataFrame(cv_rows).set_index("model"),
        test=pd.DataFrame(test_rows).set_index("model"),
        roc=roc, confusion=confusion, test_probs=probs, y_test=y_test, fitted=fitted,
    )


def choose_threshold(model, X_train, y_train, folds, target_recall=TARGET_RECALL) -> float:
    """Highest threshold whose out-of-fold recall on the training set is >= target.

    Missing a sick patient costs more than a false alarm, so we lower the
    default 0.5 cut-off until recall reaches the target. Only training data
    is used, so the test set stays an honest check.
    """
    oof = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    for t in np.arange(0.50, 0.04, -0.01):
        if recall_score(y_train, (oof >= t).astype(int)) >= target_recall:
            return round(float(t), 2)
    return 0.05


def logistic_coefficients(results: Results, feature_names) -> pd.Series:
    """Standardised logistic-regression coefficients, sorted.

    Positive = higher value -> higher predicted risk. Each unit is one
    standard deviation of the feature, so sizes are comparable.
    """
    lr = results.fitted["Logistic Regression"].named_steps["model"]
    return pd.Series(lr.coef_[0], index=feature_names).sort_values()


def forest_permutation_importance(results: Results, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """How much the random forest's test ROC-AUC drops when a feature is shuffled.

    Measured on the held-out test set, so it reflects what the model relies on
    for new patients, not what it memorised.
    """
    _, X_test, _, y_test = split(X, y)
    imp = permutation_importance(
        results.fitted["Random Forest"], X_test, y_test,
        scoring="roc_auc", n_repeats=50, random_state=RANDOM_STATE)
    return (pd.DataFrame({"importance": imp.importances_mean, "std": imp.importances_std},
                         index=X.columns)
            .sort_values("importance", ascending=False))


def threshold_table(y_true, y_prob, thresholds=np.arange(0.1, 0.91, 0.05)) -> pd.DataFrame:
    """Precision / recall / missed cases at a range of decision thresholds."""
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({"threshold": round(t, 2),
                     "recall": recall_score(y_true, pred, zero_division=0),
                     "precision": precision_score(y_true, pred, zero_division=0),
                     "missed_cases": fn, "false_alarms": fp})
    return pd.DataFrame(rows)


def main() -> None:
    X, y = prepare_data()
    results = evaluate(X, y)
    REPORTS_DIR.mkdir(exist_ok=True)
    results.cv.round(3).to_csv(REPORTS_DIR / "metrics_cv.csv")
    results.test.round(3).to_csv(REPORTS_DIR / "metrics_test.csv")
    print(f"Train/test: {int(len(y) * (1 - TEST_SIZE))}/{len(y) - int(len(y) * (1 - TEST_SIZE))} rows")
    print("\n5-fold CV on training set (mean):")
    print(results.cv[[f"{m}_mean" for m in METRICS]].round(3).to_string())
    print("\nHeld-out test set (threshold 0.5):")
    print(results.test.round(3).to_string())
    print(f"\n{BEST_MODEL} at recall-focused threshold {results.threshold}:")
    print({k: round(v, 3) for k, v in results.test_at_threshold.items()})


if __name__ == "__main__":
    main()
