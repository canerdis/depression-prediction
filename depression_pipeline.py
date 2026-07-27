"""
Mental-health depression prediction — cohort-aware pipeline.

Rebuild of Machine_Learning_Lab_Project for the current dataset (train.csv /
test.csv, 140,700 rows). The previous notebook was written against a
student-only file of 27,901 rows and cannot be pointed at this one: its
cleaning step reduces the data to zero rows. See README.md.

Run:
    python depression_pipeline.py [--data-dir .] [--folds 5] [--submit]

Requires pandas, numpy, scikit-learn.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RANDOM_STATE = 42
TARGET = "Depression"

DIET_MAP = {"Unhealthy": 0.0, "Moderate": 1.0, "Healthy": 2.0}
DEGREE_LEVEL = {"class 12": 0.0, "bachelor": 1.0, "master": 2.0, "doctorate": 3.0}


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def parse_sleep_hours(value) -> float:
    """Sleep duration text -> midpoint hours.

    The old notebook used a four-key lookup built from the student file, where
    the values carried literal quote characters ("'5-6 hours'"). This file has
    them unquoted plus 32 further variants, so that lookup maps 100% of rows to
    NaN. Parsing the numbers out is robust to both.
    """
    if not isinstance(value, str):
        return np.nan
    text = value.strip().strip("'\"").lower()
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return np.nan
    if "less than" in text:
        return max(nums[0] - 1.0, 0.0)
    if "more than" in text:
        return nums[0] + 1.0
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2.0
    return nums[0]


def degree_level(value) -> float:
    """Map a degree string to an ordinal study level."""
    if not isinstance(value, str):
        return np.nan
    text = value.strip().strip("'\"").lower().replace(".", "").replace(" ", "")
    if text.startswith("class"):
        return DEGREE_LEVEL["class 12"]
    if text in {"phd", "md"}:
        return DEGREE_LEVEL["doctorate"]
    if text.startswith("m") or text.startswith("llm"):
        return DEGREE_LEVEL["master"]
    if text.startswith("b") or text.startswith("llb"):
        return DEGREE_LEVEL["bachelor"]
    return np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cohort-aware feature table.

    Academic Pressure / Work Pressure are the same 1-5 question asked of
    different cohorts: 0 rows carry both and only 21 carry neither. Same for
    Study / Job Satisfaction. Unifying each pair turns a column that is 80%
    "missing" into one that is fully populated, and keeps the cohort itself as
    an explicit flag — students are depressed at 58.6% against 8.2% for working
    professionals, which is the strongest single signal in the data.
    """
    out = pd.DataFrame(index=df.index)

    is_student = df["Working Professional or Student"].eq("Student")
    out["is_student"] = is_student.astype(int)

    out["pressure"] = df["Academic Pressure"].combine_first(df["Work Pressure"])
    out["satisfaction"] = df["Study Satisfaction"].combine_first(df["Job Satisfaction"])

    out["age"] = df["Age"]
    out["cgpa"] = df["CGPA"]                    # students only; left NaN elsewhere
    out["work_study_hours"] = df["Work/Study Hours"]
    out["financial_stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")
    out["sleep_hours"] = df["Sleep Duration"].map(parse_sleep_hours)
    out["diet"] = df["Dietary Habits"].map(DIET_MAP)
    out["degree_level"] = df["Degree"].map(degree_level)

    out["suicidal_thoughts"] = df["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
    out["family_history"] = df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})
    out["gender_male"] = df["Gender"].map({"Male": 1, "Female": 0})

    # High-cardinality nominals kept as categories; rare levels folded by the
    # encoder rather than hand-listed, so new junk values cannot break the run.
    out["city"] = df["City"].astype("string").fillna("Unknown")
    out["profession"] = df["Profession"].astype("string").fillna("None")

    # "Name" is an identifier (422 values) and is deliberately not a feature.
    return out


NUMERIC = ["is_student", "pressure", "satisfaction", "age", "cgpa", "work_study_hours",
           "financial_stress", "sleep_hours", "diet", "degree_level",
           "suicidal_thoughts", "family_history", "gender_male"]
CATEGORICAL = ["city", "profession"]


def make_preprocessor(sparse_ok: bool) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=50, sparse_output=sparse_ok), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models() -> dict[str, Pipeline]:
    """All models keep every row and handle imbalance with class weights.

    The previous approach deleted majority-class rows until the classes were
    even. On this data that would discard 89,566 rows — 63.7% of the sample —
    and it also rebalanced the test split, so accuracy was being measured on a
    distribution that does not exist.
    """
    return {
        "Majority baseline": Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(True)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                         random_state=RANDOM_STATE)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", make_preprocessor(False)),
            ("model", HistGradientBoostingClassifier(class_weight="balanced",
                                                     max_iter=300, learning_rate=0.08,
                                                     early_stopping=True,
                                                     random_state=RANDOM_STATE)),
        ]),
    }


SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "accuracy": "accuracy",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--submit", action="store_true", help="also write submission.csv for test.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise SystemExit(f"train.csv not found in {data_dir.resolve()}")

    raw = pd.read_csv(train_path)
    y = raw[TARGET].astype(int)
    X = build_features(raw)

    print(f"Loaded {len(raw):,} rows · positive rate {y.mean():.4f}")
    print(f"  cohort mix: {dict(raw['Working Professional or Student'].value_counts())}")
    print(f"  a model predicting the majority class every time scores {max(y.mean(), 1-y.mean()):.4f} accuracy\n")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in build_models().items():
        print(f"Cross-validating {name}...")
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score="raise")
        row = {"model": name}
        for metric in SCORING:
            row[metric] = float(np.mean(res["test_" + metric]))
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    print("\n" + "=" * 78)
    print(f"{args.folds}-fold cross-validated results")
    print("=" * 78)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out_dir = data_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "model_comparison.csv", index=False)

    # ---- permutation importance on a held-out split, best model -------------
    best_name = str(table.iloc[0]["model"])
    best = build_models()[best_name]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
    best.fit(X_tr, y_tr)
    print(f"\nPermutation importance for {best_name} (drop in ROC-AUC):")
    imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": X.columns,
                            "importance": imp.importances_mean,
                            "std": imp.importances_std})
              .sort_values("importance", ascending=False))
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    print(imp_df.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summary = {
        "rows": int(len(raw)),
        "positive_rate": float(y.mean()),
        "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
        "folds": args.folds,
        "best_model": best_name,
        "results": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.submit:
        test_path = data_dir / "test.csv"
        if not test_path.exists():
            print(f"\ntest.csv not found in {data_dir.resolve()}; skipping submission.")
        else:
            test_raw = pd.read_csv(test_path)
            full = build_models()[best_name].fit(X, y)
            pred = full.predict(build_features(test_raw))
            pd.DataFrame({"id": test_raw["id"], TARGET: pred.astype(int)}).to_csv(
                out_dir / "submission.csv", index=False)
            print(f"\nWrote {len(test_raw):,} predictions to {out_dir / 'submission.csv'}")

    print(f"\nOutputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
