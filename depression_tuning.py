"""
Automated hyperparameter search for the depression pipeline.

Replaces hand-picked parameters with three search strategies — grid, randomized
and Bayesian — mirroring the approach used in the Optimisation Techniques
project, and answers the question that matters: does the search actually beat
the values that were chosen by hand?

Evaluation is honest about tuning optimism. A 20% holdout is split off before
any search runs and is never seen by a search. For each strategy the report
shows the best inner cross-validated score, the holdout score, and the gap
between them — the gap is how much the tuned score flatters itself.

Run:
    python depression_tuning.py [--data-dir .] [--budget 12] [--folds 3]
    python depression_tuning.py --nested        # nested CV; slower, no holdout needed

Requires pandas, numpy, scikit-learn, scikit-optimize.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from depression_pipeline import RANDOM_STATE, TARGET, build_features, make_preprocessor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCORING = "average_precision"   # PR-AUC: the right target at an 18% positive rate


def manual_models() -> dict[str, Pipeline]:
    """The hand-picked configurations, kept as the reference to beat."""
    return {
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


def build_searches(cv, budget: int) -> dict[str, tuple[str, object]]:
    """One search per strategy. Two of them tune the same model, so the
    randomized and Bayesian strategies can be compared like for like."""
    searches: dict[str, tuple[str, object]] = {}

    logistic = Pipeline([
        ("prep", make_preprocessor(True)),
        ("model", LogisticRegression(solver="liblinear", class_weight="balanced",
                                     max_iter=2000, random_state=RANDOM_STATE)),
    ])
    searches["Logistic Regression"] = ("GridSearchCV", GridSearchCV(
        logistic,
        param_grid={"model__C": [0.01, 0.1, 0.5, 1.0, 5.0], "model__penalty": ["l1", "l2"]},
        scoring=SCORING, cv=cv, n_jobs=-1, refit=True,
    ))

    def hgb() -> Pipeline:
        return Pipeline([
            ("prep", make_preprocessor(False)),
            ("model", HistGradientBoostingClassifier(class_weight="balanced",
                                                     early_stopping=True,
                                                     random_state=RANDOM_STATE)),
        ])

    searches["HistGradientBoosting (random)"] = ("RandomizedSearchCV", RandomizedSearchCV(
        hgb(),
        param_distributions={
            "model__learning_rate": loguniform(0.02, 0.3),
            "model__max_iter": randint(120, 500),
            "model__max_leaf_nodes": randint(15, 64),
            "model__min_samples_leaf": randint(10, 80),
            "model__l2_regularization": loguniform(1e-6, 1.0),
            "model__max_features": uniform(0.5, 0.5),
        },
        n_iter=budget, scoring=SCORING, cv=cv, n_jobs=-1,
        random_state=RANDOM_STATE, refit=True,
    ))

    try:
        from skopt import BayesSearchCV
        from skopt.space import Integer, Real
    except ImportError:
        print("  scikit-optimize not installed — skipping the Bayesian search.")
        return searches

    searches["HistGradientBoosting (bayes)"] = ("BayesSearchCV", BayesSearchCV(
        hgb(),
        search_spaces={
            "model__learning_rate": Real(0.02, 0.3, prior="log-uniform"),
            "model__max_iter": Integer(120, 500),
            "model__max_leaf_nodes": Integer(15, 64),
            "model__min_samples_leaf": Integer(10, 80),
            "model__l2_regularization": Real(1e-6, 1.0, prior="log-uniform"),
            "model__max_features": Real(0.5, 1.0),
        },
        n_iter=budget, scoring=SCORING, cv=cv, n_jobs=-1,
        random_state=RANDOM_STATE, refit=True,
    ))
    return searches


def score_both(estimator, X, y) -> tuple[float, float]:
    prob = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, prob), roc_auc_score(y, prob)


def run_holdout(X, y, folds: int, budget: int, out_dir: Path) -> None:
    X_pool, X_hold, y_pool, y_hold = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Search pool {len(X_pool):,} rows · untouched holdout {len(X_hold):,} rows\n")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows, best_params = [], {}

    print("Reference — the hand-picked configurations:")
    for name, pipe in manual_models().items():
        started = time.time()
        pipe.fit(X_pool, y_pool)
        pr, roc = score_both(pipe, X_hold, y_hold)
        rows.append({"model": name, "strategy": "manual", "fits": 1,
                     "cv_pr_auc": np.nan, "holdout_pr_auc": pr, "holdout_roc_auc": roc,
                     "seconds": round(time.time() - started, 1)})
        print(f"  {name:<32} holdout PR-AUC {pr:.4f}")

    print("\nAutomated searches:")
    for name, (strategy, search) in build_searches(cv, budget).items():
        started = time.time()
        print(f"  {name} via {strategy}...", flush=True)
        search.fit(X_pool, y_pool)
        elapsed = time.time() - started
        pr, roc = score_both(search.best_estimator_, X_hold, y_hold)
        n_fits = len(search.cv_results_["params"]) * folds
        # Spread across candidates says whether the budget was the binding
        # constraint: a flat spread means more candidates cannot help.
        cand = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
        rows.append({"model": name, "strategy": strategy, "fits": n_fits,
                     "cv_pr_auc": float(search.best_score_),
                     "holdout_pr_auc": pr, "holdout_roc_auc": roc,
                     "cand_worst": float(np.nanmin(cand)),
                     "cand_spread": float(np.nanmax(cand) - np.nanmin(cand)),
                     "seconds": round(elapsed, 1)})
        best_params[name] = {k: (v.item() if hasattr(v, "item") else v)
                             for k, v in dict(search.best_params_).items()}
        print(f"    cv {search.best_score_:.4f} · holdout {pr:.4f} "
              f"· optimism {search.best_score_ - pr:+.4f} · {n_fits} fits in {elapsed:.0f}s")
        print(f"    candidates spanned {np.nanmin(cand):.4f}–{np.nanmax(cand):.4f} "
              f"(spread {np.nanmax(cand) - np.nanmin(cand):.4f})")

    table = pd.DataFrame(rows).sort_values("holdout_pr_auc", ascending=False)
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "tuning_comparison.csv", index=False)
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print("Ranked by holdout PR-AUC (the honest number)")
    print("=" * 92)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="—"))

    manual_best = max(r["holdout_pr_auc"] for r in rows if r["strategy"] == "manual")
    tuned_best = max(r["holdout_pr_auc"] for r in rows if r["strategy"] != "manual")
    delta = tuned_best - manual_best
    print(f"\nBest hand-picked: {manual_best:.4f}   Best searched: {tuned_best:.4f}   "
          f"Difference: {delta:+.4f}")
    print("Searching " + ("beat" if delta > 0.002 else "did not meaningfully beat")
          + " the hand-picked configuration.")


def run_nested(X, y, folds: int, budget: int, out_dir: Path) -> None:
    """Nested CV: the inner loop tunes, the outer loop scores. Slower, but it
    needs no holdout and removes tuning optimism from the reported number."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, (strategy, search) in build_searches(inner, budget).items():
        started = time.time()
        print(f"  nested {name} via {strategy}...", flush=True)
        scores = cross_val_score(search, X, y, cv=outer, scoring=SCORING, n_jobs=1)
        rows.append({"model": name, "strategy": strategy,
                     "nested_pr_auc": float(scores.mean()), "std": float(scores.std()),
                     "seconds": round(time.time() - started, 1)})
        print(f"    nested PR-AUC {scores.mean():.4f} ± {scores.std():.4f}")

    table = pd.DataFrame(rows).sort_values("nested_pr_auc", ascending=False)
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "nested_cv.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--budget", type=int, default=12, help="candidates per randomized/Bayesian search")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--nested", action="store_true", help="nested CV instead of a holdout")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise SystemExit(f"train.csv not found in {data_dir.resolve()}")

    raw = pd.read_csv(train_path)
    y = raw[TARGET].astype(int)
    X = build_features(raw)
    print(f"Loaded {len(raw):,} rows · positive rate {y.mean():.4f} · scoring on {SCORING}\n")

    out_dir = data_dir / "outputs"
    if args.nested:
        run_nested(X, y, args.folds, args.budget, out_dir)
    else:
        run_holdout(X, y, args.folds, args.budget, out_dir)
    print(f"\nOutputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
