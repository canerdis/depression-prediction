"""Evaluation layer for the depression pipeline.

Answers the questions the pipeline leaves open: which threshold to deploy at,
whether the probabilities mean anything, whether `age` and `is_student` are the
same signal, which positives get missed, and whether the gap between the two
models is larger than noise.

Nothing here changes the models. See docs/evaluation-design.md.

Run:
    python evaluate.py [--data-dir .] [--bootstrap 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             brier_score_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from depression_pipeline import RANDOM_STATE, TARGET, build_features, make_preprocessor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRECISION_FLOOR = 0.5
RULES = ("f1", "f2", "recall_at_precision")


def _fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """F-beta across arrays, with 0 where precision and recall are both 0."""
    b2 = beta * beta
    denom = b2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (1 + b2) * precision * recall / denom
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def choose_threshold(y_true, y_prob, rule: str = "f2") -> tuple[float, dict]:
    """Pick a decision threshold on one rule, and report its metrics.

    `precision_recall_curve` returns `thresholds` one element shorter than
    `precision` and `recall`. The final point is the degenerate recall=0 corner
    that no threshold produces. Dropping it is what keeps the indices aligned;
    without that the returned threshold belongs to a different operating point
    than the returned metrics.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0}
    if len(np.unique(y_true)) < 2:
        return 0.5, empty

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    if len(thresholds) == 0:
        return 0.5, empty

    if rule == "recall_at_precision":
        eligible = precision >= PRECISION_FLOOR
        if not eligible.any():
            idx = int(np.argmax(precision))
        else:
            masked = np.where(eligible, recall, -1.0)
            idx = int(np.argmax(masked))
    else:
        beta = 1.0 if rule == "f1" else 2.0
        idx = int(np.argmax(_fbeta(precision, recall, beta)))

    return float(thresholds[idx]), {
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(_fbeta(precision, recall, 1.0)[idx]),
        "f2": float(_fbeta(precision, recall, 2.0)[idx]),
    }


def _metrics_at(y_true, y_prob, threshold: float) -> dict:
    """Compute precision, recall, f1, f2 at a single threshold."""
    pred = (y_prob >= threshold).astype(int)
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred, zero_division=0)
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    f2 = 0.0 if 4 * p + r == 0 else 5 * p * r / (4 * p + r)
    return {"precision": p, "recall": r, "f1": f1, "f2": f2}


def run_threshold_selection(models, split, out_dir: Path) -> dict:
    """Choose on validation, report on test.

    The 0.5 row is what the pipeline's `.predict()` silently uses. Quoting it
    beside the chosen thresholds is what makes the cost of the default visible.
    """
    rows = []
    chosen = {}
    for name, pipe in models.items():
        val_prob = pipe.predict_proba(split.X_val)[:, 1]
        test_prob = pipe.predict_proba(split.X_test)[:, 1]

        for rule in RULES:
            thr, val_metrics = choose_threshold(split.y_val, val_prob, rule=rule)
            test_metrics = _metrics_at(split.y_test, test_prob, thr)
            rows.append({"model": name, "rule": rule, "threshold": thr,
                         **{f"val_{k}": v for k, v in val_metrics.items()},
                         **{f"test_{k}": v for k, v in test_metrics.items()}})
            if rule == "f2":
                chosen[name] = thr

        default_metrics = _metrics_at(split.y_test, test_prob, 0.5)
        rows.append({"model": name, "rule": "default_0.5", "threshold": 0.5,
                     **{f"val_{k}": float("nan") for k in ("precision", "recall", "f1", "f2")},
                     **{f"test_{k}": v for k, v in default_metrics.items()}})

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "threshold_selection.csv", index=False)
    print("\nThreshold selection (chosen on validation, measured on test):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return chosen


def run_calibration(models, split, out_dir: Path) -> dict:
    """Fit calibrators on validation, score them on test.

    `class_weight="balanced"` deliberately shifts the predicted probabilities
    toward the positive class, so the raw outputs are not population risks.
    Sigmoid is strictly increasing, so it preserves rank order exactly and
    leaves ROC-AUC and PR-AUC untouched. Isotonic is only weakly monotonic: it
    can map distinct scores onto the same calibrated value, and the ties
    it creates can move rank-sensitive metrics by a small but non-negligible
    amount (observed: up to ~0.006 PR-AUC on test). Either way, calibration
    changes what the numbers mean, and therefore where a sensible threshold
    sits.

    FrozenEstimator is how a pre-fitted model is calibrated in scikit-learn 1.9;
    the older cv="prefit" argument is deprecated.

    Skipped, with a message, if the validation split holds only one class,
    there is nothing to calibrate against.
    """
    if len(np.unique(split.y_val)) < 2:
        print("\nskipping calibration: validation split holds only one class")
        return {}

    rows = []
    summary = {}
    for name, pipe in models.items():
        variants = {"raw": pipe}
        for method in ("sigmoid", "isotonic"):
            calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method=method)
            calibrated.fit(split.X_val, split.y_val)
            variants[method] = calibrated

        briers = {}
        for variant, est in variants.items():
            prob = est.predict_proba(split.X_test)[:, 1]
            briers[variant] = float(brier_score_loss(split.y_test, prob))
            # Recorded per variant (repeated across that variant's bin rows) so
            # the "isotonic moves rank-sensitive metrics" claim above, and the
            # design doc's ad hoc "~0.006 PR-AUC" figure, can be checked against
            # this file instead of taken on faith.
            roc_auc = float(roc_auc_score(split.y_test, prob))
            pr_auc = float(average_precision_score(split.y_test, prob))
            frac_pos, mean_pred = calibration_curve(split.y_test, prob, n_bins=10,
                                                    strategy="quantile")
            for bin_idx, (mp, fp) in enumerate(zip(mean_pred, frac_pos)):
                rows.append({"model": name, "variant": variant, "bin": bin_idx,
                             "mean_predicted": float(mp), "observed_frequency": float(fp),
                             "roc_auc": roc_auc, "pr_auc": pr_auc})

        best = min(briers, key=briers.get)
        summary[name] = {**briers, "best": best}
        print(f"\n{name} Brier — " + " · ".join(f"{k} {v:.4f}" for k, v in briers.items())
              + f"  → best: {best}")

    pd.DataFrame(rows).to_csv(out_dir / "calibration.csv", index=False)
    return summary


def bootstrap_pr_auc_gap(y_true, prob_a, prob_b, n_boot: int = 1000,
                         seed: int = 42) -> tuple[float, float, float]:
    """Resample the evaluation set and measure the PR-AUC gap each time.

    Returns (mean gap, 2.5th percentile, 97.5th percentile). A positive gap means
    `prob_a` scores higher. An interval spanning zero means the data does not
    support ranking one model above the other.

    Resamples that come back single-class are skipped: average_precision_score is
    undefined there, and including them would quietly bias the interval.
    """
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        gaps.append(average_precision_score(y_boot, prob_a[idx])
                    - average_precision_score(y_boot, prob_b[idx]))

    if len(gaps) < max(20, n_boot // 10):
        raise ValueError(
            f"only {len(gaps)} of {n_boot} resamples were usable; "
            "the evaluation set holds too few positives for a stable interval"
        )

    gaps = np.asarray(gaps)
    return (float(gaps.mean()),
            float(np.percentile(gaps, 2.5)),
            float(np.percentile(gaps, 97.5)))


Split = namedtuple("Split", "X_train X_val X_test y_train y_val y_test")


def load_and_split(data_dir: Path) -> Split:
    """60/40 then 50/50, giving a stratified 60/20/20.

    Validation makes every decision: the threshold, the calibrator. Test
    produces every reported number and has no say in any choice. Choosing a
    threshold and reporting its performance on the same rows would be the same
    error as tuning on the test set.
    """
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise SystemExit(f"train.csv not found in {data_dir.resolve()}")

    raw = pd.read_csv(train_path)
    X = build_features(raw)
    y = raw[TARGET].astype(int)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)

    return Split(X_train, X_val, X_test, y_train, y_val, y_test)


def fit_models(X_train, y_train) -> dict[str, Pipeline]:
    """The same two configurations the pipeline cross-validates."""
    models = {
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
    for name, pipe in models.items():
        print(f"  fitting {name}...")
        pipe.fit(X_train, y_train)
    return models


def grouped_permutation_drop(model, X, y, columns, n_repeats: int = 5,
                             seed: int = RANDOM_STATE) -> float:
    """Mean drop in PR-AUC when a set of columns is shuffled together.

    Shuffling one column at a time is what understates a feature with a
    correlated twin: the model reads the same information off the survivor and
    barely loses anything. Shuffling the pair together removes that escape route.
    """
    rng = np.random.default_rng(seed)
    baseline = average_precision_score(y, model.predict_proba(X)[:, 1])

    drops = []
    for _ in range(n_repeats):
        shuffled = X.copy()
        order = rng.permutation(len(shuffled))
        for col in columns:
            shuffled[col] = shuffled[col].to_numpy()[order]
        drops.append(baseline - average_precision_score(
            y, model.predict_proba(shuffled)[:, 1]))
    return float(np.mean(drops))


def run_feature_redundancy(models, split, out_dir: Path) -> dict:
    """Measure the age / is_student claim instead of asserting it.

    If the two are twins, each alone drops little while the pair together drops
    a lot, so `joint - (age + is_student)` is clearly positive. The refit is the
    harder check: drop `age` entirely and see whether PR-AUC survives.
    """
    name = "HistGradientBoosting"
    model = models[name]

    age_only = grouped_permutation_drop(model, split.X_test, split.y_test, ["age"])
    student_only = grouped_permutation_drop(model, split.X_test, split.y_test, ["is_student"])
    joint = grouped_permutation_drop(model, split.X_test, split.y_test, ["age", "is_student"])

    with_age = average_precision_score(
        split.y_test, model.predict_proba(split.X_test)[:, 1])

    no_age_train = split.X_train.drop(columns=["age"])
    no_age_test = split.X_test.drop(columns=["age"])

    # make_preprocessor() reads the module-level NUMERIC list at call time, and
    # it still names "age" even though age has been dropped from the frame here.
    # Rebinding dp.NUMERIC to a new list (rather than mutating the existing list
    # in place, e.g. with .remove()) is what keeps this safe: the ColumnTransformers
    # already built for `models` hold their own closed-over reference to the old
    # list object, so they are unaffected, and the `finally` below only has to
    # restore the module attribute, not undo an in-place mutation.
    import depression_pipeline as dp
    original = dp.NUMERIC
    dp.NUMERIC = [c for c in original if c != "age"]
    try:
        refit = fit_models(no_age_train, split.y_train)[name]
        without_age = average_precision_score(
            split.y_test, refit.predict_proba(no_age_test)[:, 1])
    finally:
        dp.NUMERIC = original

    result = {
        "individual_age": age_only,
        "individual_is_student": student_only,
        "joint": joint,
        "excess": joint - (age_only + student_only),
        "pr_auc_with_age": float(with_age),
        "pr_auc_without_age": float(without_age),
    }
    pd.DataFrame([result]).to_csv(out_dir / "feature_redundancy.csv", index=False)

    print(f"\nage alone {age_only:.4f} · is_student alone {student_only:.4f} "
          f"· together {joint:.4f} · excess {result['excess']:+.4f}")
    print(f"PR-AUC with age {with_age:.4f} → without age {without_age:.4f}")
    return result


def run_significance(models, split, n_boot: int, out_dir: Path) -> dict:
    """Is the gap between the two models larger than resampling noise?

    The cross-validated gap is 0.9068 against 0.9034. If this interval spans
    zero, the honest conclusion is that the data does not support ranking one
    model above the other.
    """
    hgb = models["HistGradientBoosting"].predict_proba(split.X_test)[:, 1]
    logreg = models["Logistic Regression"].predict_proba(split.X_test)[:, 1]

    mean_gap, lower, upper = bootstrap_pr_auc_gap(
        split.y_test, hgb, logreg, n_boot=n_boot, seed=RANDOM_STATE)

    # mean_gap is a difference; record the two PR-AUCs it is a difference of so
    # the gap can be checked against its own parts instead of taken on faith.
    pr_auc = {"HistGradientBoosting": float(average_precision_score(split.y_test, hgb)),
              "Logistic Regression": float(average_precision_score(split.y_test, logreg))}

    result = {"mean_gap": mean_gap, "lower": lower, "upper": upper,
              "contains_zero": bool(lower <= 0.0 <= upper), "n_boot": n_boot,
              "gap_direction": "HistGradientBoosting - Logistic Regression",
              "pr_auc": pr_auc, "status": "ok"}
    (out_dir / "model_significance.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    verdict = ("the interval spans zero — the data does not support ranking them"
               if result["contains_zero"]
               else "the interval excludes zero — the gap is real")
    print(f"\nPR-AUC gap (HistGB − LogReg): {mean_gap:+.4f} "
          f"[{lower:+.4f}, {upper:+.4f}] · {verdict}")
    return result


def false_negative_profile(X, y_true, y_pred) -> pd.DataFrame:
    """Compare the positives the model missed against the ones it caught."""
    y_true = np.asarray(y_true)
    missed = (y_true == 1) & (y_pred == 0)
    caught = (y_true == 1) & (y_pred == 1)

    if not missed.any():
        raise ValueError("No missed positives in this group — cannot profile an empty set")
    if not caught.any():
        raise ValueError("No caught positives in this group — cannot profile an empty set")

    numeric = X.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        m = float(numeric.loc[missed, col].mean())
        c = float(numeric.loc[caught, col].mean())
        # .mean() silently skips NaN, so a column with a structural null rate
        # (cgpa, degree_level, sleep_hours) compares near-disjoint subpopulations
        # rather than like-for-like groups. Recording how many non-null values
        # each mean is actually computed over makes that visible instead of hidden.
        missed_n = int(numeric.loc[missed, col].count())
        caught_n = int(numeric.loc[caught, col].count())
        rows.append({"feature": col, "missed_mean": m, "caught_mean": c,
                     "difference": m - c, "missed_n": missed_n, "caught_n": caught_n})
    return pd.DataFrame(rows).sort_values("difference", key=abs, ascending=False)


def run_false_negatives(models, split, chosen, out_dir: Path) -> dict:
    """Who the model fails, at the threshold actually chosen.

    For a screening model this is the question with the most at stake: a missed
    case is the expensive error, and nothing else in the project looks at one.
    """
    name = "HistGradientBoosting"
    prob = models[name].predict_proba(split.X_test)[:, 1]
    pred = (prob >= chosen[name]).astype(int)

    profile = false_negative_profile(split.X_test, split.y_test, pred)
    profile.to_csv(out_dir / "false_negatives.csv", index=False)

    y_test = np.asarray(split.y_test)
    n_missed = int(((y_test == 1) & (pred == 0)).sum())
    n_positive = int((y_test == 1).sum())

    print(f"\nMissed {n_missed} of {n_positive} positives "
          f"({n_missed / n_positive:.1%}) at threshold {chosen[name]:.3f}")
    print(profile.head(6).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return {"missed": n_missed, "positives": n_positive,
            "miss_rate": n_missed / n_positive, "threshold": chosen[name]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = data_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    split = load_and_split(data_dir)
    print(f"train {len(split.y_train):,} · val {len(split.y_val):,} · test {len(split.y_test):,}")
    print(f"positive rate — train {split.y_train.mean():.4f} · "
          f"val {split.y_val.mean():.4f} · test {split.y_test.mean():.4f}")

    models = fit_models(split.X_train, split.y_train)
    print(f"fitted {len(models)} models")

    chosen = run_threshold_selection(models, split, out_dir)

    calibration = run_calibration(models, split, out_dir)

    redundancy = run_feature_redundancy(models, split, out_dir)

    false_negatives = run_false_negatives(models, split, chosen, out_dir)

    significance = None
    try:
        significance = run_significance(models, split, args.bootstrap, out_dir)
    except ValueError as e:
        print(f"\nskipping significance: {e}")
        # Written on both paths so a stale success file from a previous run can
        # never survive a skip: without this, model_significance.json could
        # keep reporting a confident interval while evaluation_summary.json's
        # model_significance is null, and the two would contradict each other.
        (out_dir / "model_significance.json").write_text(
            json.dumps({"status": "skipped", "reason": str(e)}, indent=2) + "\n",
            encoding="utf-8")

    summary = {
        "split": {"train": len(split.y_train), "val": len(split.y_val),
                  "test": len(split.y_test),
                  "positive_rate": {"train": float(split.y_train.mean()),
                                    "val": float(split.y_val.mean()),
                                    "test": float(split.y_test.mean())}},
        "thresholds_f2": chosen,
        "calibration": calibration,
        "feature_redundancy": redundancy,
        "false_negatives": false_negatives,
        "model_significance": significance,
    }
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nOutputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
