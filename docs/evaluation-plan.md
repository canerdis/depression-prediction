# Evaluation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `evaluate.py`, which answers the five questions `depression_pipeline.py` and `depression_tuning.py` leave open — which threshold, whether probabilities are calibrated, whether `age` and `is_student` are one signal, which positives are missed, and whether the model gap is larger than noise.

**Architecture:** One new script plus one test file. Four pure functions carry the logic that could be silently wrong; a thin `main()` splits the data three ways, fits the two models, calls the four, and writes six files to `outputs/`. Feature construction is imported from `depression_pipeline` so there remains one definition of the feature table.

**Tech Stack:** Python 3.12.3, scikit-learn 1.9.0, pandas 2.2.2, numpy 2.1.0, pytest (to be installed).

**Spec:** `docs/evaluation-design.md`

## Global Constraints

- **Nothing in `depression_pipeline.py` or `depression_tuning.py` changes.** `evaluate.py` imports from the pipeline; it never edits it.
- **No model is retrained to improve a score.** This layer explains the existing models. The one refit that happens (Task 6, without `age`) exists to test a claim, not to win.
- **Split discipline:** train 60% fits, validation 20% makes every decision, test 20% produces every reported number. A number computed on validation is never quoted as a result.
- **Seed everything from `RANDOM_STATE`** (imported from `depression_pipeline`, value 42). Reruns must reproduce exactly.
- **`precision_recall_curve` returns `thresholds` one element shorter than `precision` and `recall`.** Align by dropping the final precision/recall point before indexing. Getting this wrong shifts the chosen threshold silently.
- **Calibrating a pre-fitted model uses `sklearn.frozen.FrozenEstimator`**, not `cv="prefit"`, which is deprecated in this version.
- Outputs go to `outputs/`, matching the existing convention.

## File Structure

| File | Responsibility |
|---|---|
| `evaluate.py` (create) | Four pure functions plus `main()` orchestration |
| `test_evaluate.py` (create) | Covers the two functions that can return a plausible wrong number |
| `outputs/threshold_selection.csv` | One row per rule: threshold, validation metrics, test metrics |
| `outputs/calibration.csv` | Reliability bins for raw and calibrated probabilities |
| `outputs/feature_redundancy.csv` | Individual and joint permutation drops, plus refit-without-`age` |
| `outputs/false_negatives.csv` | Per-feature comparison of missed against caught positives |
| `outputs/model_significance.json` | Bootstrap mean gap and 95% interval |
| `outputs/evaluation_summary.json` | Headline numbers from all five components |

---

### Task 1: `choose_threshold`

**Files:**
- Create: `test_evaluate.py`
- Create: `evaluate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `choose_threshold(y_true, y_prob, rule) -> tuple[float, dict]`. `rule` is one of `"f1"`, `"f2"`, `"recall_at_precision"`. The dict has keys `precision`, `recall`, `f1`, `f2`. Used by Task 4.

- [ ] **Step 1: Install pytest**

Run: `pip install pytest`
Expected: installs cleanly. It is not currently present in this environment.

- [ ] **Step 2: Write the failing test**

Create `test_evaluate.py`:

```python
import numpy as np
import pytest

from evaluate import choose_threshold


def test_f2_prefers_recall_over_precision():
    # Two candidate operating points. The lower threshold catches both positives
    # at the cost of one false positive; the higher one catches a single positive
    # cleanly. F2 weights recall, so it must pick the lower threshold.
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.45, 0.9])

    thr_f2, metrics_f2 = choose_threshold(y_true, y_prob, rule="f2")
    thr_f1, _ = choose_threshold(y_true, y_prob, rule="f1")

    assert thr_f2 <= thr_f1
    assert metrics_f2["recall"] == 1.0


def test_recall_at_precision_respects_the_floor():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.45, 0.9])

    _, metrics = choose_threshold(y_true, y_prob, rule="recall_at_precision")

    assert metrics["precision"] >= 0.5


def test_single_class_returns_default_threshold():
    # A degenerate split has no usable curve. Returning 0.5 with zeroed metrics
    # is what lets main() carry on instead of crashing mid-run.
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    thr, metrics = choose_threshold(y_true, y_prob, rule="f2")

    assert thr == 0.5
    assert metrics["recall"] == 0.0


def test_unknown_rule_is_rejected():
    y_true = np.array([0, 1])
    y_prob = np.array([0.2, 0.8])

    with pytest.raises(ValueError):
        choose_threshold(y_true, y_prob, rule="accuracy")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluate'`.

- [ ] **Step 4: Write the implementation**

Create `evaluate.py`:

```python
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

import sys

import numpy as np
from sklearn.metrics import precision_recall_curve

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
    `precision` and `recall` — the final point is the degenerate recall=0 corner
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest test_evaluate.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add evaluate.py test_evaluate.py
git commit -m "Add threshold selection with tests"
```

---

### Task 2: `bootstrap_pr_auc_gap`

**Files:**
- Modify: `evaluate.py`
- Modify: `test_evaluate.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `bootstrap_pr_auc_gap(y_true, prob_a, prob_b, n_boot, seed) -> tuple[float, float, float]` returning `(mean_gap, lower, upper)` where the bounds are the 2.5th and 97.5th percentiles. Positive means `prob_a` scores higher. Used by Task 8.

- [ ] **Step 1: Write the failing test**

Append to `test_evaluate.py`:

```python
from evaluate import bootstrap_pr_auc_gap


def _separable_case(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    strong = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    weak = rng.uniform(0.0, 1.0, n)
    return y, strong, weak


def test_identical_scores_give_an_interval_containing_zero():
    y, strong, _ = _separable_case()

    mean, lo, hi = bootstrap_pr_auc_gap(y, strong, strong, n_boot=200, seed=1)

    assert mean == pytest.approx(0.0, abs=1e-9)
    assert lo <= 0.0 <= hi


def test_a_clearly_better_model_gives_an_interval_above_zero():
    y, strong, weak = _separable_case()

    mean, lo, hi = bootstrap_pr_auc_gap(y, strong, weak, n_boot=200, seed=1)

    assert mean > 0.0
    assert lo > 0.0


def test_the_gap_is_reproducible_for_a_fixed_seed():
    y, strong, weak = _separable_case()

    first = bootstrap_pr_auc_gap(y, strong, weak, n_boot=100, seed=7)
    second = bootstrap_pr_auc_gap(y, strong, weak, n_boot=100, seed=7)

    assert first == second
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest test_evaluate.py -v`
Expected: FAIL — `ImportError: cannot import name 'bootstrap_pr_auc_gap'`.

- [ ] **Step 3: Write the implementation**

Add to `evaluate.py`, below `choose_threshold`:

```python
from sklearn.metrics import average_precision_score


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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest test_evaluate.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py test_evaluate.py
git commit -m "Add bootstrap comparison of the two models"
```

---

### Task 3: The split and the fitted models

**Files:**
- Modify: `evaluate.py`

**Interfaces:**
- Consumes: `build_features`, `RANDOM_STATE`, `TARGET` from `depression_pipeline`.
- Produces: `load_and_split(data_dir)` returning a `Split` namedtuple with fields `X_train, X_val, X_test, y_train, y_val, y_test`; and `fit_models(X_train, y_train)` returning `dict[str, Pipeline]` with keys `"Logistic Regression"` and `"HistGradientBoosting"`. Used by Tasks 4–8.

- [ ] **Step 1: Write the split and fitting helpers**

Add to `evaluate.py`:

```python
import argparse
import json
from collections import namedtuple
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from depression_pipeline import RANDOM_STATE, TARGET, build_features, make_preprocessor

Split = namedtuple("Split", "X_train X_val X_test y_train y_val y_test")


def load_and_split(data_dir: Path) -> Split:
    """60/40 then 50/50, giving a stratified 60/20/20.

    Validation makes every decision — the threshold, the calibrator. Test
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
```

- [ ] **Step 2: Add a minimal `main()` that exercises the split**

Add at the end of `evaluate.py`:

```python
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `python evaluate.py`
Expected: prints `train 84,420 · val 28,140 · test 28,140`, three positive rates all near 0.1817, then `fitted 2 models`. Takes roughly a minute.

- [ ] **Step 4: Confirm the tests still pass**

Run: `python -m pytest test_evaluate.py -v`
Expected: PASS, 7 tests. The new imports must not break them.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py
git commit -m "Add three-way split and model fitting"
```

---

### Task 4: Threshold selection report

**Files:**
- Modify: `evaluate.py`

**Interfaces:**
- Consumes: `choose_threshold` (Task 1), `Split` and `fit_models` (Task 3).
- Produces: `run_threshold_selection(models, split, out_dir) -> dict` returning the chosen threshold per model under the F2 rule, keyed by model name. Used by Tasks 7 and 9.

- [ ] **Step 1: Write the component**

Add to `evaluate.py`:

```python
from sklearn.metrics import precision_score, recall_score


def _metrics_at(y_true, y_prob, threshold: float) -> dict:
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
```

- [ ] **Step 2: Call it from `main()`**

In `main()`, after `models = fit_models(...)`, add:

```python
    chosen = run_threshold_selection(models, split, out_dir)
```

- [ ] **Step 3: Run it**

Run: `python evaluate.py`
Expected: an 8-row table, and `outputs/threshold_selection.csv` exists. The F2 threshold should sit below 0.5 for both models, and its `test_recall` should exceed the `default_0.5` row's recall — that difference is the finding.

- [ ] **Step 4: Commit**

```bash
git add evaluate.py outputs/threshold_selection.csv
git commit -m "Report threshold choice against the 0.5 default"
```

---

### Task 5: Calibration

**Files:**
- Modify: `evaluate.py`

**Interfaces:**
- Consumes: `Split` and `fit_models` (Task 3).
- Produces: `run_calibration(models, split, out_dir) -> dict` mapping model name to `{"raw": brier, "sigmoid": brier, "isotonic": brier, "best": name}`. Used by Task 9.

- [ ] **Step 1: Write the component**

Add to `evaluate.py`:

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss


def run_calibration(models, split, out_dir: Path) -> dict:
    """Fit calibrators on validation, score them on test.

    `class_weight="balanced"` deliberately shifts the predicted probabilities
    toward the positive class, so the raw outputs are not population risks.
    Both calibrators here are monotonic, so neither changes ROC-AUC or PR-AUC by
    a rounding error — they change what the numbers mean, and therefore where a
    sensible threshold sits.

    FrozenEstimator is how a pre-fitted model is calibrated in scikit-learn 1.9;
    the older cv="prefit" argument is deprecated.
    """
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
            frac_pos, mean_pred = calibration_curve(split.y_test, prob, n_bins=10,
                                                    strategy="quantile")
            for bin_idx, (mp, fp) in enumerate(zip(mean_pred, frac_pos)):
                rows.append({"model": name, "variant": variant, "bin": bin_idx,
                             "mean_predicted": float(mp), "observed_frequency": float(fp)})

        best = min(briers, key=briers.get)
        summary[name] = {**briers, "best": best}
        print(f"\n{name} Brier — " + " · ".join(f"{k} {v:.4f}" for k, v in briers.items())
              + f"  → best: {best}")

    pd.DataFrame(rows).to_csv(out_dir / "calibration.csv", index=False)
    return summary
```

- [ ] **Step 2: Call it from `main()`**

After the threshold call, add:

```python
    calibration = run_calibration(models, split, out_dir)
```

- [ ] **Step 3: Run it**

Run: `python evaluate.py`
Expected: Brier scores for raw, sigmoid and isotonic per model, and `outputs/calibration.csv` with 60 rows (2 models × 3 variants × 10 bins).

If `raw` wins for both models, that is a real result and the write-up should say the probabilities were already usable rather than adding a calibration step for form's sake.

- [ ] **Step 4: Commit**

```bash
git add evaluate.py outputs/calibration.csv
git commit -m "Check whether the predicted probabilities are calibrated"
```

---

### Task 6: Feature redundancy

**Files:**
- Modify: `evaluate.py`

**Interfaces:**
- Consumes: `Split` and `fit_models` (Task 3).
- Produces: `run_feature_redundancy(models, split, out_dir) -> dict` with keys `individual_age`, `individual_is_student`, `joint`, `excess`, `pr_auc_with_age`, `pr_auc_without_age`. Used by Task 9.
- Also produces the reusable `grouped_permutation_drop(model, X, y, columns, n_repeats, seed) -> float`.

- [ ] **Step 1: Write the component**

Add to `evaluate.py`:

```python
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
    refit = fit_models(no_age_train, split.y_train)[name]
    without_age = average_precision_score(
        split.y_test, refit.predict_proba(no_age_test)[:, 1])

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
```

**Note on the refit:** `fit_models` builds `make_preprocessor` against the module-level `NUMERIC` list, which still names `age`. Dropping the column from the frame means the `ColumnTransformer` will raise for a missing column. Before calling `fit_models` on the reduced frame, patch the list for the duration of the call:

```python
    import depression_pipeline as dp
    original = dp.NUMERIC
    dp.NUMERIC = [c for c in original if c != "age"]
    try:
        refit = fit_models(no_age_train, split.y_train)[name]
        without_age = average_precision_score(
            split.y_test, refit.predict_proba(no_age_test)[:, 1])
    finally:
        dp.NUMERIC = original
```

Use this form rather than the plain call shown above.

- [ ] **Step 2: Call it from `main()`**

```python
    redundancy = run_feature_redundancy(models, split, out_dir)
```

- [ ] **Step 3: Run it**

Run: `python evaluate.py`
Expected: `outputs/feature_redundancy.csv` exists. A clearly positive `excess` supports the twin claim. If `pr_auc_without_age` drops materially below `pr_auc_with_age`, the README's twin explanation is **wrong** and must be corrected — that outcome was recorded in the spec in advance.

- [ ] **Step 4: Commit**

```bash
git add evaluate.py outputs/feature_redundancy.csv
git commit -m "Test whether age and is_student carry the same signal"
```

---

### Task 7: False-negative profile

**Files:**
- Modify: `evaluate.py`

**Interfaces:**
- Consumes: `chosen` thresholds (Task 4), `Split` and `fit_models` (Task 3).
- Produces: `false_negative_profile(X, y_true, y_pred) -> pd.DataFrame` with columns `feature, missed_mean, caught_mean, difference`; and `run_false_negatives(models, split, chosen, out_dir) -> dict` with the missed count and rate.

- [ ] **Step 1: Write the component**

Add to `evaluate.py`:

```python
def false_negative_profile(X, y_true, y_pred) -> pd.DataFrame:
    """Compare the positives the model missed against the ones it caught."""
    y_true = np.asarray(y_true)
    missed = (y_true == 1) & (y_pred == 0)
    caught = (y_true == 1) & (y_pred == 1)

    numeric = X.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        m = float(numeric.loc[missed, col].mean())
        c = float(numeric.loc[caught, col].mean())
        rows.append({"feature": col, "missed_mean": m, "caught_mean": c,
                     "difference": m - c})
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
```

- [ ] **Step 2: Call it from `main()`**

```python
    false_negatives = run_false_negatives(models, split, chosen, out_dir)
```

- [ ] **Step 3: Run it**

Run: `python evaluate.py`
Expected: `outputs/false_negatives.csv` exists, and the printed miss rate is lower than it would be at 0.5, because the F2 threshold was chosen to favour recall.

- [ ] **Step 4: Commit**

```bash
git add evaluate.py outputs/false_negatives.csv
git commit -m "Profile the positives the model misses"
```

---

### Task 8: Model significance

**Files:**
- Modify: `evaluate.py`

**Interfaces:**
- Consumes: `bootstrap_pr_auc_gap` (Task 2), `Split` and `fit_models` (Task 3).
- Produces: `run_significance(models, split, n_boot, out_dir) -> dict` with keys `mean_gap`, `lower`, `upper`, `contains_zero`.

- [ ] **Step 1: Write the component**

Add to `evaluate.py`:

```python
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

    result = {"mean_gap": mean_gap, "lower": lower, "upper": upper,
              "contains_zero": bool(lower <= 0.0 <= upper), "n_boot": n_boot}
    (out_dir / "model_significance.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    verdict = ("the interval spans zero — the data does not support ranking them"
               if result["contains_zero"]
               else "the interval excludes zero — the gap is real")
    print(f"\nPR-AUC gap (HistGB − LogReg): {mean_gap:+.4f} "
          f"[{lower:+.4f}, {upper:+.4f}] · {verdict}")
    return result
```

- [ ] **Step 2: Call it from `main()`**

```python
    significance = run_significance(models, split, args.bootstrap, out_dir)
```

- [ ] **Step 3: Run it**

Run: `python evaluate.py --bootstrap 1000`
Expected: `outputs/model_significance.json` exists. The spec predicts the interval will contain zero. Either outcome is reportable; record which one happened.

- [ ] **Step 4: Commit**

```bash
git add evaluate.py outputs/model_significance.json
git commit -m "Bootstrap the gap between the two models"
```

---

### Task 9: Summary and README section

**Files:**
- Modify: `evaluate.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the return values of Tasks 4–8.
- Produces: `outputs/evaluation_summary.json` and a new README section.

- [ ] **Step 1: Write the summary**

At the end of `main()`, add:

```python
    summary = {
        "split": {"train": len(split.y_train), "val": len(split.y_val),
                  "test": len(split.y_test)},
        "thresholds_f2": chosen,
        "calibration": calibration,
        "feature_redundancy": redundancy,
        "false_negatives": false_negatives,
        "model_significance": significance,
    }
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nOutputs written to {out_dir.resolve()}")
```

- [ ] **Step 2: Run the whole script once more**

Run: `python evaluate.py --bootstrap 1000`
Expected: all six files present in `outputs/`. Confirm with `ls outputs/`.

- [ ] **Step 3: Confirm the tests still pass**

Run: `python -m pytest test_evaluate.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 4: Write the README section**

Add a section to `README.md` after "What predicts depression", using the **actual numbers produced by the run** — not the predictions in this plan. Cover: the threshold chosen and what 0.5 was costing; whether calibration helped; whether the age/is_student excess supports the twin claim and whether PR-AUC survived dropping `age`; the miss rate and who is missed; and whether the model gap interval spans zero.

If `pr_auc_without_age` came out materially lower, correct the existing twin explanation in the "What predicts depression" section rather than leaving both claims standing.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py README.md outputs/evaluation_summary.json
git commit -m "Summarise the evaluation findings"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Three-way 60/20/20 stratified split | 3 |
| Threshold selection, three rules plus the 0.5 default | 1, 4 |
| F2 as the chosen rule | 1, 4 |
| Calibration, sigmoid and isotonic, Brier on test | 5 |
| Monotonicity note (AUC unchanged) | 5, in the docstring |
| Joint permutation of age + is_student | 6 |
| Refit without `age` | 6 |
| False-negative profile at the chosen threshold | 7 |
| Bootstrap CI on the PR-AUC gap | 2, 8 |
| Six output files | 4, 5, 6, 7, 8, 9 |
| `choose_threshold` and `bootstrap_pr_auc_gap` tested | 1, 2 |
| Seeded from `RANDOM_STATE` | 3, 6, 8 |
| Pipeline files unchanged | enforced by the constraint; no task modifies them |
| Missing `train.csv` handled | 3 |
| Too few positives for bootstrap handled | 2, raises with a readable message |
| Single-class validation handled | 1, returns the 0.5 default |

**Placeholder scan:** no TBDs. Every code step carries the code. Task 9 Step 4 deliberately does not pre-write the README text, because it must quote numbers that do not exist until the run happens — the instruction names exactly which findings to cover.

**Type consistency:** `choose_threshold` returns `(float, dict)` in Task 1 and is consumed that way in Task 4. `bootstrap_pr_auc_gap` returns a 3-tuple in Task 2, unpacked as three values in Task 8. `Split` field names are used identically in Tasks 3–8. `fit_models` returns a dict keyed by the two model names used in Tasks 4–8. `grouped_permutation_drop` is defined and called only in Task 6.

**Known trap, handled:** the `age` refit in Task 6 would fail because `make_preprocessor` reads the module-level `NUMERIC` list. Step 1 carries the patch-and-restore form rather than the naive call.
