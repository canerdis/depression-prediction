# Evaluation layer — design

**Date:** 2026-07-29
**Status:** approved for planning

## Purpose

`depression_pipeline.py` builds and cross-validates the models. `depression_tuning.py`
answers whether searching hyperparameters helps. Neither answers the questions that
follow once a model exists and someone has to defend it:

1. What decision threshold is used, and why?
2. Are the predicted probabilities calibrated?
3. `age` dominates permutation importance and `is_student` scores near zero. Is that
   real, or are they the same signal twice?
4. Which positives does the model miss?
5. Is the gap between the two models larger than noise?

This adds a third script that answers all five. It does not change the models or the
figures already reported.

## Goals

- Every number that gets quoted comes from data that had no part in producing it.
- Each answer is an artefact on disk, not a claim in prose.
- Reuses the existing feature definition so the evaluation cannot drift from the model.

## Non-goals

- Raising PR-AUC. The tuning study already showed a flat response surface; this is
  about defensibility, not performance.
- Changing `depression_pipeline.py` or `depression_tuning.py`.
- Retrofitting the original Ulm group notebook. That was submitted coursework; its
  weaknesses are documented in the write-up rather than rewritten.
- Any plotting. This script writes numbers; the portfolio renders them separately.

## The split

A single stratified three-way split, seeded from `RANDOM_STATE`:

| Split | Share | Used for | Never used for |
|---|---|---|---|
| Train | 60% | fitting | anything reported |
| Validation | 20% | choosing the threshold, fitting the calibrator | reporting |
| Test | 20% | every reported number | any decision |

Choosing a threshold and reporting its performance on the same rows is the same error
as tuning on the test set. The third split is what makes the threshold answer hold up.

**Expected consequence:** these numbers will sit slightly below the 5-fold figures in
`model_comparison.csv`, because the model trains on 60% rather than 80% and the result
is a single split rather than an average. The write-up must say so, or the two tables
will look like they contradict each other.

## Components

### 1. Threshold selection

Sweep thresholds over the validation scores and evaluate three rules:

| Rule | Rationale |
|---|---|
| Maximum F1 | the neutral default |
| **Maximum F2** | **the chosen rule** — weights recall twice precision |
| Highest recall at precision ≥ 0.5 | a constraint-shaped alternative |

F2 is the default because a missed depression case costs more than a false alarm.
All three are reported so the choice is visible rather than buried, along with the
metrics the untouched **0.5 default** would have produced — quantifying what the
default was costing is the strongest form of this answer.

The chosen threshold is then applied **once** to the test split, and those are the
numbers quoted.

### 2. Calibration

Reliability curve (10 bins) and Brier score on test, for the raw model and for
`CalibratedClassifierCV` fitted on validation. Both sigmoid and isotonic are tried;
the one with the better test Brier score is reported as the recommendation.

Two points the write-up must make, because they are the interesting part:

- `class_weight="balanced"` deliberately shifts predicted probabilities toward the
  positive class. The raw outputs are therefore **not** population risks, and reading
  them as such would be wrong.
- Sigmoid calibration is strictly increasing, so it preserves rank order exactly and
  leaves ROC-AUC and PR-AUC untouched. Isotonic is only weakly **monotonic** — it can
  map distinct scores onto the same calibrated value — so the ties it creates can move
  rank-sensitive metrics by a small but non-negligible amount (observed: up to ~0.006
  PR-AUC on test). Either way, calibration changes what the numbers mean, and therefore
  where a sensible threshold sits — not the ranking that produced them.

### 3. Feature redundancy: `age` and `is_student`

The README currently asserts that permutation importance understates a feature with a
correlated twin. This measures it.

- Permute `age` alone → drop A
- Permute `is_student` alone → drop B
- Permute **both together** → drop AB

If the two are redundant, A and B are each small (the survivor carries the signal) while
AB is large. The quantity reported is `AB − (A + B)`; a clearly positive value is the
evidence.

Then the harder check: **refit without `age`** and compare test PR-AUC against the full
model. If the score holds, the redundancy is demonstrated rather than argued.

### 4. False-negative profile

At the chosen threshold, on test, compare missed positives against caught positives
across every feature: rate of `suicidal_thoughts`, mean `age`, share `is_student`,
mean `pressure`, and so on. One row per feature, both group means, and the difference.

This is the question with the most at stake in a mental-health model and the one
nothing in the project currently touches.

### 5. Is the model gap real

Bootstrap the test split 1,000 times with replacement. On each resample compute PR-AUC
for Logistic Regression and for HistGradientBoosting, and record the difference. Report
the mean gap and the 2.5th/97.5th percentiles.

The cross-validated gap is 0.9068 against 0.9034. **The interval is expected to contain
zero.** If it does, the honest conclusion is that the data does not support ranking one
model above the other, and that conclusion is worth more than the 0.003.

## Code structure

Four pure functions carry the logic that could be silently wrong; a thin `main()`
orchestrates and writes files.

| Unit | Contract | Depends on |
|---|---|---|
| `choose_threshold(y_true, y_prob, rule) -> (float, dict)` | threshold and its metrics for one rule | numpy |
| `bootstrap_pr_auc_gap(y_true, prob_a, prob_b, n_boot, seed) -> (float, float, float)` | mean gap, lower and upper bound | numpy, sklearn |
| `grouped_permutation_drop(model, X, y, columns, n_repeats, seed) -> float` | score drop when a set of columns is permuted together | sklearn |
| `false_negative_profile(X, y_true, y_pred) -> DataFrame` | per-feature comparison of missed vs caught | pandas |

`main()` performs no arithmetic worth testing.

Feature construction is imported from `depression_pipeline` (`build_features`,
`make_preprocessor`, `RANDOM_STATE`, `TARGET`) — the same import that
`depression_tuning.py` already uses, so there remains one definition of the feature
table.

## Outputs

Written to `outputs/`, matching the existing convention:

| File | Contents |
|---|---|
| `threshold_selection.csv` | one row per rule: threshold, validation metrics, test metrics, plus the 0.5 default |
| `calibration.csv` | bin, mean predicted probability, observed frequency, plus ROC-AUC and PR-AUC per variant, for raw and calibrated |
| `feature_redundancy.csv` | individual and joint permutation drops, and the refit-without-`age` comparison |
| `false_negatives.csv` | per-feature missed-vs-caught profile, with non-null counts for both groups |
| `model_significance.json` | bootstrap mean gap and 95% interval, plus each model's test PR-AUC |
| `evaluation_summary.json` | the headline numbers from all five |

## Error handling

- Missing `train.csv` exits with the resolved directory it searched, matching
  `depression_pipeline.py`.
- All randomness is seeded from `RANDOM_STATE`; repeated runs reproduce exactly.
- If the test split holds too few positives for a stable bootstrap, the script says so
  and skips component 5 rather than returning a confident interval built on nothing.
- Calibration is skipped with a message if validation holds only one class.

## Tests

The project currently has no tests. One file, `test_evaluate.py`, covering only the two
functions that can return a **plausible wrong number** rather than crash:

- `choose_threshold` — synthetic scores where the correct threshold is known by
  construction, plus the degenerate case where predictions are all one class.
- `bootstrap_pr_auc_gap` — two identical score vectors must yield an interval containing
  zero; a clearly superior vector must yield one excluding zero.

Four tests, no pytest configuration, no CI wiring. `grouped_permutation_drop` and
`false_negative_profile` fail loudly if they fail at all, so they are not covered.

## What would change the conclusions

Stated in advance so the result is not rationalised afterwards:

- If the bootstrap interval **excludes** zero, HistGradientBoosting genuinely is the
  better model and the write-up should say so plainly.
- If refitting without `age` **drops** PR-AUC materially, then `age` is not merely a
  proxy for cohort and the README's twin explanation is wrong and must be corrected.
- If calibration **fails to improve** the Brier score, the raw probabilities were
  already usable and the recommendation should say so rather than add a step for form's
  sake.
