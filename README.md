# Depression Prediction — Cohort-Aware Pipeline

Rebuild of the Machine Learning Lab group project for the current dataset. The
data underneath this project changed, and not cosmetically. The original
notebook cannot be pointed at the new files at all.

**[Write-up →](https://canerdis.github.io/data-science/depression-prediction/)**

## What changed in the data

| | Old `student_depression_dataset.csv` | New `train.csv` |
|---|---|---|
| Rows | 27,901 | 140,700 |
| Population | Students only | Students *and* working professionals |
| Depression rate | 58.6% | 18.2% |
| Missing values | None | 80% on the academic columns (structural) |
| `Sleep Duration` values | 5 | 36 |
| `Degree` values | 28 | 115 |

The old file turns out to be the student subset of the new one. The student
depression rate in `train.csv` is 0.5855, matching the old file's rate exactly.
A held-out `test.csv` (93,800 rows) carries no target column, so evaluation has
to come from cross-validation on train rather than a test split.

### The nulls are structural, not missing

Students answer the academic block; working professionals answer the work
block. The two never overlap:

| Column pair | Both populated | Neither | Exactly one |
|---|---|---|---|
| Academic Pressure / Work Pressure | 0 | 21 | 140,679 |
| Study Satisfaction / Job Satisfaction | 2 | 15 | 140,683 |

Both pairs are measured on the same 1–5 scale, so each pair collapses into a
single feature with no information loss. A column that was 80% "missing"
becomes one that is fully populated.

## Why the old notebook cannot run on this data

Four failures, each verified against the file:

1. **`dropna()` returns 0 of 140,700 rows.** Every row is null on either the
   academic block or the work block, so dropping incomplete rows drops the
   dataset.
2. **The sleep-duration lookup maps 100% of rows to NaN.** The old file quoted
   these values (`"'5-6 hours'"`); this one does not, and adds 32 further
   variants. A four-key dictionary matches nothing.
3. **Three columns are dropped that now carry the signal.** `Profession`,
   `Work Pressure` and `Job Satisfaction` were dropped for sharing one value
   across 99%+ of rows. Correct for a students-only file. Here, 112,799 working
   professionals have real values in them.
4. **Balancing by undersampling would discard 89,566 rows, 63.7% of the data.**
   The old class split was 58/42, so undersampling cost little. This one is
   82/18.

A fifth issue is latent. Rows are removed with
`pd.concat([df, sample]).drop_duplicates(keep=False)`, which also deletes
*naturally* duplicated rows. Both files happen to have none once `id` is
included, so nothing was lost, but nobody checked.

## Why accuracy is the wrong headline metric now

At an 82/18 split, **predicting "not depressed" for everyone scores 0.8183
accuracy.** The original notebook reported 81.67% – 83.80% across four models.
Honest on balanced data; here, indistinguishable from a model that learned
nothing. So the baseline is a row in the table rather than an assumption:

| Model | ROC-AUC | PR-AUC | Balanced acc. | F1 | Accuracy |
|---|---|---|---|---|---|
| HistGradientBoosting | **0.9747** | **0.9068** | **0.9214** | **0.8044** | 0.9181 |
| Logistic Regression | 0.9740 | 0.9034 | 0.9194 | 0.7976 | 0.9145 |
| Majority baseline | 0.5000 | 0.1817 | 0.5000 | 0.0000 | 0.8183 |

5-fold stratified cross-validation on all 140,700 rows. The baseline row is the
point: 0.8183 accuracy alongside 0.5 ROC-AUC and 0.0 F1. Both real models land
within 0.004 ROC-AUC of each other, so the ceiling is set by the data rather
than the algorithm.

## Automated hyperparameter search

`depression_tuning.py` replaces the hand-picked parameters above with grid,
randomized and Bayesian search, the same set used in the Optimisation Techniques
project, and measures whether searching helps. A 20% holdout is split off before
any search runs and is never seen by one, so each strategy can report its inner
CV score, its holdout score, and the gap between them.

| Model | Strategy | Fits | CV PR-AUC | Holdout PR-AUC | Candidate spread | Time |
|---|---|---|---|---|---|---|
| HistGradientBoosting | BayesSearchCV | 36 | 0.9062 | **0.9064** | 0.0046 | 124s |
| HistGradientBoosting | RandomizedSearchCV | 36 | 0.9067 | 0.9063 | 0.0040 | 68s |
| HistGradientBoosting | *hand-picked* | 1 | — | 0.9063 | — | 9s |
| Logistic Regression | *hand-picked* | 1 | — | 0.9022 | — | 0.6s |
| Logistic Regression | GridSearchCV | 30 | 0.9035 | 0.9022 | 0.0020 | 20s |

**Searching did not beat the hand-picked configuration.** It bought +0.0001
PR-AUC for 102 extra model fits and roughly 30× the compute.

The candidate spread is why that is a result rather than a failure. Across 12
configurations drawn from a wide six-dimensional space, the *worst* scored
0.9026 and the best 0.9067. When every point performs within half a percentage
point of every other, the response surface is flat and more budget finds
nothing. Had the spread been wide, 12 candidates would have been the binding
constraint and the honest conclusion would have been "search harder."

Two secondary observations. Tuning optimism is negligible here (+0.0013,
+0.0003, −0.0002): with 112,560 rows in the search pool, a 3-fold estimate is
stable enough that taking the maximum over a dozen candidates barely overfits
it. And Bayesian search cost 1.8× the wall time of random search for no gain,
because it is sequential and cannot parallelise across candidates.

Effort on this problem belongs in features and data quality, not the optimiser.
The earlier optimisation project reached the same conclusion when four model
families landed within 0.5 percentage points of each other.

```bash
python depression_tuning.py --data-dir . --budget 12 --folds 3
python depression_tuning.py --nested          # nested CV: slower, no holdout needed
```

## What predicts depression

Permutation importance (mean drop in ROC-AUC, 5 repeats):

| Feature | Importance |
|---|---|
| age | 0.1425 |
| suicidal_thoughts | 0.0270 |
| pressure | 0.0169 |
| financial_stress | 0.0099 |
| satisfaction | 0.0059 |

Read `age` carefully. It does two jobs at once. Depression falls monotonically
from 64.3% in the 18–22 band to 0.8% in the 45–60 band, and `age < 30` also
predicts student status with 84.5% accuracy. `is_student` scores only 0.0026,
not because the cohort is irrelevant, but because permuting it leaves `age`
carrying the same information.

That much holds up. The claim that the two are *twins* does not, and the
evaluation layer below tested it. Permuting `age` alone costs 0.3593 PR-AUC;
`is_student` alone costs 0.0126; together they cost 0.4181, an excess of only
+0.0462 over the sum. Real twins would each be cheap to permute alone, because
the survivor carries the signal for both. `is_student` fits that; `age` does
not, so the redundancy runs one way.

Refitting without `age` altogether moves PR-AUC from 0.9040 to 0.8570, a loss
of 0.0470 and 7.6× smaller than permuting it. No single feature substitutes for
`age`, but the rest of the feature set collectively recovers most of what it
carried. Generalising from this pair would be a mistake: permutation importance
understated one member and not the other.

`suicidal_thoughts` remains the strongest non-demographic predictor. As the
original notebook argued, it is arguably a symptom rather than a cause, and it
is a PHQ-9 diagnostic criterion. That reasoning still holds.

## Evaluation layer

The numbers above answer whether the models work. `evaluate.py` asks what
follows once someone has to defend one: what threshold it runs at, whether its
probabilities mean anything, who it misses, and whether the gap between the two
models is more than noise. It re-fits both on a fresh stratified 60/20/20 split
(train 84,420, validation 28,140, test 28,140, positive rate 18.17% in all
three). Validation makes every decision; test reports every number.

**The 0.5 default costs recall.** Maximising F2 on validation, which weights
recall twice precision because a missed case costs more than a false alarm,
picks 0.4638 for Logistic Regression and 0.4918 for HistGradientBoosting. On
test that moves Logistic Regression from 0.9247 to 0.9319 recall against a
precision cost of 0.6954 to 0.6801, and HistGradientBoosting from 0.9222 to
0.9243, precision 0.7103 to 0.7080. Modest, but it is the quantified price of
leaving `.predict()` alone.

The raw scores are not probabilities. `class_weight="balanced"` shifts them
toward the positive class on purpose. Fitting a calibrator on validation and
scoring on test cuts Brier loss by roughly a quarter: Logistic Regression 0.0644
to 0.0469 with sigmoid, HistGradientBoosting 0.0611 to 0.0461 with isotonic.
Sigmoid is strictly increasing, so it rescales without touching ROC-AUC or
PR-AUC. Isotonic is only weakly monotonic and can tie distinct scores together,
so rank metrics are worth re-checking after it.

**387 of 5,113 test positives are missed** at HistGradientBoosting's chosen
threshold, a 7.6% miss rate. The missed group averages 35.0 years old against
24.8 for the group the model catches, and reports fewer work/study hours, 6.66
against 7.85. That is a property of this model on this synthetic dataset, not a
finding about who is actually at risk, and it licenses no claim about screening.

The gap between the two models is real, and small. Bootstrapping the test split
1,000 times gives a mean PR-AUC gap of +0.00409 for HistGradientBoosting over
Logistic Regression, 95% interval [+0.00204, +0.00625]. The design for this
layer expected that interval to contain zero, though with a method that holds
both models fixed and resamples 28,140 rows, excluding a gap this size was close
to a foregone conclusion. Two limits. The margin is 0.004 PR-AUC, the same size
the cross-validated table already showed, and resolving a difference is not the
same as it mattering. And the bootstrap sees evaluation-set noise only: under a
different partition the models themselves would differ and the order could flip.

These figures sit slightly below the 5-fold table above, 0.9040 PR-AUC against
0.9068, because this layer trains on 60% of the data instead of 80% and reports
one split rather than an average.

```bash
python evaluate.py --data-dir . --bootstrap 1000
```

Writes `outputs/threshold_selection.csv`, `outputs/calibration.csv`,
`outputs/feature_redundancy.csv`, `outputs/false_negatives.csv`,
`outputs/model_significance.json`, and `outputs/evaluation_summary.json`.

## Design decisions

- Every row is kept. Class imbalance is handled with `class_weight="balanced"`,
  not by deleting majority-class rows.
- All transforms live inside a `Pipeline`. Imputation and scaling are fitted per
  fold, so nothing leaks from validation into training.
- Rare categories fold automatically. `City` and `Profession` go through
  `OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist")`, so
  junk values like `Kalyan` in the Degree column cannot break a run. The old
  approach hand-listed 22 bad values it had found by eye.
- `Name` is not a feature. It is an identifier with 422 values.
- `Sleep Duration` is parsed rather than looked up, so both quoting styles and
  all 36 variants work.

## Data

`train.csv` and `test.csv` are not tracked here. They belong to the Kaggle
competition that published them, and the rows carry name, city and mental-health
fields that do not need a second public home. Download them from the competition
page and place them in this directory first. Everything in `outputs/` was
produced from them and is tracked, so the results can be inspected without
re-running anything.

## Running it

```bash
python depression_pipeline.py --data-dir . --folds 5 --submit
```

Writes `outputs/model_comparison.csv`, `outputs/feature_importance.csv`,
`outputs/summary.json`, and with `--submit` a `submission.csv` of predictions
for `test.csv`.

## Limitations

- `test.csv` has no labels, so every number here is cross-validated on train.
  No score against the competition's held-out set is claimed.
- `CGPA` exists only for students and is left missing for professionals rather
  than imputed across cohorts, which would invent a grade point average for
  people who have none.
- This is a synthetic Kaggle dataset generated from a real survey. Relationships
  in it are not evidence about actual mental health.

## Files

| File | Description |
|---|---|
| `depression_pipeline.py` | Full pipeline: features, models, evaluation |
| `depression_tuning.py` | Grid / randomized / Bayesian search with holdout and nested modes |
| `Machine_Learning_Lab_Project_finished_version (1).ipynb` | Original group notebook (old dataset) |
| `outputs/` | Cross-validation results, tuning comparison, best params, feature importance, submission |
