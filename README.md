# Depression Prediction — Cohort-Aware Pipeline

Rebuild of the Machine Learning Lab group project for the **current** dataset.
The data underneath this project changed, and the change is not cosmetic — the
original notebook cannot be pointed at the new files.

## What changed in the data

| | Old `student_depression_dataset.csv` | New `train.csv` |
|---|---|---|
| Rows | 27,901 | 140,700 |
| Population | Students only | Students **and** working professionals |
| Depression rate | 58.6% | 18.2% |
| Missing values | None | 80% on the academic columns (structural) |
| `Sleep Duration` values | 5 | 36 |
| `Degree` values | 28 | 115 |

The old file turns out to be the **student subset** of the new one — the
student depression rate in `train.csv` is 0.5855, matching the old file's rate
exactly. A held-out `test.csv` (93,800 rows) carries no target column, so
evaluation has to come from cross-validation on train, not from a test split.

### The nulls are structural, not missing

Students answer the academic block; working professionals answer the work
block. The two never overlap:

| Column pair | Both populated | Neither | Exactly one |
|---|---|---|---|
| Academic Pressure / Work Pressure | 0 | 21 | 140,679 |
| Study Satisfaction / Job Satisfaction | 2 | 15 | 140,683 |

Both pairs are measured on the same 1–5 scale, so each pair collapses into a
single feature with no information loss — turning a column that is 80%
"missing" into one that is fully populated.

## Why the old notebook cannot run on this data

Four failures, each verified against the file:

1. **`dropna()` returns 0 of 140,700 rows.** Every row is null on either the
   academic block or the work block, so dropping incomplete rows drops
   everything. The old pipeline would hand an empty frame to the model.
2. **The sleep-duration lookup maps 100% of rows to NaN.** The old file stored
   these values with literal quote characters (`"'5-6 hours'"`); this one does
   not, and adds 32 further variants. A four-key dictionary matches nothing.
3. **Three columns are dropped that now carry the signal.** `Profession`,
   `Work Pressure` and `Job Satisfaction` were dropped because 99%+ of rows
   shared one value — correct for a students-only file, wrong here, where
   112,799 working professionals have real values in exactly those fields.
4. **Balancing by undersampling would discard 89,566 rows — 63.7% of the
   data.** The old class split was 58/42, so undersampling cost little. This
   one is 82/18.

A fifth issue is latent rather than live: rows are removed with
`pd.concat([df, sample]).drop_duplicates(keep=False)`, which also deletes any
*naturally* duplicated rows. Both files happen to have none once `id` is
included, so nothing was lost — but the correctness depends on a property of
the data that nobody checked.

## Why accuracy is the wrong headline metric now

At an 82/18 split, **predicting "not depressed" for every single person scores
0.8183 accuracy.** The original notebook reported 81.67% – 83.80% across four
models. Those numbers were honest on balanced data; carried over to this
dataset they would be indistinguishable from a model that has learned nothing.

The baseline is therefore reported as a row in the results table, not assumed:

| Model | ROC-AUC | PR-AUC | Balanced acc. | F1 | Accuracy |
|---|---|---|---|---|---|
| HistGradientBoosting | **0.9747** | **0.9068** | **0.9214** | **0.8044** | 0.9181 |
| Logistic Regression | 0.9740 | 0.9034 | 0.9194 | 0.7976 | 0.9145 |
| Majority baseline | 0.5000 | 0.1817 | 0.5000 | 0.0000 | 0.8183 |

5-fold stratified cross-validation on all 140,700 rows. The baseline row is the
point: 0.8183 accuracy alongside 0.5 ROC-AUC and 0.0 F1.

Both real models land within 0.004 ROC-AUC of each other. As in the earlier
optimisation project, the ceiling is set by the data rather than the algorithm.

## Automated hyperparameter search

The parameters above were chosen by hand, so `depression_tuning.py` replaces
that with three search strategies — the same set used in the Optimisation
Techniques project — and measures whether searching actually helps.

A 20% holdout is split off before any search runs and no search ever sees it.
Each strategy reports its best inner cross-validated score, its holdout score,
and the gap between them, because a tuned CV score always flatters itself.

| Model | Strategy | Fits | CV PR-AUC | Holdout PR-AUC | Candidate spread | Time |
|---|---|---|---|---|---|---|
| HistGradientBoosting | BayesSearchCV | 36 | 0.9062 | **0.9064** | 0.0046 | 124s |
| HistGradientBoosting | RandomizedSearchCV | 36 | 0.9067 | 0.9063 | 0.0040 | 68s |
| HistGradientBoosting | *hand-picked* | 1 | — | 0.9063 | — | 9s |
| Logistic Regression | *hand-picked* | 1 | — | 0.9022 | — | 0.6s |
| Logistic Regression | GridSearchCV | 30 | 0.9035 | 0.9022 | 0.0020 | 20s |

**Searching did not beat the hand-picked configuration** — +0.0001 PR-AUC for
102 extra model fits and roughly 30× the compute.

That is a result, not a failure, and the candidate spread is the evidence for
it. Across 12 configurations drawn from a wide six-dimensional space, the
*worst* one scored 0.9026 and the best 0.9067 — a spread of 0.004. When every
point in the space performs within half a percentage point of every other
point, the response surface is flat and no amount of extra budget will find
anything, because there is nothing to find. Had the spread been wide, a
12-candidate budget would have been the binding constraint and the honest
conclusion would have been "search harder."

Two secondary observations:

- **Tuning optimism is negligible here** (+0.0013, +0.0003, −0.0002). With
  112,560 rows in the search pool, a 3-fold CV estimate is stable enough that
  selecting the maximum over a dozen candidates barely overfits it. On a small
  dataset this gap is where over-optimistic results come from.
- **Bayesian search cost 1.8× the wall time of random search for no gain.**
  Bayesian optimisation is sequential — each candidate depends on the previous
  result — so it cannot parallelise across candidates the way random search
  can. It earns its keep when individual fits are expensive and the surface has
  real structure. Neither is true here.

The practical conclusion is that effort on this problem belongs in features and
data quality, not in the optimiser — the same conclusion the earlier
optimisation project reached when four model families landed within 0.5
percentage points of each other.

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

**Read `age` carefully.** It is doing two jobs at once: depression falls
monotonically from 64.3% in the 18–22 band to 0.8% in the 45–60 band, and
`age < 30` also predicts student status with 84.5% accuracy. `is_student`
scores only 0.0026 — not because the cohort is irrelevant, but because
permuting it leaves `age` carrying the same information. Permutation
importance understates any feature that has a correlated twin, and these two
are twins.

`suicidal_thoughts` remains the strongest non-demographic predictor. As the
original notebook argued, it is arguably a symptom rather than a cause and is
a PHQ-9 diagnostic criterion — that reasoning still holds and is worth keeping.

## Design decisions

- **Every row is kept.** Class imbalance is handled with `class_weight="balanced"`,
  not by deleting majority-class rows.
- **All transforms live inside a `Pipeline`.** Imputation and scaling are fitted
  per fold, so nothing leaks from validation into training.
- **Rare categories fold automatically.** `City` and `Profession` go through
  `OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist")`, so
  junk values like `Kalyan` in the Degree column or an unseen city cannot break
  a run. The old approach hand-listed 22 bad values it had found by eye.
- **`Name` is not a feature.** It is an identifier with 422 values.
- **`Sleep Duration` is parsed, not looked up** — numbers are extracted and
  converted to midpoint hours, so both quoting styles and all 36 variants work.

## Data

`train.csv` and `test.csv` are not tracked in this repository — they belong to
the Kaggle competition that published them, and the rows carry name, city and
mental-health fields that do not need a second public home. Download them from
the competition page and place them in this directory before running anything.
Everything in `outputs/` was produced from them and is tracked, so the results
below can be inspected without re-running the pipeline.

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
