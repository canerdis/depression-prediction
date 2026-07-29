import numpy as np
import pytest

from evaluate import choose_threshold, bootstrap_pr_auc_gap


def test_f2_prefers_recall_over_precision():
    # Constructed so F1 and F2 genuinely disagree rather than tie. Two positives
    # score highest (0.90, 0.95) with nothing above them, giving F1's optimum at
    # threshold 0.90: precision 1.0, recall 0.4. Lowering the threshold all the
    # way to 0.05 pulls in every remaining positive at the cost of eight false
    # positives (precision 0.3846, recall 1.0) — worse on F1, but F2 weights
    # recall four times as heavily as precision and prefers it.
    #
    # Verified by hand (via a scratch script, since discarded) that this fixture
    # is not inert: forcing `_fbeta`'s beta to 1.0 for both rules collapses both
    # thresholds to 0.90 and both assertions below to False.
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.95, 0.90, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61,
                        0.60, 0.55, 0.05])

    thr_f2, metrics_f2 = choose_threshold(y_true, y_prob, rule="f2")
    thr_f1, metrics_f1 = choose_threshold(y_true, y_prob, rule="f1")

    assert thr_f2 < thr_f1
    assert metrics_f2["recall"] > metrics_f1["recall"]


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
