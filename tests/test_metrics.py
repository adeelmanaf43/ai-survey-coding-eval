"""Tests for coding_eval/metrics.py: tiny made-up examples where the right answer is known by hand.
Run all tests with:  pytest"""
import numpy as np

from coding_eval.metrics import bootstrap, gap, mcnemar_exact, mean_seconds, review_curve, scores


def test_perfect_codes_score_1():
    truth = ["Health", "Economy", "Immigration", "Economy"]
    assert scores(truth, truth) == {"accuracy": 1.0, "macro_f1": 1.0, "kappa": 1.0}


def test_always_guessing_the_biggest_category():
    # 3 of 4 answers are Economy, so always guessing Economy is 75% right, but no better than chance (kappa 0).
    # Macro-F1: Economy F1 = 2 x 0.75 x 1 / (0.75 + 1) = 0.857, Health F1 = 0, average = 0.429
    s = scores(["Economy", "Economy", "Economy", "Health"], ["Economy"] * 4)
    assert s == {"accuracy": 0.75, "macro_f1": 0.429, "kappa": 0.0}


def test_mcnemar_by_hand():
    # 3 answers where only one method is right, all 3 for the same method:
    # chance of a split this uneven = 1/8 each way, so p = 2 x 1/8 = 0.25
    assert mcnemar_exact(0, 3) == 0.25
    # 1 vs 9: (1 + 10) / 1024 each way, so p = 22 / 1024
    assert mcnemar_exact(1, 9) == 22 / 1024


def test_mcnemar_edge_cases():
    assert mcnemar_exact(0, 0) == 1.0              # no answer where the methods differ: no evidence at all
    assert mcnemar_exact(27, 26) == 1.0            # an almost even split: capped at 1
    assert mcnemar_exact(41, 17) == mcnemar_exact(17, 41)   # the order of the two methods doesn't matter
    assert round(mcnemar_exact(41, 17), 3) == 0.002


def test_review_curve_small_example():
    # 5 answers, riskiest first. Llama got the first 2 wrong.
    auto, overall = review_curve([0, 0, 1, 1, 1])
    # nobody checks: 3 of 5 right; check 1: 3 of the 4 left; check 2: 3 of 3 left
    assert list(auto) == [0.6, 0.75, 1.0, 1.0, 1.0]
    # whole dataset if every checked answer ends up right: (right answers left + answers checked) / 5
    assert list(overall) == [0.6, 0.8, 1.0, 1.0, 1.0]


def test_checking_more_never_lowers_the_whole_dataset():
    # The answers left to the machine can get worse after one more check (0.8 -> 0.75 here),
    # but the whole dataset never does, because a checked answer ends up right
    auto, overall = review_curve([0, 1, 0, 1, 1, 1])
    assert auto[1] == 0.8 and auto[2] == 0.75
    assert all(np.diff(overall) >= 0)


def test_bootstrap_is_reproducible():
    truth = ["A", "B", "A", "C", "B", "A", "C", "A", "B", "A"]
    pred = ["A", "B", "B", "C", "B", "A", "A", "A", "B", "C"]
    first, _ = bootstrap(truth, {"m": pred}, resamples=200, seed=1)
    second, _ = bootstrap(truth, {"m": pred}, resamples=200, seed=1)
    assert first == second                                    # same seed, same intervals
    point, low, high = first["m"]["accuracy"]
    assert point == scores(truth, pred)["accuracy"] == 0.7    # the point value is the plain score
    assert 0 <= low <= high <= 1


def test_gap_between_identical_methods_is_zero():
    truth = ["A", "B", "A", "C", "B", "A"]
    pred = ["A", "B", "B", "C", "B", "A"]
    _, draws = bootstrap(truth, {"one": pred, "two": pred}, resamples=100, seed=1)
    assert gap(draws, "one", "two", 0.0) == [0.0, 0.0, 0.0]


def test_mean_seconds_leaves_out_pauses():
    # 42,012 s is a laptop that went to sleep, not a slow answer (D-022)
    assert mean_seconds([6, 7, 42012], timeout=600) == (6.5, 1)
