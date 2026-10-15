"""Tests for coding_eval/baseline.py"""
import pandas as pd

from coding_eval.baseline import normalise


def test_normalise_ignores_case_and_punctuation():
    text = pd.Series(['Cost of living!!', "NHS.", "cost-of-living"])
    assert list(normalise(text)) == ['cost of living', 'nhs', 'cost of living']
