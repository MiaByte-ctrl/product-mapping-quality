import pandas as pd
import pytest

from pmq.evaluate import count_group_overlap, precision_at_k, random_expected_precision, rank_by_suspicion


def test_precision_at_k():
    is_error = pd.Series([True, True, False, True, False])
    assert precision_at_k(is_error, 2) == 1.0
    assert precision_at_k(is_error, 4) == 0.75


def test_precision_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(pd.Series([True]), 2)
    with pytest.raises(ValueError):
        precision_at_k(pd.Series([True]), 0)


def test_random_expected_precision_is_error_rate():
    assert random_expected_precision(pd.Series([True, False, False, False])) == 0.25


def test_rank_by_suspicion_orders_and_keeps_ties_stable():
    df = pd.DataFrame({"pid": ["a", "b", "c", "d"]})
    scores = pd.Series([0.2, 0.9, 0.2, 0.5])
    ranked = rank_by_suspicion(df, scores)
    assert list(ranked["pid"]) == ["b", "d", "a", "c"]
    assert list(ranked["rank"]) == [1, 2, 3, 4]


def test_count_group_overlap_detects_split_group():
    groups = pd.Series(["a", "a", "b"])
    splits = [([0], [1, 2])]  # group "a" is in both training and validation
    assert count_group_overlap(groups, splits) == 1
