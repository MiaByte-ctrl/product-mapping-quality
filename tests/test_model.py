import numpy as np
import pandas as pd

from pmq.evaluate import count_group_overlap
from pmq.model import grouped_splits, out_of_fold_probabilities, suspicion_scores


def make_offers():
    rows = []
    for i in range(30):
        for label, word in [("W", "women dress"), ("M", "men shirt"), ("K", "kids frock")]:
            # Two offers per group, so splitting a group would be possible.
            for variant in range(2):
                rows.append({"text": f"{word} {i} v{variant}", "label": label, "group": f"{label}{i}"})
    return pd.DataFrame(rows)


def test_no_group_in_both_training_and_validation():
    df = make_offers()
    splits = grouped_splits(df["label"], df["group"], n_splits=5, seed=42)
    assert count_group_overlap(df["group"], splits) == 0


def test_every_offer_validated_exactly_once():
    df = make_offers()
    splits = grouped_splits(df["label"], df["group"], n_splits=5, seed=42)
    validation_positions = np.concatenate([validation for _, validation in splits])
    assert sorted(validation_positions) == list(range(len(df)))


def test_out_of_fold_probabilities_are_complete_and_valid():
    df = make_offers()
    splits = grouped_splits(df["label"], df["group"], n_splits=5, seed=42)
    probabilities = out_of_fold_probabilities(df["text"], df["label"], splits, seed=42)
    assert probabilities.notna().all().all()
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_suspicion_score_uses_assigned_category():
    probabilities = pd.DataFrame({"K": [0.1, 0.7], "M": [0.2, 0.2], "W": [0.7, 0.1]})
    assigned = pd.Series(["W", "W"])
    assert np.allclose(suspicion_scores(probabilities, assigned), [0.3, 0.9])
