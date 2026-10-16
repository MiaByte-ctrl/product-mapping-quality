"""Ranking and evaluation helpers."""


def rank_by_suspicion(df, scores):
    """Return df sorted from most to least suspicious; ties keep source order."""
    ranked = df.copy()
    ranked["suspicion_score"] = scores
    ranked = ranked.sort_values("suspicion_score", ascending=False, kind="stable")
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def precision_at_k(is_error_in_rank_order, k):
    """Share of true errors among the first k ranked offers."""
    if k <= 0 or k > len(is_error_in_rank_order):
        raise ValueError("k must be between 1 and the number of offers")
    top = is_error_in_rank_order.iloc[:k]
    return top.sum() / k


def random_expected_precision(is_error):
    """Expected precision@k of a uniformly random ordering (same for every k)."""
    return is_error.mean()


def count_group_overlap(groups, splits):
    """Number of validation offers whose group is also in training (should be 0)."""
    shared_count = 0
    for train_positions, validation_positions in splits:
        train_groups = groups.iloc[train_positions]
        validation_groups = groups.iloc[validation_positions]
        shared_count = shared_count + validation_groups.isin(train_groups).sum()
    return shared_count
