"""Simulation of category mapping errors."""

from pmq.config import MEN, WOMEN


def simulate_swaps(df, rate, seed):
    """Swap Women's <-> Men's for a random share of the combined adult pool.

    Returns a copy with `assigned_category` (labels after the swaps) and
    `is_corrupted` (True for swapped offers, used only for evaluation).
    Kids' Clothing is never changed.
    """
    result = df.copy()
    result["assigned_category"] = result["original_category"]

    eligible = result[result["original_category"].isin([WOMEN, MEN])]
    selected = eligible.sample(frac=rate, random_state=seed)  # without replacement

    for row_index in selected.index:
        if result.loc[row_index, "original_category"] == WOMEN:
            result.loc[row_index, "assigned_category"] = MEN
        else:
            result.loc[row_index, "assigned_category"] = WOMEN

    result["is_corrupted"] = result["assigned_category"] != result["original_category"]
    return result
