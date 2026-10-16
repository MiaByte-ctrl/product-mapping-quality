import pandas as pd

from pmq.config import KIDS, MEN, WOMEN
from pmq.noise import simulate_swaps


def make_offers():
    categories = [WOMEN] * 60 + [MEN] * 40 + [KIDS] * 20
    return pd.DataFrame(
        {
            "product_name": [f"item {i}" for i in range(len(categories))],
            "original_category": categories,
        }
    )


def test_swaps_requested_share_of_adult_pool():
    result = simulate_swaps(make_offers(), rate=0.10, seed=42)
    assert result["is_corrupted"].sum() == 10  # 10% of 100 adult offers


def test_swaps_only_between_women_and_men():
    result = simulate_swaps(make_offers(), rate=0.10, seed=42)
    changed = result[result["is_corrupted"]]
    for original, assigned in zip(changed["original_category"], changed["assigned_category"], strict=True):
        assert {original, assigned} == {WOMEN, MEN}


def test_kids_and_text_unchanged():
    offers = make_offers()
    result = simulate_swaps(offers, rate=0.50, seed=0)
    kids = result[result["original_category"] == KIDS]
    assert (kids["assigned_category"] == KIDS).all()
    pd.testing.assert_series_equal(result["product_name"], offers["product_name"])
    assert "assigned_category" not in offers.columns  # input not modified


def test_same_seed_gives_same_swaps():
    first = simulate_swaps(make_offers(), rate=0.10, seed=7)
    second = simulate_swaps(make_offers(), rate=0.10, seed=7)
    pd.testing.assert_series_equal(first["is_corrupted"], second["is_corrupted"])
