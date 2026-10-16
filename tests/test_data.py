import pandas as pd
import pytest

from pmq.config import KIDS, MEN, WOMEN
from pmq.data import (
    drop_exact_text_duplicates,
    level_2_path,
    normalize_product_name,
    select_clothing,
    split_category_path,
)


def test_split_category_path_strips_levels():
    raw = '["Clothing >> Women\'s Clothing >> Western Wear"]'
    assert split_category_path(raw) == ["Clothing", "Women's Clothing", "Western Wear"]


def test_split_category_path_rejects_empty_level():
    with pytest.raises(ValueError):
        split_category_path('["Clothing >>  >> Tops"]')


def test_split_category_path_rejects_multiple_paths():
    with pytest.raises(ValueError):
        split_category_path('["A >> B", "C >> D"]')


def test_level_2_path_single_level_is_missing():
    assert level_2_path('["Clothing >> Men\'s Clothing >> Shirts"]') == MEN
    assert pd.isna(level_2_path('["Clothing"]'))


def test_select_clothing_keeps_only_three_labels():
    df = pd.DataFrame(
        {
            "product_category_tree": [
                '["Clothing >> Women\'s Clothing >> Tops"]',
                '["Clothing >> Kids\' Clothing >> Girls Wear"]',
                '["Footwear >> Women\'s Footwear >> Flats"]',
                '["Clothing"]',
            ]
        }
    )
    result = select_clothing(df)
    assert list(result["original_category"]) == [WOMEN, KIDS]


def test_drop_exact_text_duplicates_keeps_first():
    df = pd.DataFrame(
        {
            "pid": ["a", "b", "c"],
            "product_name": ["Shirt", "Shirt", "Shirt"],
            "description": ["blue", "blue", "red"],
            "original_category": [MEN, MEN, MEN],
        }
    )
    assert list(drop_exact_text_duplicates(df)["pid"]) == ["a", "c"]


def test_drop_exact_text_duplicates_stops_on_conflicting_categories():
    df = pd.DataFrame(
        {
            "product_name": ["Shirt", "Shirt"],
            "description": ["blue", "blue"],
            "original_category": [MEN, WOMEN],
        }
    )
    with pytest.raises(ValueError):
        drop_exact_text_duplicates(df)


def test_normalize_product_name():
    assert normalize_product_name("  Alisha  Solid\tWomen's Cycling SHORTS ") == (
        "alisha solid women's cycling shorts"
    )
