"""Loading, category parsing and cleaning of the Flipkart product data."""

import hashlib
import json

import pandas as pd

from pmq.config import LABELS


def file_sha256(path):
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def load_raw(path):
    # Read everything as text; empty cells stay "" instead of becoming NaN.
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def split_category_path(raw_tree):
    """Parse '["A >> B >> C"]' into ['A', 'B', 'C']."""
    paths = json.loads(raw_tree)
    if len(paths) != 1:
        raise ValueError("Expected exactly one category path")
    path = paths[0]

    levels = []
    for part in path.split(">>"):
        level = part.strip()
        if level == "":
            raise ValueError("Category path contains an empty level")
        levels.append(level)
    return levels


def level_2_path(raw_tree):
    """Return the full path of the first two levels, e.g. 'Clothing >> Men's Clothing'."""
    levels = split_category_path(raw_tree)
    if len(levels) < 2:
        return pd.NA
    return levels[0] + " >> " + levels[1]


def select_clothing(df):
    """Keep offers from the three selected Clothing categories."""
    result = df.copy()
    result["original_category"] = result["product_category_tree"].apply(level_2_path)
    result = result[result["original_category"].isin(LABELS)].copy()
    return result


def drop_exact_text_duplicates(df):
    """Keep the first offer of each identical product_name + description pair."""
    text_columns = ["product_name", "description"]
    categories_per_text = df.groupby(text_columns)["original_category"].nunique()
    if (categories_per_text > 1).any():
        # Keeping the first row would then silently pick one of the categories.
        raise ValueError("Identical texts have different categories; review them first")
    return df.drop_duplicates(subset=text_columns, keep="first").copy()


def normalize_product_name(name):
    """Lowercase and collapse whitespace; used as the grouping key for splits."""
    return " ".join(name.lower().split())


def prepare_dataset(path):
    """Full preparation: select categories, deduplicate and add group ids."""
    df = select_clothing(load_raw(path))
    df = drop_exact_text_duplicates(df)
    df["group_id"] = df["product_name"].apply(normalize_product_name)
    return df
