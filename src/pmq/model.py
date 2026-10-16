"""Grouped cross-validation and out-of-fold suspicion scores."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


def build_model_text(df):
    """The only model input: product name and description joined with a space."""
    return df["product_name"] + " " + df["description"]


def make_pipeline(seed):
    # A fresh pipeline per fold, so TF-IDF is fitted on training rows only.
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def grouped_splits(labels, groups, n_splits, seed):
    """List of (train_positions, validation_positions); groups never cross folds."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # The splitter only needs X for its length, so the labels are passed as X too.
    return list(cv.split(labels, labels, groups))


def out_of_fold_probabilities(texts, labels, splits, seed):
    """Each offer is scored by a model that never saw it (or its group) in training."""
    categories = sorted(labels.unique())
    probabilities = pd.DataFrame(np.nan, index=labels.index, columns=categories)

    for train_positions, validation_positions in splits:
        model = make_pipeline(seed)
        model.fit(texts.iloc[train_positions], labels.iloc[train_positions])
        if list(model.classes_) != categories:
            raise ValueError("A training fold is missing a category")
        probabilities.iloc[validation_positions] = model.predict_proba(texts.iloc[validation_positions])

    if probabilities.isna().any().any():
        raise ValueError("Some offers did not receive a held-out prediction")
    return probabilities


def suspicion_scores(probabilities, assigned):
    """1 - P(assigned category): high when the model does not support the label."""
    scores = pd.Series(np.nan, index=assigned.index)
    for category in probabilities.columns:
        rows = assigned == category
        scores[rows] = 1 - probabilities.loc[rows, category]
    return scores
