"""Fixed settings shared by the whole project."""

from pathlib import Path

SEED = 42
NOISE_RATE = 0.10
N_SPLITS = 5
REVIEW_CAPACITY = 50

WOMEN = "Clothing >> Women's Clothing"
MEN = "Clothing >> Men's Clothing"
KIDS = "Clothing >> Kids' Clothing"
LABELS = [WOMEN, MEN, KIDS]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "flipkart_com-ecommerce_sample.csv"
REVIEW_PATH = PROJECT_ROOT / "results" / "manual_review.csv"
