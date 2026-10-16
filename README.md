# Product Mapping Quality

Marketplace offers are sometimes listed in the wrong category. This project ranks offers by how
likely their category is wrong, so that a reviewer can check the most suspicious ones first.

## Results

| Test | Top 50 suspects | Precision@50 |
| --- | --- | ---: |
| Simulated errors (500 swapped labels) | 48 swapped | **96%** (random order: 9.1%) |
| Original labels, reviewed by hand | 8 likely errors, 27 false alarms, 15 unclear | **16%** |

The 96% only shows that the model finds errors I inserted myself. On the original labels, 8 of
the top 50 looked like real errors. All review verdicts are in
[results/manual_review.csv](results/manual_review.csv).

## Data

[Flipkart Products](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products) (Kaggle,
CC BY-SA 4.0). I used three categories: Women's, Men's and Kids' Clothing. After removing exact
duplicates, 5,501 offers remain. The data is not included in the repository; the product names
in `results/` come from it and are shared under the same license.

The source data is messy:

- 1,732 of 20,000 category paths are cut off with "...", and in 2,050 the last category level is
  just the product name.
- 11% of the clothing offers (693 of 6,194) repeat another offer's name and description.
- Brand is missing for 29% of all offers, and 91% have no rating.
- Category labels are not verified. Clear conflicts are rare: in 13 of 5,501 clothing offers the
  name points only to a different audience than the category (e.g. "Women's" in Men's Clothing).

## Method

1. Swap Women's ↔ Men's for 10% of adult offers (seed 42).
2. Train TF-IDF + logistic regression on the product name and description.
3. Score every offer with 5-fold cross-validation. Offers with the same name stay in the same fold.
4. Rank offers by `1 − P(assigned category)` and check the top 50.
5. Repeat steps 2–4 on the original labels and review the top 50 by hand.

## Error analysis

- Likely real errors: e.g. *"Women's A-line Red Skirt"* listed in Kids', *"Women's Ankle Length
  Socks"* listed in Men's.
- Main weakness: 18 of the 27 false alarms are girls' clothes that the model calls Women's.
- Texts naming two audiences (*"Girl's, Women's"*) or none were marked unclear, not as errors.

## Limitations and next steps

- Random swaps may be easier to find than real errors.
- Words like "Women's" in product names may be a shortcut. Next step: remove them and re-run.
- Only one seed was run.

## How to run

```bash
uv sync --group dev --group notebook
uv run pytest
```

Put the CSV in `data/raw/` and run the notebooks in `notebooks/`. The code is in `src/pmq/`.

## AI assistance

The project and methodology decisions, and the review verdicts, are mine. The code was written
with an AI assistant and reviewed by me.

## License

Code: [MIT](LICENSE). Data excerpts in `results/`: CC BY-SA 4.0, as the source dataset.
