# Diamond Price Prediction

This project predicts a diamond's price in US dollars from its physical measurements and quality attributes. The work is documented in `diamond_price_prediction.ipynb`.

## Dataset

The local `diamonds.csv` has 53,940 observations. Its fields match the [`ggplot2::diamonds`](https://ggplot2.tidyverse.org/reference/diamonds.html) dataset, with an additional local `ID` column. It includes carat, cut, colour, clarity, dimensions, depth, table, and price.

### Source and redistribution status

The `ggplot2` documentation describes the dataset as prices and attributes for almost 54,000 diamonds. Its original author states that it was scraped from `diamondse.info` in February 2007 in this [R-help archive post](https://www.stat.math.ethz.ch/pipermail/r-help/2023-April/477277.html).

`ggplot2` software is distributed under the [MIT License](https://github.com/tidyverse/ggplot2/blob/main/LICENSE.md). That software license does **not** by itself establish a redistribution license for the independently sourced diamond-price data. No dedicated redistribution license for the underlying scraped data was found during this review. Therefore, `diamonds.csv` is excluded from Git by default. Obtain the dataset from its source or your course materials, review the applicable terms, and place it in the repository root as `diamonds.csv` before running the notebook.

## Methodology

1. Load the data and remove the non-predictive `ID` field.
2. Treat zero values in `x`, `y`, and `z` as invalid dimensions and impute them with the median inside the preprocessing pipeline.
3. Explore price distributions, feature relationships, quality groups, and numerical correlations.
4. Split the data into 80% training and 20% test sets using `random_state=42`.
5. Establish a mean-price baseline.
6. Encode ordinal quality features (`cut`, `color`, and `clarity`) using their domain order and median-impute numerical features.
7. Train a `HistGradientBoostingRegressor`.
8. Tune the model with `RandomizedSearchCV`, shuffled 5-fold cross-validation, and RMSE scoring.
9. Evaluate only once on the held-out test set with MAE, RMSE, and R2; inspect actual-vs-predicted and residual plots.

## Results

The notebook was executed from a clean kernel. Results below use the 80/20 held-out test split. The tuned model was selected with randomized search and shuffled 5-fold cross-validation on the training split only.

| Model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Mean baseline | $3,020.51 | $3,987.22 | -0.0001 |
| Initial HistGradientBoosting | $279.77 | $538.22 | 0.9818 |
| Tuned HistGradientBoosting | $278.01 | $537.22 | 0.9818 |

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `diamond_price_prediction.ipynb` and run all cells from a fresh kernel. The hyperparameter search uses all available CPU cores. If your machine cannot create parallel workers, set `n_jobs=1` in the `RandomizedSearchCV` cell.
