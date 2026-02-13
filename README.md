# CW1 Regression Coursework

This repository contains my implementation for Coursework 1 (Regression). The objective is to train a regression model to predict the variable **`outcome`** from the remaining features in the provided tabular dataset.

---

## Repository Structure

* `data/`
  Contains the training and test datasets (`CW1_train.csv`, `CW1_test.csv`).

* `src/`
  Source code for:

  * Exploratory data analysis
  * Preprocessing and feature engineering
  * Model selection and tuning
  * Final model training and prediction

* `notebooks/`
  Jupyter notebooks used for exploratory data analysis.

---

## Requirements

* Python 3.10+
* Required libraries:

```bash
pip install scikit-learn pandas numpy matplotlib
```

---

## Reproducing the Final Submission

From the repository root directory, run:

```bash
python -m src.final_predict
```

This script:

1. Loads the full training dataset.
2. Applies the final preprocessing and feature engineering pipeline.
3. Trains the selected model on the full training data.
4. Generates predictions for `CW1_test.csv`.
5. Saves the output to:

```
CW1_submission_K23065725.csv
```

The submission file contains a single column named `yhat`, as required.

---

## Evaluation Protocol

* Model selection and hyperparameter tuning were performed using **5-fold cross-validation on the training set only**.
* No information from the test set was used during model development, preventing data leakage.
* Performance is measured using out-of-sample **R²**.

---

## Model Comparison

| Model                        | CV R²  |
| ---------------------------- | ------ |
| Stacking (HistGB + RF)       | 0.4718 |
| HistGradientBoosting (tuned) | 0.4710 |
| RandomForest (tuned)         | 0.4624 |
| Baseline Ridge               | 0.2876 |

---

## Final Model

The final model is a **StackingRegressor** with:

* Base learners:

  * Tuned HistGradientBoostingRegressor
  * Tuned RandomForestRegressor
* Meta-learner:

  * Ridge regression
* Internal stacking cross-validation:

  * 5-fold

The final model is trained on the full training dataset before generating test predictions.
