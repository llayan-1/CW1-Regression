CW1 Regression Coursework

This repository contains the implementation for Coursework 1.
The task is to train and evaluate a regression model to predict the variable "outcome"
using tabular data provided by the course.

## Structure
- `data/` - training and test datasets
- `src/` - source code for EDA, preprocessing, feature engineering, model selection, tuning, and prediction
- `notebooks/` - exploratory data analysis

## Requirements

- Python 3.10+
- Install dependencies:
```
pip install scikit-learn pandas numpy matplotlib
```

## How to Reproduce

From the repo root, run:
```
python -m src.final_predict
```
This trains the final Stacking model (HistGradientBoosting + RandomForest) on the full training set
and saves predictions to `CW1_submission_K23065725.csv`.

## Model Summary

| Model               | CV R2   |
|---------------------|---------|
| Stack (HistGB + RF) | 0.4718  |
| HistGB tuned        | 0.4710  |
| RF tuned            | 0.4624  |
| Baseline Ridge      | 0.2876  |

Final model: **StackingRegressor** with tuned HistGradientBoosting and RandomForest base learners,
Ridge meta-learner, 5-fold internal CV stacking.