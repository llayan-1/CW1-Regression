import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

from src.feature_engineering import add_engineered_features

RANDOM_STATE = 123

def main():
    # Load data
    df = pd.read_csv("data/CW1_train.csv")

    y = df["outcome"]
    X = df.drop(columns=["outcome"])

    # Feature engineering (no leakage: uses only X)
    X = add_engineered_features(X)

    # Column types AFTER engineering
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    # Model
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # CV
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Grid (safe size: good improvement, not too slow)
    param_grid = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_iter": [200, 400],
        "model__max_depth": [3, 5],
        "model__l2_regularization": [0.1, 1.0],
        "model__min_samples_leaf": [5, 10, 20, 50],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X, y)

    print("\nBest CV R2:", search.best_score_)
    print("Best params:", search.best_params_)

    # Show top 10 configs
    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values("mean_test_score", ascending=False)
    cols = ["mean_test_score", "std_test_score", "params"]
    print("\nTop 10 configurations:")
    print(results[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()