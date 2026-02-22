import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src.feature_engineering import add_engineered_features

RANDOM_STATE = 123

def main():
    df = pd.read_csv("data/CW1_train.csv")
    y = df["outcome"]
    X = df.drop(columns=["outcome"])
    X = add_engineered_features(X) # Add engineered features before preprocessing

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    
    # Preprocessing: scale numerics, one-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(random_state=RANDOM_STATE)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    
    # 3-fold used here to reduce compute time during RandomizedSearch
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    # Fit the search
    search.fit(X, y)

    print("\nBest CV R2:", search.best_score_)
    print("Best params:", search.best_params_)

    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values("mean_test_score", ascending=False)
    cols = ["mean_test_score", "std_test_score", "params"]
    print("\nTop 10 configurations:")
    print(results[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
