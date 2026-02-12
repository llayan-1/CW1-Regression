import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)

from src.feature_engineering import add_engineered_features

RANDOM_STATE = 123


def main():
    np.random.seed(RANDOM_STATE)

    # Load data
    train_df = pd.read_csv("data/CW1_train.csv")
    test_df = pd.read_csv("data/CW1_test.csv")

    y_train = train_df["outcome"]
    X_train = train_df.drop(columns=["outcome"])
    X_train = add_engineered_features(X_train)
    X_test = add_engineered_features(test_df)

    # Column types
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Stacking: HistGB_tuned + RF_tuned, meta-learner = Ridge
    stacking = StackingRegressor(
        estimators=[
            ("histgb", HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=200, max_depth=3,
                l2_regularization=1.0, random_state=RANDOM_STATE,
            )),
            ("rf", RandomForestRegressor(
                n_estimators=300, max_depth=15, min_samples_leaf=10,
                max_features=0.5, min_samples_split=2, random_state=RANDOM_STATE,
            )),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", stacking),
    ])

    # Train on full training set
    pipe.fit(X_train, y_train)
    print("Model trained on full training set.")

    # Predict on test set
    yhat = pipe.predict(X_test)

    # Save submission
    out = pd.DataFrame({"yhat": yhat})
    out.to_csv("CW1_submission_K23065725.csv", index=False)
    print("Saved: CW1_submission_K23065725.csv")


if __name__ == "__main__":
    main()