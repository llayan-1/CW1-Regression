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

from sklearn.model_selection import cross_val_score, KFold

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
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
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

    # CV score (comparable to model_selection.py output)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
    print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full training set
    pipe.fit(X_train, y_train)
    print("Model trained on full training set.")

    # Feature importances from base estimators
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    for name, est in pipe.named_steps["model"].named_estimators_.items():
        if hasattr(est, "feature_importances_"):
            imp = pd.Series(est.feature_importances_, index=feature_names)
            print(f"\nTop 10 feature importances ({name}):")
            print(imp.sort_values(ascending=False).head(10).to_string())

    # Residual check on training data
    y_pred_train = pipe.predict(X_train)
    residuals = y_train.values - y_pred_train
    print("\nTraining residuals:")
    print(f"  Mean : {residuals.mean():.4f}")
    print(f"  Std  : {residuals.std():.4f}")
    print(f"  Min  : {residuals.min():.4f}")
    print(f"  Max  : {residuals.max():.4f}")

    # Predict on test set
    yhat = pipe.predict(X_test)

    # Save submission
    out = pd.DataFrame({"yhat": yhat})
    out.to_csv("CW1_submission_K23065725.csv", index=False)
    print("Saved: CW1_submission_K23065725.csv")


if __name__ == "__main__":
    main()