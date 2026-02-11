import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from src.feature_engineering import add_engineered_features

np.random.seed(123)

# Load training data
df = pd.read_csv("data/CW1_train.csv")
X = df.drop(columns=["outcome"])
y = df["outcome"]
X = add_engineered_features(X)

# Identify feature types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing (fit inside CV via pipeline)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

models = {
    "Ridge(alpha=1.0)": Ridge(alpha=1.0),
    "Lasso(alpha=0.01)": Lasso(alpha=0.01, max_iter=5000),
    "RandomForest(200)": RandomForestRegressor(n_estimators=200, random_state=123),
    "HistGB": HistGradientBoostingRegressor(random_state=123),
    "HistGB_tuned": HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_depth=3,
        l2_regularization=1.0, random_state=123,
    ),
    "Stack(HistGB+RF)": StackingRegressor(
        estimators=[
            ("histgb", HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=200, max_depth=3,
                l2_regularization=1.0, random_state=123)),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=123)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
    ),
}

cv = KFold(n_splits=5, shuffle=True, random_state=123)

rows = []
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

    scores = cross_val_score(pipe, X, y, scoring="r2", cv=cv, n_jobs=-1)
    rows.append(
        {"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()}
    )

results_df = pd.DataFrame(rows).sort_values("mean_r2", ascending=False)
print(results_df.to_string(index=False))

print("\nDone running model selection.\n")
