from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def make_preprocessor(X):
    """
    Create a preprocessing transformer for the CW1 dataset.
    Categorical features are one-hot encoded.
    Numerical features are passed through unchanged.
    """
    categorical_cols = ["cut", "color", "clarity"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    return preprocessor

