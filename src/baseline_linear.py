import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from .preprocessing import make_preprocessor

def main():
    np.random.seed(123)

    # Load data
    train_df = pd.read_csv("data/CW1_train.csv")
    test_df = pd.read_csv("data/CW1_test.csv")

    #split features and target
    y_train = train_df["outcome"]
    x_train = train_df.drop(columns=["outcome"])

    #Build pipline: preprocess then model
    preprocesser = make_preprocessor(x_train)
    model = Pipeline(
        steps=[
        ("preprocess", preprocesser),
        ("regressor", LinearRegression()),   
        ]
    )


    #Train
    model.fit(x_train, y_train)

    # Predict on test
    yhat = model.predict(test_df)

    # Save submission
    out = pd.DataFrame({"yhat": yhat})
    out.to_csv("CW1_submission_K23065725.csv", index=False)
    print("Saved: CW1_submission_K23065725.csv")


if __name__ == "__main__":
    main()