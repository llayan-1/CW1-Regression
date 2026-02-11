import numpy as np

def add_engineered_features(df):
    """
    Add engineered features focused on the strongest predictors of outcome.
    Returns a new dataframe.
    """
    df_new = df.copy()

    # Depth transforms — depth is the strongest predictor (r=0.41)
    df_new["depth_sq"] = df_new["depth"] ** 2
    df_new["log_depth"] = np.log(df_new["depth"])

    # Interactions between top predictors
    df_new["depth_b3"] = df_new["depth"] * df_new["b3"]
    df_new["depth_b1"] = df_new["depth"] * df_new["b1"]
    df_new["depth_a1"] = df_new["depth"] * df_new["a1"]
    df_new["b3_b1"] = df_new["b3"] * df_new["b1"]

    return df_new
