def add_engineered_features(df):
    """
    Add physically/structurally justified engineered features.
    Modifies the dataframe in-place and returns it.
    """
    # Volume of the diamond (x * y * z dimensions)
    df["volume"] = df["x"] * df["y"] * df["z"]

    # Squared carat — captures non-linear price/outcome scaling
    df["carat_sq"] = df["carat"] ** 2

    # Carat * depth interaction — heavier stones with more depth behave differently
    df["carat_depth"] = df["carat"] * df["depth"]

    return df
