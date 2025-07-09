import pandas as pd
from pandas.api.types import is_float_dtype


def lag_col(df, column, periods, fill=None):
    df_copy = df.copy()
    if not isinstance(periods, list):
        raise TypeError(f"Expected list, but got {type(periods).__name__}")

    if fill == None:
        for lag in periods:
            df_copy[f"{column}_lag_{lag}"] = df_copy[column].shift(lag)
    else:
        for lag in periods:
            df_copy[f"{column}_lag_{lag}"] = df_copy[column].shift(lag, fill_value=fill)
    return df_copy


def add_rolling_statistics(df, periods):
    df = df.sort_values("Timestamp")
    df_features = pd.DataFrame()
    df_features["Timestamp"] = df["Timestamp"].copy()

    columns = df.columns

    for c in columns:
        if pd.api.types.is_float_dtype(df[c]):
            for window in periods:
                df_features[f"{c}_{window}D_mean"] = df[c].rolling(window=window).mean()
                df_features[f"{c}_{window}D_std"] = df[c].rolling(window=window).std()
                df_features[f"{c}_{window}D_sum"] = df[c].rolling(window=window).sum()
                df_features[f"{c}_{window}D_median"] = (
                    df[c].rolling(window=window).median()
                )

    return df_features
