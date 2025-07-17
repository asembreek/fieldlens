import numpy as np
import pandas as pd


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


def lag_col(df, columns, periods):
    df = df.sort_values("Timestamp")
    df_lag = pd.DataFrame(df["Timestamp"].copy())

    if not isinstance(periods, list):
        raise TypeError(f"Expected list, but got {type(periods).__name__}")

    for c in columns:
        if pd.api.types.is_float_dtype(df[c]):
            for lag in periods:
                df_lag[f"{c}_lag_{lag}"] = df[c].shift(lag)
    return df_lag


def sin_cos_doy(df):
    df["DOY_sin"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
    df["DOY_cos"] = np.cos(2 * np.pi * df["DOY"] / 365.25)
    return df
