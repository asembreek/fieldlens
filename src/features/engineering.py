import numpy as np
import pandas as pd
# from pandas.api.types import is_float_dtype


def add_rolling_statistics(df, columns, periods):
    df = df.sort_values("Timestamp")
    df_roll = pd.DataFrame(df["Timestamp"].copy())

    # columns = df.columns

    for c in columns:
        if pd.api.types.is_float_dtype(df[c]):
            for window in periods:
                df_roll[f"{c}_{window}D_mean"] = df[c].rolling(window=window).mean()
                df_roll[f"{c}_{window}D_std"] = df[c].rolling(window=window).std()
                df_roll[f"{c}_{window}D_sum"] = df[c].rolling(window=window).sum()
                df_roll[f"{c}_{window}D_median"] = df[c].rolling(window=window).median()

    return df_roll


def roll_cols(df, columns, periods):
    df = df.sort_values("Timestamp")
    features = {}

    for c in columns:
        if pd.api.types.is_float_dtype(df[c]):
            s = df[c]

            for window in periods:
                roll = s.rolling(window=window, min_periods=window)

                features[f"{c}_{window}D_mean"] = roll.mean()
                # features[f"{c}_{window}D_std"] = roll.std()
                features[f"{c}_{window}D_sum"] = roll.sum()
            # features[f"{c}_{window}D_median"] = roll.median()

    df_roll = pd.concat(
        [df[["Timestamp"]], pd.DataFrame(features, index=df.index)], axis=1
    )
    return df_roll


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
