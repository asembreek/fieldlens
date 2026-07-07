import numpy as np
import pandas as pd
# from pandas.api.types import is_float_dtype


def add_rolling_features(df, columns, periods):
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


def add_cyclic_doy(df):
    df["DOY_sin"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
    df["DOY_cos"] = np.cos(2 * np.pi * df["DOY"] / 365.25)
    return df


def add_gdd(df, tmin_col="temperature_2m_min", tmax_col="temperature_2m_max", base=10):
    df_gdd = df.copy()

    df_gdd["gdd"] = _calculate_gdd(df[tmin_col], df[tmax_col], base)

    return df_gdd


def _calculate_gdd(tmin, tmax, base):
    gdd = (tmax + tmin) / 2 - base
    return np.maximum(gdd, 0)
