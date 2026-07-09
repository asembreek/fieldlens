import numpy as np
import pandas as pd


class RollingFeatures:
    def __init__(self, columns, windows=(7, 14)):
        self.columns = columns
        self.windows = windows

    def fit(self, X):
        return self

    def transform(self, X):
        df = X.sort_values("Timestamp").copy()

        for c in self.columns:
            if pd.api.types.is_float_dtype(df[c]):
                s = df[c]

                for window in self.windows:
                    roll = s.rolling(window=window, min_periods=window)

                    df[f"{c}_{window}D_mean"] = roll.mean()
                    # features[f"{c}_{window}D_std"] = roll.std()
                    df[f"{c}_{window}D_sum"] = roll.sum()
                    # features[f"{c}_{window}D_median"] = roll.median()

        return df

    def fit_transform(self, X):
        return self
