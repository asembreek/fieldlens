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


class PhenologyFeatures:
    def __init__(self, base_temp=10, growth_stages=True, cumulative_gdd=True):
        self.base_temp = base_temp
        self.growth_stages = growth_stages
        self.cumulative_gdd = cumulative_gdd

    def transform(self, X):
        pass

    def add_gdd(self, X, tmin_col="temperature_2m_min", tmax_col="temperature_2m_max"):
        df_gdd = X.copy()

        df_gdd["gdd"] = self._calculate_gdd(X[tmin_col], X[tmax_col])

        return df_gdd

    def _calculate_gdd(self, tmin, tmax):
        gdd = (tmax + tmin) / 2 - self.base_temp
        return np.maximum(gdd, 0)


class SoilFeatures:
    def __init__(
        self,
        root_weighted_moisture=True,
        shallow_temperature=True,
        moisture_pca=True,
        temperature_pca=True,
    ):
        self.root_weighted_moisture = root_weighted_moisture
        self.shallow_temperature = shallow_temperature
        self.moisture_pca = moisture_pca
        self.temperature_pca = temperature_pca

    def transform(self, X):
        pass


class TemporalFeatures:
    def __init__(self, cyclic_doy=True):
        self.cyclic_doy = cyclic_doy

    def transform(self, X):
        pass

    def add_cyclic_doy(self, X):
        X["DOY_sin"] = np.sin(2 * np.pi * X["DOY"] / 365.25)
        X["DOY_cos"] = np.cos(2 * np.pi * X["DOY"] / 365.25)
        return X
