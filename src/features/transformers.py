import numpy as np
import pandas as pd
from models import NDVIClimatologyGAM as shift
from data import utils


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


class PhenologyFeatures:
    def __init__(
        self,
        start_of_season,
        end_of_season,
        base_temp=10,
        growth_stages=True,
        cumulative_gdd=True,
        season_markers=True,
    ):
        self.start_of_season = start_of_season
        self.end_of_season = end_of_season

        self.base_temp = base_temp
        self.growth_stages = growth_stages
        self.cumulative_gdd = cumulative_gdd
        self.season_markers = season_markers

    def fit(self, X):
        return self

    def transform(self, X):
        df = X.copy()

        date_cols = ["doy", "year"]
        contains_dates = True

        if not all(col in X.columns for col in date_cols):
            df = utils.add_date_info(df)
            contains_dates = False

        if not contains_dates:
            df = df.drop(["doy", "year"], axis=1)

        return self

    def fit_transform(self, X):
        return self.transform(X)

    def _add_growth_season_markers(self, X):
        df = X.copy()

        df["Ag_year"] = np.where(
            df["doy"] >= self.start_of_season, df["year"], df["year"] - 1
        )
        df["in_season"] = (
            (df["doy"] >= self.start_of_season) & (df["doy"] <= self.end_of_season)
        ).astype(int)
        df = df.sort_values(by=["Ag_year", "doy"])

        return df

    def _add_gdd(self, X, tmin_col="temperature_2m_min", tmax_col="temperature_2m_max"):
        df = X.copy()

        df["gdd"] = self._calculate_gdd(df[tmin_col], df[tmax_col])

        return df

    def _calculate_gdd(self, tmin, tmax):
        gdd = (tmax + tmin) / 2 - self.base_temp
        return np.maximum(gdd, 0)

    def _add_cumulative_gdd(self, X):
        df = X.copy()
        # Only sum GDD observations that are in season
        df["GDD_in_season"] = df["GDD"].where(df["in_season"] == 1, 0)

        df["cumulative_GDD"] = df.groupby("Ag_year")["GDD_in_season"].cumsum()
        df["cumulative_GDD"] = df["cumulative_GDD"].where(df["in_season"] == 1, 0)

        return df.drop("GDD_in_season", axis=1)

    def _add_growth_stages(self, X):
        df = X.copy()

        stages = [
            "R6",
            "R5",
            "R4",
            "R3",
            "R2",
            "R1",
            "VT",
            "V16+",
            "V13",
            "V12",
            "V8-V9",
            "V6",
            "V4",
            "V2",
            "VE",
            "SoS",
        ]

        conditions = [
            df["cumulative_GDD"] >= 1278,
            df["cumulative_GDD"] >= 1083,
            df["cumulative_GDD"] >= 1042,
            df["cumulative_GDD"] >= 944,
            df["cumulative_GDD"] >= 833,
            df["cumulative_GDD"] >= 750,
            df["cumulative_GDD"] >= 639,
            df["cumulative_GDD"] >= 531,
            df["cumulative_GDD"] >= 500,
            df["cumulative_GDD"] >= 378,
            df["cumulative_GDD"] >= 289,
            df["cumulative_GDD"] >= 200,
            df["cumulative_GDD"] >= 111,
            df["cumulative_GDD"] >= 56,
            df["cumulative_GDD"] > 0,
        ]

        df["Growth_Stage"] = np.select(conditions, stages, default="Off-Season")
        df["Growth_Stage"] = np.where(
            df["in_season"] == 0, "Off-Season", df["Growth_Stage"]
        )
        return df


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

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X):
        pass

    def _do_moisture_layers(self, X):
        required = {"Timestamp", "Growth_Stage"}

        missing = required - set(X.columns)
        if missing and self.root_weighted_moisture:
            raise ValueError(
                f"Missing required columns: {missing}. "
                "Run PhenologyFeatures(growth_stages=True) first."
            )

        df = X.copy()
        contains_dates = True
        if "doy" not in X.columns:
            df = utils.add_date_info(df)

        prefix = "volumetric_soil_water_layer"
        layers = range(1, 5)
        df = df[[f"{prefix}_{l}" for l in layers]].copy()
        df[["Growth_Stage", "doy", "Timestamp"]] = X[
            ["Growth_Stage", "doy", "Timestamp"]
        ].copy()

        if not contains_dates:
            df = df.drop("doy", axis=1)
        return df


class TemporalFeatures:
    def __init__(self, cyclic_doy=True):
        self.cyclic_doy = cyclic_doy

    def fit(self, X):
        return self

    def transform(self, X):
        pass

    def fit_transform(self, X):
        return self

    def _add_cyclic_doy(self, X):
        X["DOY_sin"] = np.sin(2 * np.pi * X["DOY"] / 365.25)
        X["DOY_cos"] = np.cos(2 * np.pi * X["DOY"] / 365.25)
        return X
