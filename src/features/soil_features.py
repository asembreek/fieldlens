import numpy as np
import pandas as pd

from data import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# soil_features.py

DEFAULT_PCA_CONFIG = {
    "soil_temp": {
        "columns": [f"soil_temperature_layer_{l}" for l in range(1, 4)],
        "n_components": 1,
    },
    "soil_water": {
        "columns": [f"volumetric_soil_water_layer_{l}" for l in range(1, 3)],
        "n_components": 1,
    },
}


class SoilFeatures:
    EARLY_WEIGHTS = np.array([0.9375 / 2, 0.9375 / 2, 0.0625, 0.0])
    MEDIAN_WEIGHTS = np.array([0.4285 / 2, 0.4285 / 2, 0.4762, 0.0953])
    MATURE_WEIGHTS = np.array([0.3846 / 2, 0.3846 / 2, 0.4615, 0.1538])

    ROOT_WEIGHTS = {
        "Off-Season": np.array([0.0, 0.0, 0.0, 0.0]),
        "SoS": EARLY_WEIGHTS,
        "VE": EARLY_WEIGHTS,
        "V2": EARLY_WEIGHTS,
        "V4": EARLY_WEIGHTS,
        "V6": EARLY_WEIGHTS,
        "V8-V9": MEDIAN_WEIGHTS,
        "V12": MEDIAN_WEIGHTS,
        "V13": MEDIAN_WEIGHTS,
        "V16+": MATURE_WEIGHTS,
        "VT": MATURE_WEIGHTS,
        "R1": np.array([0.3500 / 2, 0.3500 / 2, 0.5500, 0.1000]),
        "R2": np.array([0.2500 / 2, 0.2500 / 2, 0.6667, 0.0417]),
        "R3": np.array([0.3333 / 2, 0.3333 / 2, 0.6000, 0.1333]),
        "R4": np.array([0.2857 / 2, 0.2857 / 2, 0.5714, 0.1429]),
        "R5": np.array([0.2069 / 2, 0.2069 / 2, 0.7241, 0.1034]),
        "R6": np.array([0.3478 / 2, 0.3478 / 2, 0.5217, 0.1304]),
    }

    def __init__(
        self,
        root_weighted_moisture=True,
        shallow_temperature=True,
        moisture_pca=True,
        temperature_pca=True,
        pca_config=None,
    ):
        self.root_weighted_moisture = root_weighted_moisture
        self.shallow_temperature = shallow_temperature
        self.moisture_pca = moisture_pca
        self.temperature_pca = temperature_pca
        self.pca_config = DEFAULT_PCA_CONFIG if pca_config is None else pca_config

    def fit(self, X):
        return self

    def transform(self, X):
        return df

    def fit_transform(self, X):
        self.fit(X)
        df = self.transform(X)
        return df

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

        df = df[[f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]].copy()
        df[["Growth_Stage", "doy", "Timestamp"]] = X[
            ["Growth_Stage", "doy", "Timestamp"]
        ].copy()

        if self.root_weighted_moisture:
            df = self._add_root_weighted_moisture(df)

        if not contains_dates:
            df = df.drop("doy", axis=1)
        return df

    def _add_root_weighted_moisture(self, X):
        df = X.copy()

        moisture_cols = [f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]
        moisture = df[moisture_cols].to_numpy()

        weights = np.empty(shape=(len(df), 4), dtype=float)
        for i, stage in enumerate(df["Growth_Stage"]):
            weights[i] = self.ROOT_WEIGHTS[stage]

        df["root_weighted_moisture"] = np.sum(
            moisture * weights,
            axis=1,
        )
