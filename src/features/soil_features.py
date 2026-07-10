import numpy as np
from numpy._core.numeric import require
import pandas as pd

from data import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_PCA_CONFIG = {
    # First three layers due to high correlation (see first EDA notebook)
    "soil_temp": {
        "columns": [f"soil_temperature_level_{l}" for l in range(1, 4)],
        "n_components": 1,
    },
    # First two layers since deeper layers aren't correlated (See first EDA notebook)
    "soil_moisture": {
        "columns": [f"volumetric_soil_water_layer_{l}" for l in range(1, 3)],
        "n_components": 1,
    },
}


class SoilFeatures:
    SOIL_MOISTURE = "soil_moisture"
    SOIL_TEMP = "soil_temp"

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
        root_weighted_moisture=False,
        moisture_pca=False,
        temperature_pca=False,
        pca_config=None,
    ):
        self.root_weighted_moisture = root_weighted_moisture

        self.moisture_pca = moisture_pca
        self.temperature_pca = temperature_pca
        self.pca_config = DEFAULT_PCA_CONFIG if pca_config is None else pca_config

        required = set()
        received = set(self.pca_config.keys())
        if moisture_pca:
            required.add("soil_moisture")
        if temperature_pca:
            required.add("soil_temp")

        if required - received:
            raise TypeError(
                f"PCA config must be exactly {required}. Received: {received}"
            )

        self.pca_models = {}
        self.scalers = {}

    def fit(self, X):
        X_pca = X.copy()

        if self.moisture_pca or self.temperature_pca:
            for feature, config in self.pca_config.items():
                cols = config["columns"]
                n_comp = int(config["n_components"])

                self.scalers[feature] = StandardScaler().fit(X_pca[cols])
                X_scaled = self.scalers[feature].transform(X_pca[cols])
                self.pca_models[feature] = PCA(n_components=n_comp).fit(X_scaled)

        return self

    def transform(self, X):
        df = X.copy()
        df = self._do_moisture_layers(df)
        df = self._do_temp_layers(df)
        return df

    def fit_transform(self, X):
        self.fit(X)
        df = self.transform(X)
        return df

    def _do_moisture_layers(self, X):
        moisture_cols = [f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]
        required = list({"Timestamp", "Growth_Stage"}) + moisture_cols
        self._check_required_cols(cols=X.columns, required=required, hint=True)

        vw_df = X[moisture_cols].copy()
        vw_df[list(required)] = X[list(required)].copy()

        contains_dates = True
        if "doy" not in X.columns:
            vw_df = utils.add_date_info(vw_df)
            contains_dates = False
        else:
            vw_df[["doy"]] = X[["doy"]]

        if self.root_weighted_moisture:
            vw_df = vw_df.join(self._compute_root_weighted_moisture(vw_df))
        if self.moisture_pca:
            vw_df = vw_df.join(self._compute_pca(vw_df, self.SOIL_MOISTURE))

        if not contains_dates:
            vw_df = vw_df.drop(["doy", "year"], axis=1)

        drop_cols = list(required) + moisture_cols
        return X.join(vw_df.drop(columns=drop_cols))

    def _do_temp_layers(self, X):
        temp_cols = [f"soil_temperature_level_{i}" for i in range(1, 5)]
        self._check_required_cols(cols=X.columns, required=set(temp_cols), hint=False)

        st_df = X[temp_cols].copy()
        if self.temperature_pca:
            st_df = st_df.join(self._compute_pca(st_df, self.SOIL_TEMP))
        return X.join(st_df.drop(columns=temp_cols))

    def _compute_root_weighted_moisture(self, X):
        moisture_cols = [f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]
        moisture = X[moisture_cols].to_numpy()

        weights = np.empty(shape=(len(X), 4), dtype=float)
        for i, stage in enumerate(X["Growth_Stage"]):
            weights[i] = self.ROOT_WEIGHTS[stage]

        root_weighted_moisture = np.sum(
            moisture * weights,
            axis=1,
        )

        rwm_df = pd.DataFrame(
            root_weighted_moisture, columns=["root_weighted_soil_moisture"]
        )

        return rwm_df

    def _compute_pca(self, X, measure):
        pca = self.pca_models[measure]
        scaler = self.scalers[measure]
        cols = self.pca_config.get(measure)["columns"]

        prefix = "VW" if measure == self.SOIL_MOISTURE else "ST"

        X_scaled = scaler.transform(X[cols])
        X_pca = pca.transform(X_scaled)
        pca_df = pd.DataFrame(
            X_pca,
            columns=[f"{prefix}_PC{i + 1}" for i in range(pca.n_components_)],
            index=X.index,
        )

        return pca_df

    def _check_required_cols(self, cols, required, hint=False):
        missing = set(required) - set(cols)

        if missing:
            if hint:
                raise ValueError(
                    f"Missing required columns: {missing}. For growth stages, run PhenologyFeatures(growth_stages=True) first."
                )
            else:
                raise ValueError(f"Missing required columns: {missing}. ")
