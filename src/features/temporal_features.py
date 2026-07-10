import numpy as np
from data import utils
from models import NDVIClimatologyGAM as shifter


class TemporalFeatures:
    def __init__(self, start_of_season, cyclic_doy=False):
        self.cyclic_doy = cyclic_doy
        self.start_of_season = start_of_season

    def fit(self, X):
        return self

    def transform(self, X):
        df = X.copy()
        if self.cyclic_doy:
            required = {"doy", "Timestamp"}
            missing = required - set(X.columns)
            if len(missing) == 2:
                raise ValueError(
                    "Cyclic encoding requires one of {required} as columns."
                )

            missing_date = False

            if "doy" in missing:
                df = utils.add_date_info(df)
                missing_date = True

            df["shifted_doy"] = shifter.shift_doy(df["doy"], self.start_of_season)

            df = self._add_cyclic_doy(df)
            if missing_date:
                df = df.drop(["doy", "year"], axis=1)

            df = df.drop("shifted_doy", axis=1)
        return df

    def fit_transform(self, X):
        return self.transform(X)

    def _add_cyclic_doy(self, X):
        X["doy_sin"] = np.sin(2 * np.pi * X["shifted_doy"] / 365.25)
        X["doy_cos"] = np.cos(2 * np.pi * X["shifted_doy"] / 365.25)
        return X
