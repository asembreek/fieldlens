import numpy as np
import pandas as pd
from models import NDVIClimatologyGAM as shift
from data import utils


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
