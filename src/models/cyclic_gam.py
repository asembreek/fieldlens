from pygam import LinearGAM, s
import numpy as np


class NDVIClimatologyGAM:
    x_ticks = np.arange(1, 366, 30)
    doys = np.arange(1, 366)

    ERR_NOT_FITTED = (
        "The model has not been fitted. Call 'fit()' or 'gridsearch()' first."
    )

    def __init__(self, n_splines=10, plot_shift=None):
        self.n_splines = n_splines

        self.model = None
        self.sos = None
        self.eos = None

    @property
    def climatology(self):
        self._check_is_fitted()
        return self.model.predict(self.doys.reshape(-1, 1))

    @property
    def dy_dx(self):
        self._check_is_fitted()
        return np.gradient(self.climatology, self.doys)

    @property
    def d2y_dx2(self):
        self._check_is_fitted()
        return np.gradient(self.dy_dx, self.doys)

    @property
    def is_fitted(self):
        return self.model is not None

    def fit(self, doys, ndvi):
        self.model = LinearGAM(s(0, basis="cp", n_splines=self.n_splines))
        self.model.fit(doys, ndvi)

        return self

    def gridsearch(self, doys, ndvi, lams=np.logspace(-5, 5, 25)):
        self.model = LinearGAM(s(0, basis="cp", n_splines=self.n_splines))
        self.model.gridsearch(doys, ndvi, lams)
        return self

    def summary(self):
        pass

    def plot_derivatives(self, shifted=True):
        self._check_is_fitted()

    def plot_climatology(self, shifted=True):
        self._check_is_fitted()
        pass

    def _set_sos_eos(self):
        pass

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(self.ERR_NOT_FITTED)

    @staticmethod
    def shift_doy(d, shift):
        shifted_doy = ((d - shift) % 365) + 1
        return shifted_doy

    @staticmethod
    def inv_shift_doy(d, shift):
        inv_shifted_doy = (d + shift - 1) % 365
        return inv_shifted_doy
