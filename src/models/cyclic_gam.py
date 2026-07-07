from pygam import LinearGAM, s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NDVIClimatologyGAM:
    x_ticks = np.arange(1, 366, 30)
    doys = np.arange(1, 366)

    ERR_NOT_FITTED = (
        "The model has not been fitted. Call 'fit()' or 'gridsearch()' first."
    )
    ERR_INTERVAL = "`position` must either be of type float in interval [0.0, 1.0], or integer equal to 1 or 0, or one of strings in ['start', 'early', 'middle', 'late', 'end']."

    _INTERVAL_POSITIONS = {
        "start": 0.0,
        "early": 0.25,
        "middle": 0.5,
        "late": 0.75,
        "end": 1.0,
    }

    def __init__(self, n_splines=10):
        self.n_splines = n_splines

        self.model = None
        self._doys = None
        self._ndvi = None

        # Derived DOYS
        self._sos = None
        self._eos = None
        self._decay_start = None
        self._zero_crossing = None
        self._first_peak = None
        self._second_peak = None

    @property
    def climatology(self):
        self._check_is_fitted()
        return self.model.predict(self.doys.reshape(-1, 1))

    @property
    def lam(self):
        self._check_is_fitted()
        return self.model.terms[0].lam[0]

    @property
    def confidence_intervals(self):
        self._check_is_fitted()
        return self.model.confidence_intervals(self.doys.reshape(-1, 1))

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

    @property
    def sos_interval(self):
        return (self._zero_crossing, self._first_peak)

    def fit(self, doys, ndvi):
        self.model = LinearGAM(s(0, basis="cp", n_splines=self.n_splines))
        self.model.fit(doys, ndvi)
        self._doys = doys
        self._ndvi = ndvi
        self._calculate_sos_eos()

        return self

    def gridsearch(self, doys, ndvi, lams=np.logspace(-5, 5, 25)):
        self.model = LinearGAM(s(0, basis="cp", n_splines=self.n_splines))
        self.model.gridsearch(doys, ndvi, lam=lams)
        self._doys = doys
        self._ndvi = ndvi
        self._calculate_sos_eos()

        return self

    def start_of_season(self, position="start"):
        return self._select_interval_point(
            left=self._zero_crossing, right=self._first_peak, position=position
        )

    def end_of_season(self):
        return self._eos

    def plot_derivatives(self, shifted=True, crit_lines=True):
        self._check_is_fitted()
        if shifted:
            shift = self._sos
        else:
            shift = 0

        x_labels = self.shift_doy(self.x_ticks, shift)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        ax[0].set_title("GAM First Derivative")
        ax[0].axhline(y=0, color="red", linestyle="--")
        ax[0].set_xticks(self.x_ticks, labels=x_labels)
        ax[0].set_xlabel("Day of Year")
        ax[0].plot(self.dy_dx)

        ax[1].set_title("GAM Second Derivative")
        ax[1].axhline(y=0, color="red", linestyle="--")
        ax[1].set_xticks(self.x_ticks, labels=x_labels)
        ax[1].set_xlabel("Day of Year")
        ax[1].plot(self.d2y_dx2)

        if crit_lines:
            ax[0].axvline(
                x=self.shift_doy(self._decay_start, shift),
                linestyle=":",
                alpha=0.6,
                color="brown",
            )

            ax[0].axvline(
                x=self.shift_doy(self._zero_crossing, shift),
                linestyle=":",
                alpha=0.6,
                color="green",
            )

            ax[1].axvline(
                x=self.shift_doy(self._first_peak, shift),
                linestyle=":",
                alpha=0.6,
                color="brown",
            )
            ax[1].axvline(
                x=self.shift_doy(self._second_peak, shift),
                linestyle=":",
                alpha=0.6,
                color="green",
            )

        plt.tight_layout()
        plt.show()

    def plot_climatology(self, shifted=True, season_lines=False):
        ci = self.confidence_intervals

        if shifted:
            shift = self._sos
        else:
            shift = 0

        x_labels = self.shift_doy(self.x_ticks, shift)

        plot_df = pd.DataFrame(
            {
                "doy": self._doys.flatten(),
                "ndvi": self._ndvi,
            }
        )

        plot_df["shifted_doy"] = self.shift_doy(self._doys, shift)

        plt.figure(figsize=(10, 5))
        plt.title(f"NDVI Seasonal Cycle (λ={self.lam:.2f}) ")

        plt.plot(
            self.doys.reshape(-1, 1),
            self.climatology,
            linewidth=2,
            color="red",
            label="GAM climatology",
        )
        plt.fill_between(
            self.doys.reshape(-1, 1),
            ci[:, 0],
            ci[:, 1],
            alpha=0.4,
            label="95% CI",
        )

        plt.scatter(
            x=plot_df["shifted_doy"],
            y=plot_df["ndvi"],
            alpha=0.3,
            s=5,
            color="grey",
        )

        plt.ylabel("NDVI")

        plt.xticks(self.x_ticks, labels=x_labels)
        plt.xlabel("Day of Year")

        if season_lines:
            left_sos = self.shift_doy(self._zero_crossing, shift)
            right_sos = self.shift_doy(self._first_peak, shift)

            plt.axvspan(
                left_sos, right_sos, color="green", alpha=0.2, label="SOS interval"
            )
            plt.axvline(
                x=left_sos,
                linestyle=":",
                alpha=0.6,
                color="green",
            )

            plt.axvline(
                x=right_sos,
                linestyle=":",
                alpha=0.6,
                color="green",
            )

            plt.axvline(
                x=self.shift_doy(self._second_peak, shift),
                linestyle=":",
                alpha=0.6,
                color="brown",
                label="EOS",
            )

        plt.legend()
        plt.tight_layout()
        plt.show()

    def _calculate_sos_eos(self):
        dy_dx = self.dy_dx
        d2y_dx2 = self.d2y_dx2

        self._decay_start = self._calculate_decay_doy(dy_dx)

        # Negative-to-positive crossing of first derivative
        self._zero_crossing = self._calculate_zero_crossing(dy_dx)

        # First and second peaks of 2nd derivative
        self._first_peak = self._calculate_first_peak(self._decay_start, d2y_dx2)
        self._second_peak = self._calculate_second_peak(self._decay_start, d2y_dx2)
        self._sos = self._select_interval_point(
            left=self._zero_crossing, right=self._first_peak, position="start"
        )

        self._eos = self._second_peak

    def _calculate_decay_doy(self, dy_dx):
        for i in range(len(dy_dx) - 1):
            if dy_dx[i] > 0 and dy_dx[i + 1] < 0:
                return i

    def _calculate_zero_crossing(self, dy_dx):
        for i in range(len(dy_dx) - 1):
            if dy_dx[i] < 0 and dy_dx[i + 1] > 0:
                return i

    def _calculate_first_peak(self, decay_start_doy, d2y_dx2):
        max_deriv = -np.inf
        first_peak = None
        for i in range(len(d2y_dx2) - 1):
            if i > 1 or i < decay_start_doy:
                if d2y_dx2[i] > max_deriv:
                    max_deriv = d2y_dx2[i]
                    first_peak = i
        return first_peak

    def _calculate_second_peak(self, decay_start_doy, d2y_dx2):
        max_deriv = -np.inf
        second_peak = None

        for i in range(len(d2y_dx2) - 1):
            if i > decay_start_doy and i < 1:
                if d2y_dx2[i] > max_deriv:
                    max_deriv = d2y_dx2[i]
                    second_peak = i
        return second_peak

    def _select_interval_point(self, left, right, position):
        if isinstance(position, str):
            if position not in self._INTERVAL_POSITIONS:
                raise ValueError(self.ERR_INTERVAL)
            position = self._INTERVAL_POSITIONS[position]

        elif isinstance(position, (int, float)):
            if not 0.0 <= position <= 1.0:
                raise ValueError(self.ERR_INTERVAL)
        else:
            raise TypeError(self.ERR_INTERVAL)

        return round(left + position * (right - left))

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(self.ERR_NOT_FITTED)

    # Shifting methods

    @staticmethod
    def shift_doy(d, shift):
        shifted_doy = ((d - shift) % 365) + 1
        return shifted_doy

    @staticmethod
    def inv_shift_doy(d, shift):
        inv_shifted_doy = (d + shift - 1) % 365
        return inv_shifted_doy

    def summary(self):
        model_rows = []
        event_rows = []
        metric_rows = []
