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

    def __init__(
        self, n_splines=10, reference_shift=pd.Timestamp("2019-08-01").dayofyear
    ):
        self.n_splines = n_splines

        self.model = None
        self.reference_shift_ = reference_shift

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
        shifted = self.shift_doy(self.doys, self.reference_shift_)
        return self.model.predict(shifted.reshape(-1, 1))

    @property
    def lam(self):
        self._check_is_fitted()
        return self.model.terms[0].lam[0]

    @property
    def confidence_intervals(self):
        self._check_is_fitted()
        shifted = self.shift_doy(self.doys, self.reference_shift_)
        return self.model.confidence_intervals(shifted.reshape(-1, 1))

    @property
    def dy_dx(self):
        self._check_is_fitted()
        shifted = self.shift_doy(self.doys, self.reference_shift_)
        return np.gradient(self.climatology, shifted)

    @property
    def d2y_dx2(self):
        self._check_is_fitted()
        shifted = self.shift_doy(self.doys, self.reference_shift_)
        return np.gradient(self.dy_dx, shifted)

    @property
    def is_fitted(self):
        return self.model is not None

    @property
    def sos_interval(self):
        return (int(self._zero_crossing), int(self._first_peak))

    # Shifting methods

    @staticmethod
    def shift_doy(d, shift):
        shifted_doy = ((d - shift) % 365) + 1
        return shifted_doy

    @staticmethod
    def inv_shift_doy(d, shift):
        inv_shifted_doy = (d + shift - 1) % 365
        return inv_shifted_doy

    def fit(self, doys, ndvi):
        self.model = LinearGAM(s(0, basis="cp", n_splines=self.n_splines))
        shifted_doys = self.shift_doy(doys, self.reference_shift_)
        self.model.fit(shifted_doys, ndvi)
        self._doys = doys
        self._ndvi = ndvi
        self._calculate_sos_eos()

        return self

    def gridsearch(self, doys, ndvi, lams=np.logspace(-6, 6, 25)):
        self.model = LinearGAM(s(0, basis="cp", n_splines=self.n_splines))
        shifted_doys = self.shift_doy(doys, self.reference_shift_)
        self.model.gridsearch(shifted_doys, ndvi, lam=lams)
        self._doys = doys
        self._ndvi = ndvi
        self._calculate_sos_eos()

        return self

    def predict(self, doy, shifted=True):
        shifted_doy = self.shift_doy(doy, self._sos)
        pred = self.model.predict(shifted_doy)
        return pred

    def start_of_season(self, position="start"):
        self._sos = self._select_interval_point(
            left=self._zero_crossing, right=self._first_peak, position=position
        )
        return self._sos

    def end_of_season(self):
        return self._eos

    def plot_derivatives(self, shifted=True, crit_lines=True):
        self._check_is_fitted()
        if shifted:
            shift = self._sos
        else:
            shift = 0

        x_labels = self.inv_shift_doy(self.x_ticks, shift)
        plot_doys = self.shift_doy(self.doys.flatten(), shift)

        order = np.argsort(plot_doys)
        plot_doys = plot_doys[order]
        dy_dx = self.dy_dx[order]
        d2y_dx2 = self.d2y_dx2[order]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        ax[0].set_title("GAM First Derivative")
        ax[0].axhline(y=0, color="red", linestyle="--")
        ax[0].set_xticks(self.x_ticks)
        ax[0].set_xticklabels(x_labels)
        ax[0].set_xlabel("Day of Year")
        ax[0].plot(plot_doys, dy_dx)

        ax[1].set_title("GAM Second Derivative")
        ax[1].axhline(y=0, color="red", linestyle="--")
        ax[1].set_xticks(self.x_ticks, labels=x_labels)
        ax[1].set_xlabel("Day of Year")
        ax[1].plot(plot_doys, d2y_dx2)

        if crit_lines:
            decay_line = self.shift_doy(self._decay_start, shift)
            zero_line = self.shift_doy(self._zero_crossing, shift)
            first_peak_line = self.shift_doy(self._first_peak, shift)
            second_peak_line = self.shift_doy(self._second_peak, shift)

            ax[0].axvline(
                x=decay_line,
                linestyle=":",
                alpha=0.6,
                color="brown",
                label="Decay Start",
            )

            ax[0].axvline(
                x=zero_line,
                linestyle=":",
                alpha=0.6,
                color="green",
                label="Zero Crossing (SOS start)",
            )
            ax[0].legend()

            ax[1].axvline(
                x=first_peak_line,
                linestyle=":",
                alpha=0.6,
                color="green",
                label="1st Peak (SOS End)",
            )
            ax[1].axvline(
                x=second_peak_line,
                linestyle=":",
                alpha=0.6,
                color="brown",
                label="2nd Peak (EOS)",
            )
            ax[1].legend()

        plt.tight_layout()
        plt.show()

    def plot_climatology(self, shifted=True, season_lines=False):
        if shifted:
            shift = self._sos
        else:
            shift = 0

        x_labels = self.inv_shift_doy(self.x_ticks, shift)
        fitted_doys = self.shift_doy(self._doys.flatten(), shift)
        plot_doys = self.shift_doy(self.doys.flatten(), shift)

        order = np.argsort(plot_doys)
        plot_doys = plot_doys[order]
        climatology = self.climatology[order]
        ci = self.confidence_intervals[order]

        plt.figure(figsize=(10, 5))
        plt.title(f"NDVI Seasonal Cycle (λ={self.lam:.2f}) ")

        plt.scatter(
            x=fitted_doys,
            y=self._ndvi,
            alpha=0.3,
            s=5,
            color="grey",
        )

        plt.plot(
            plot_doys,
            climatology,
            linewidth=2,
            color="red",
            label="GAM climatology",
        )
        plt.fill_between(
            plot_doys,
            ci[:, 0],
            ci[:, 1],
            alpha=0.4,
            label="95% CI",
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

    def summary(self):
        self._check_is_fitted()
        lines = []
        model_rows = [("Observations", len(self._ndvi)), ("Response", "NDVI")]
        event_rows = [
            ("Start-of-Season interval", self.sos_interval),
            ("Selected SOS DOY", self._sos),
            ("Peak Growth", "TODO"),
            ("Senescence DOY", self._eos),
        ]
        metric_rows = [
            ("Season Length", int(365 - np.abs(self._sos - self._eos))),
            ("Peak NDVI", np.max(self._ndvi)),
            ("Base NDVI", np.min(self._ndvi)),
            ("Average NDVI", np.average(self._ndvi)),
        ]

        model_heading = f"ClimatologyGAM\n{'─' * 60}"
        event_heading = f"Phenological Events\n{'─' * 60}"
        metric_heading = f"Seasonal Metrics\n{'─' * 60}"

        lines.append(model_heading)
        for name, value in model_rows:
            lines.append(f"{name:<30} {value}")
        lines.append(f"\n{event_heading}")
        for name, value in event_rows:
            lines.append(f"{name:<30} {value}")

        lines.append(f"\n{metric_heading}")
        for name, value in metric_rows:
            formatted = f"{value:.2f}" if isinstance(value, float) else str(value)
            lines.append(f"{name:<30} {formatted}")

        print("\n".join(lines))

    def _calculate_sos_eos(self):
        dy_dx = self.dy_dx
        d2y_dx2 = self.d2y_dx2

        self._decay_start = self._calculate_decay_doy(dy_dx)
        self._zero_crossing = self._calculate_zero_crossing(dy_dx)
        self._first_peak = self._calculate_first_peak(self._decay_start, d2y_dx2)
        self._second_peak = self._calculate_second_peak(self._decay_start, d2y_dx2)

        self._sos = self._select_interval_point(
            left=self._zero_crossing, right=self._first_peak, position="start"
        )
        self._eos = self._second_peak

    def _calculate_decay_doy(self, dy_dx):
        for i in range(len(dy_dx) - 1):
            if dy_dx[i] > 0 and dy_dx[i + 1] < 0:
                return self.doys[i]

    def _calculate_zero_crossing(self, dy_dx):
        for i in range(len(dy_dx) - 1):
            if dy_dx[i] < 0 and dy_dx[i + 1] > 0:
                return self.doys[i]

    def _calculate_first_peak(self, decay_start_doy, d2y_dx2):
        max_deriv = -np.inf
        first_peak = None
        for i in range(len(d2y_dx2)):
            true_doy = self.doys[i]
            if (true_doy > self.inv_shift_doy(1, self.reference_shift_)) or (
                true_doy < decay_start_doy
            ):
                if d2y_dx2[i] > max_deriv:
                    max_deriv = d2y_dx2[i]
                    first_peak = true_doy
        return first_peak

    def _calculate_second_peak(self, decay_start_doy, d2y_dx2):
        max_deriv = -np.inf
        second_peak = None

        for i in range(len(d2y_dx2)):
            true_doy = self.doys[i]

            if true_doy > decay_start_doy and true_doy < self.inv_shift_doy(
                1, self.reference_shift_
            ):
                if d2y_dx2[i] > max_deriv:
                    max_deriv = d2y_dx2[i]
                    second_peak = true_doy
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
