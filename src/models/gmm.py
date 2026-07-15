import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error

from scipy.stats import multivariate_normal
from scipy.special import logsumexp


class LikelihoodGMM:
    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        max_iter=100,
        reg_covar=1e-06,
        random_state=None,
        n_init=1,
    ):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.n_init = n_init

        self.is_cv_fitted = False

        self._bic_scores = None
        self._covariance_types = None
        self._components = None

    @property
    def is_fitted(self):
        return self.model is not None

    def fit(self, X):
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        self.model.fit(X)
        self.is_cv_fitted = False

        return self

    def score(self, X):
        return self.model.score(X)

    def score_samples(self, X):
        return self.model.score_samples(X)

    def bic(self, X):
        return self.model.bic(X)

    def cv_fit(
        self,
        X,
        components,
        covariance_types=("full", "tied", "diag", "spherical"),
        n_init=10,
        reg_covar=None,
    ):
        self._bic_scores = np.zeros(shape=(len(covariance_types), len(components)))

        best_bic = np.inf
        best_model = None
        best_covariance = None
        best_components = None
        reg_covar = self.reg_covar if reg_covar is None else reg_covar

        for i, cov in enumerate(covariance_types):
            for j, k in enumerate(components):
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    max_iter=self.max_iter,
                    reg_covar=reg_covar,
                    random_state=self.random_state,
                    n_init=n_init,
                )

                gmm.fit(X)
                bic = gmm.bic(X)
                self._bic_scores[i, j] = bic
                if bic < best_bic:
                    best_bic = bic
                    best_model = gmm
                    best_covariance = cov
                    best_components = k

        self.model = best_model
        self.n_components = best_components
        self.covariance_type = best_covariance
        self.reg_covar = reg_covar
        self.n_init = n_init

        self.is_cv_fitted = True

        self._covariance_types = covariance_types
        self._components = components

        print(
            f"Minimum BIC: {best_bic:.2f} "
            f"with {best_components} components "
            f"and '{best_covariance}' covariance."
        )

        return self

    def save(self):
        pass

    def load(self):
        pass

    def plot_bic_curve(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Run 'cv_fit' first.")
        if not self.is_cv_fitted:
            raise RuntimeError(
                "Model not fitted with Cross Validation. Run 'cv_fit' first."
            )

        plt.figure(figsize=(10, 6))
        for i, cov in enumerate(self._covariance_types):
            plt.plot(
                self._components,
                self._bic_scores[i],
                label=f"{cov} covariance",
                marker="o",
            )

        plt.xlabel("Number of Components (k)")
        plt.ylabel("BIC Score")
        plt.title("GMM Model Selection via BIC")
        plt.axvline(
            x=self.n_components,
            color="red",
            linestyle="--",
            label=f"Best k ({self.n_components})",
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_yearly_likelihood(self):
        pass

    def plot_likelihood_kde(self, *X, labels):
        n_graphs = len(X)
        if len(labels) != n_graphs:
            raise ValueError(
                "Ensure number of labels matches number of passed datasets."
            )

        plt.figure(figsize=(10, 6))
        plt.title(f"Log-likelihood densities for {n_graphs} datasets.")
        for i in range(n_graphs):
            scores = self.score_samples(X[i])
            sns.kdeplot(
                scores,
                label=labels[i],
                fill=True,
                alpha=0.5,
            )
        plt.ylabel("Density")
        plt.xlabel("log-Likelihood")
        plt.tight_layout()
        plt.legend()


class ForecastingGMM(LikelihoodGMM):
    def __init__(
        self,
        window_size,
        scaler,
        horizon=1,
        n_components=1,
        covariance_type="full",
        max_iter=100,
        reg_covar=1e-06,
        random_state=None,
        n_init=1,
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            reg_covar=reg_covar,
            random_state=random_state,
            n_init=n_init,
        )
        self.window_size = window_size
        self.horizon = horizon
        self.scaler = scaler
        self.feature_names_in_ = None

        self._mse_scores = None

    def fit(self, X):
        self.feature_names_in_ = X.columns
        X_embed = self._delay_embed(X)
        super().fit(X_embed)
        return self

    def forecast(self, X, responses):
        if X.shape[0] < self.horizon:
            raise ValueError(
                f"Ensure X has a row count greater than forecast horizon {self.horizon}"
            )
        for r in list(responses):
            if r not in X.columns:
                raise ValueError(f"{r} not a valid column in passed dataset.")
        response_indices = np.array([X.columns.get_loc(r) for r in responses])

        X_forecast = X.to_numpy()
        pred_window = self._get_pred_window(X_forecast)
        future_mask = self._get_future_mask(X_forecast, response_indices)
        past_mask = self._get_past_mask(X_forecast)

        x_past = pred_window[0, past_mask]

        posteriors = self._calculate_past_mixture_coeff(past_mask, future_mask, x_past)
        cond_exp = self._calculate_conditional_exp(past_mask, future_mask, x_past)
        y_hat = np.dot(posteriors, cond_exp)
        forecasts = y_hat.reshape(self.horizon, len(responses))
        unscaled_forecasted = self._rescale_forecasts(responses, forecasts)

        return pd.DataFrame(unscaled_forecasted, columns=responses)

    def cv_fit(
        self,
        X,
        validation,
        components,
    ):

        # reg_covar = reg_covar if reg_covar is not None else self.reg_covar
        self._mse_scores = np.zeros(shape=(len(components)))

        for comp_i, k in enumerate(components):
            gmm = ForecastingGMM(
                window_size=self.window_size,
                scaler=self.scaler,
                horizon=12,
                n_components=k,
                covariance_type="full",
                max_iter=self.max_iter,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            gmm.fit(X)

            mse_sum = 0
            eval_count = 0

            final_start_i = len(validation) - gmm.window_size

            for i in range(final_start_i):
                slide = validation.iloc[i : i + gmm.window_size]
                x_past = slide.iloc[: -gmm.horizon, :]
                true_vals = slide.iloc[-gmm.horizon :, :]

                pred = gmm.forecast(x_past, responses=["NDVI_mean"])
                target_true = true_vals["NDVI_mean"]

                mse_sum += mean_squared_error(target_true, pred)
                eval_count += 1
            self._mse_scores[comp_i] = (
                mse_sum / eval_count if eval_count > 0 else np.inf
            )
        best_k_i = np.argmin(self._mse_scores)
        best_k = components[best_k_i]
        best_mse = self._mse_scores[best_k_i]
        print(f"components:  {best_k}")
        print(f"train MSE:  {best_mse:.6f}")

    def _delay_embed(self, X):
        X = np.asarray(X)

        n, p = X.shape

        embedded = np.empty(
            shape=(n - self.window_size + 1, p * self.window_size), dtype=X.dtype
        )

        for i in range(n - self.window_size + 1):
            embedded[i] = X[i : i + self.window_size].reshape(-1)

        return embedded

    def _get_pred_window(self, X_forecast):
        lookback = self.window_size - self.horizon
        known_obs = X_forecast[-lookback:]
        temp = np.full((self.horizon, X_forecast.shape[1]), np.nan)
        return np.vstack([known_obs, temp]).reshape(1, -1)

    def _get_future_mask(self, X_forecast, response_indices):
        """
        Creates mask that only selects features defined in 'responses' in 'self.forecast()'. Avoids predicting unnecessary predictors.
        """

        future_mask = np.full(
            shape=(self.window_size, X_forecast.shape[1]), fill_value=False, dtype=bool
        )
        future_mask[self.horizon :, response_indices] = True
        return future_mask.flatten()

    def _get_past_mask(self, X_forecast):
        past_mask = np.full(
            shape=(self.window_size, X_forecast.shape[1]), fill_value=False, dtype=bool
        )
        past_mask[: self.horizon, :] = True
        return past_mask.flatten()

    def _calculate_past_mixture_coeff(self, past_mask, future_mask, x_past):
        log_components = np.empty(shape=self.n_components)
        for k in range(self.n_components):
            cov_PP = self.model.covariances_[k][np.ix_(past_mask, past_mask)]
            mu_P = self.model.means_[k, past_mask]

            log_pdf = multivariate_normal(mu_P, cov_PP, allow_singular=True).logpdf(
                x_past
            )
            log_components[k] = np.log(self.model.weights_[k]) + log_pdf

        log_const = logsumexp(log_components)
        posteriors = np.exp(log_components - log_const)
        return posteriors

    def _calculate_conditional_exp(self, past_mask, future_mask, x_past):

        # matrix/vector of 'predicted' y_i corresponding to component k
        y_ik = np.empty((self.n_components, future_mask.sum()))

        for k in range(self.n_components):
            cov_PP = self.model.covariances_[k][np.ix_(past_mask, past_mask)]
            cov_FP = self.model.covariances_[k][np.ix_(future_mask, past_mask)]

            mu_P = self.model.means_[k, past_mask]
            mu_F = self.model.means_[k, future_mask]

            resid = x_past - mu_P
            weights = np.linalg.solve(cov_PP, resid)

            y_ik[k, :] = mu_F + cov_FP @ weights

        return y_ik

    def _rescale_forecasts(self, responses, forecasts):
        indices = []

        for i, s in enumerate(self.scaler.feature_names_in_):
            if s in responses:
                indices.append(i)

        temp = np.zeros(
            shape=(self.window_size - self.horizon, len(self.scaler.feature_names_in_))
        )
        temp[:, indices] = forecasts
        unscaled = self.scaler.inverse_transform(temp)
        return unscaled[:, indices]
