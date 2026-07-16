import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from models import LikelihoodGMM


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

    def fit(self, X):
        self.feature_names_in_ = X.columns
        X_embed = self._delay_embed(X)

        super().fit(X_embed)
        return self

    def forecast(self, X, responses, rescale=True):
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
        if rescale:
            forecasts = self._rescale_forecasts(responses, forecasts)

        return pd.DataFrame(forecasts, columns=responses)

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
        future_mask[self.window_size - self.horizon :, response_indices] = True
        return future_mask.flatten()

    def _get_past_mask(self, X_forecast):
        past_mask = np.full(
            shape=(self.window_size, X_forecast.shape[1]), fill_value=False, dtype=bool
        )
        past_mask[: self.window_size - self.horizon, :] = True
        return past_mask.flatten()

    def _calculate_past_mixture_coeff(self, past_mask, future_mask, x_past):
        log_components = np.empty(shape=self.n_components)

        if self.covariance_type == "tied":
            cov_PP = self.model.covariances_[np.ix_(past_mask, past_mask)]

        for k in range(self.n_components):
            if self.covariance_type == "full":
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

        if self.covariance_type == "tied":
            cov_PP = self.model.covariances_[np.ix_(past_mask, past_mask)]
            cov_FP = self.model.covariances_[np.ix_(future_mask, past_mask)]

        for k in range(self.n_components):
            if self.covariance_type != "tied":
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

        temp = np.zeros(shape=(forecasts.shape[0], len(self.scaler.feature_names_in_)))
        temp[:, indices] = forecasts
        unscaled = self.scaler.inverse_transform(temp)
        return unscaled[:, indices]


class ForecastingGMMSelector:
    def __init__(
        self,
        window_size,
        horizon,
        responses,
        scaler_splitter,
        covariance_type="full",
        max_iter=100,
        reg_covar=1e-06,
        random_state=42,
        n_init=1,
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.responses = list(responses)

        self.scaler_splitter = scaler_splitter

        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.n_init = n_init

        self._test_mse_scores = None
        self._test_mae_scores = None
        self._r2_scores = None
        self._components = None

        self.best_model_ = None
        self.best_components_ = None

    def cv_fit(self, scaled_X, scaled_validation, components):

        # reg_covar = reg_covar if reg_covar is not None else self.reg_covar
        self._test_mse_scores = np.zeros(shape=(len(components)))
        self._test_mae_scores = np.zeros(shape=len(components))
        self._r2_scores = np.zeros(shape=len(components))
        self._components = components

        models = []

        for comp_i, k in enumerate(components):
            gmm = ForecastingGMM(
                window_size=self.window_size,
                scaler=self.scaler_splitter.scaler,
                horizon=self.horizon,
                n_components=k,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            gmm.fit(scaled_X)
            models.append(gmm)

            mse_sum = 0
            mae_sum = 0

            eval_count = 0

            final_start_i = len(scaled_validation) - gmm.window_size

            for i in range(final_start_i):
                slide = scaled_validation.iloc[i : i + gmm.window_size]
                x_past = slide.iloc[: -gmm.horizon, :]
                scaled_true_vals = slide.iloc[-gmm.horizon :, :]

                pred = gmm.forecast(x_past, responses=self.responses, rescale=True)
                target_scaled_true_vals = scaled_true_vals[self.responses]

                target_true = self.scaler_splitter.inverse_transform_partial(
                    target_scaled_true_vals
                )
                mse_sum += mean_squared_error(target_true, pred)
                mae_sum += mean_absolute_error(target_true, pred)

                eval_count += 1
            self._test_mse_scores[comp_i] = (
                mse_sum / eval_count if eval_count > 0 else np.inf
            )
            self._test_mae_scores[comp_i] = (
                mae_sum / eval_count if eval_count > 0 else np.inf
            )

        best_i = np.argmin(self._test_mse_scores)
        self.best_components_ = components[best_i]
        self.best_model_ = models[best_i]
        best_mse = self._test_mse_scores[best_i]

        print(f"components:  {self.best_components_}")
        print(f"test MSE:  {best_mse:.6f}")
        print(f"test RMSE: {np.sqrt(best_mse):.6f}")
        print(f"test MAE:  {self._test_mae_scores[best_i]:.6f}")

    def plot_test_mse(self):
        plt.figure(figsize=(10, 4))
        plt.title("Test MSE vs. Number of Components (k)")
        plt.plot(self._components, self._test_mse_scores)
        plt.xticks(self._components)
        plt.xlabel("Number of Mixture Components (k)")
        plt.ylabel("Test MSE")
        plt.show()
