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
