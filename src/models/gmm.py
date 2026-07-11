class BaseGMM:
    def __init__(self, n_components, max_iter, reg_cov, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.reg_cov = reg_cov
        self.is_fitted = False

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X):
        pass


class LikelihoodGMM(BaseGMM):
    pass


class ForecastingGMM(BaseGMM):
    pass
