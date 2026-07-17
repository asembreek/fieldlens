class FeaturePipeline:
    def __init__(self, *transformers):
        self.transformers = transformers
        self._fitted_transformers = []

    def fit(self, X):
        for t in self.transformers:
            transformer = t.fit(X)
            self._fitted_transformers.append(transformer)

    def transform(self, X):
        X_trans = X.copy()
        for t in self._fitted_transformers:
            X_trans = t.transform(X_trans)
        return X_trans

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
