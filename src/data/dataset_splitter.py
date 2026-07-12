import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DatasetScalerSplitter:
    def __init__(self, train_mask, test_mask, passthrough_cols=None, drop_cols=None):
        self.passthrough_cols = passthrough_cols
        self.drop_cols = drop_cols

        self.train_mask = train_mask
        self.test_mask = test_mask

        self.scaler = None
        self.features_in_ = None

    def fit(self, X):
        X_fit = X.loc[self.train_mask].copy()
        X_fit = X_fit.select_dtypes(include=[np.number])

        for col in self.drop_cols:
            if col in X_fit.columns:
                X_fit = X_fit.drop(col, axis=1)

        for col in self.passthrough_cols:
            if col in X_fit.columns:
                X_fit = X_fit.drop(col, axis=1)

        self.features_in_ = X_fit.columns
        self.scaler = StandardScaler()
        self.scaler.fit(X_fit)
        return self

    def transform(self, X):
        X = X.copy()

        X[self.features_in_] = self.scaler.transform(X[self.features_in_])
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def train_test_split_transform(self, X):
        X_train = X.loc[:, self.train_mask]
        X_test = X.loc[:, self.test_mask]

        X_train_scaled = self.transform(X_train)
        X_test_scaled = self.transform(X_test)

        X_train_df = pd.DataFrame(
            X_train_scaled, columns=self.features_in_, index=X_train.index
        )
        X_test_df = pd.DataFrame(
            X_test_scaled, columns=self.features_in_, index=X_test.index
        )
        X_train_df = pd.concat(
            [X_train_df, X[self.train_mask, self.passthrough_cols]], axis=1
        )
        X_test_df = pd.concat(
            [X_test_df, X[self.test_mask, self.passthrough_cols]], axis=1
        )

        return X_train_df, X_test_df

    def fit_train_test_split_transform(self, X):
        self.fit(X)
        return self.train_test_split_transform(X)

    def _remove_columns(self, X, columns):
        X = X.copy()

        for col in columns:
            if col in X.columns:
                X = X.drop(col, axis=1)

        return X
