import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DatasetBuilder:
    # TODO: method chaining for building X_norm and X_anom as in jupyter notebook
    def __init__(self):
        self.df = None

    def merge_non_spectral(
        self,
        non_spectral,
        engineered,
        non_spectral_features,
        engineered_features,
    ):

        self.df = non_spectral[non_spectral_features].merge(
            engineered[engineered_features], on="Timestamp"
        )

        self._interpolated = None

        return self

    def merge_spectral(self, spectral, spectral_features):
        self.df = self.df.merge(spectral[spectral_features], on="Timestamp", how="left")
        return self

    def interpolate(self, val_columns):
        self.df.loc[:, val_columns] = (
            self.df[val_columns]
            .interpolate(method="pchip", limit_area="inside")
            .bfill()
        )
        self._interpolated = val_columns
        return self

    def drop_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)
        return self

    def build(self):
        return self.df

    def plot_interpolate(self, data, x):
        if x not in data.columns:
            raise ValueError(f"{x} is not a valid column in 'data'.")
        plt.figure(figsize=(12, 6))
        plt.title("Original vs. Interpolated NDVI")
        sns.lineplot(
            x=data[x],
            y=data[self._interpolated],
            label="Original",
            c="black",
        )
        sns.lineplot(
            x=self.df[x],
            y=self.df[self._interpolated],
            label="Interpolated",
            c="red",
            alpha=0.5,
        )
        sns.lineplot()
        plt.xlabel(x)
        plt.ylabel("Data value")
        plt.legend()
        plt.tight_layout()
        plt.show()
