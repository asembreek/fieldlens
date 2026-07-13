import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DatasetBuilder:
    # TODO: method chaining for building X_norm and X_anom as in jupyter notebook
    def __init__(self, init_df=None):
        self.df = init_df.copy() if init_df is not None else None

    def merge_non_spectral(
        self,
        non_spectral,
        engineered,
        non_spectral_features=None,
        engineered_features=None,
    ):
        non_spectral_features = (
            non_spectral.columns
            if non_spectral_features is None
            else non_spectral_features
        )

        engineered_features = (
            engineered.columns if engineered_features is None else engineered_features
        )

        self.df = non_spectral[non_spectral_features].merge(
            engineered[engineered_features], on="Timestamp"
        )

        self._interpolated = None

        return self

    def merge_spectral(self, spectral, spectral_features=None):
        spectral_features = (
            spectral.columns if spectral_features is None else spectral_features
        )
        self.df = self.df.merge(spectral[spectral_features], on="Timestamp", how="left")
        return self

    def interpolate(self, val_columns=None):
        val_columns = self.df.columns if val_columns is None else val_columns
        self.df.loc[:, val_columns] = (
            self.df[val_columns]
            .interpolate(method="pchip", limit_area="inside")
            .bfill()
        )
        self._interpolated = list(val_columns)
        return self

    def drop_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)
        return self

    def build(self):
        return self.df

    def split_by_growing_season(self, *years):
        if "Ag_year" not in self.df.columns:
            raise ValueError(
                "Agricultural Year feature not in dataset. Run 'PhenologyFeatures(...).transform(X)' first."
            )
        mask = self.df["Ag_year"].isin(years)
        return self.df[~mask].copy(), self.df[mask].copy()

    def plot_interpolate(self, data, x="Timestamp", y="NDVI_mean"):
        if x not in data.columns:
            raise ValueError(f"{x} is not a valid column in 'data'.")
        if y not in self._interpolated:
            raise ValueError(
                f"Column '{y}' was not interpolated. Pleasure only select interpolated features."
            )

        plt.figure(figsize=(12, 6))
        plt.title(f"Original vs. Interpolated {y}")
        sns.lineplot(
            x=data[x],
            y=data[y],
            label=f"Original {y}",
            c="black",
        )
        sns.lineplot(
            x=self.df[x],
            y=self.df[y],
            label=f"Interpolated {y}",
            c="red",
            alpha=0.5,
        )
        sns.lineplot()
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.tight_layout()
        plt.show()
