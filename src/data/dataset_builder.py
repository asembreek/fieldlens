class DatasetBuilder:
    # TODO: method chaining for building X_norm and X_anom as in jupyter notebook
    def __init__(self):
        self.df = None

    def merge_non_spectral(
        self,
        anom_non_spectral,
        anom_engineered,
        non_spectral_features,
        engineered_features,
    ):
        return self

    def merge_spectral(self, spectral):
        return self

    def interpolate(self, val_columns):
        self.df.loc[:, val_columns] = (
            self.df[val_columns]
            .interpolate(method="pchip", limit_area="inside")
            .bfill()
        )
        return self

    def drop_columns(self, columns):
        return self

    def build(self):
        return self.df

    def plot_interpolate(self):
        return self
