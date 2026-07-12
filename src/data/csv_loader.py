import pandas as pd


class DatasetLoader:
    NON_SPECTRAL = "non_spectral_df"
    SPECTRAL = "spectral_df"
    GROUPED_SPECTRAL = "grouped_spectral_df"
    ENGINEERED_NON_SPECTRAL = "eng_non_spectral_df"
    ROLLING_STATS = "roll_df"

    def __init__(self, raw_data_dir, engineered_data_dir):
        self.raw_dir = raw_data_dir
        self.engineered_dir = engineered_data_dir

    def non_spectral(self):
        dir = f"{self.raw_dir}/{self.NON_SPECTRAL}.csv"
        return self._load_df(dir)

    def spectral(self):
        dir = f"{self.raw_dir}/{self.SPECTRAL}.csv"
        return self._load_df(dir)

    def grouped_spectral(self):
        dir = f"{self.raw_dir}/{self.GROUPED_SPECTRAL}.csv"
        return self._load_df(dir)

    def engineered(self):
        dir = f"{self.engineered_dir}/{self.ENGINEERED_NON_SPECTRAL}.csv"
        return self._load_df(dir)

    def rolling_stats(self):
        dir = f"{self.engineered_dir}/{self.ROLLING_STATS}.csv"
        return self._load_df(dir)

    def _load_df(self, dir):
        df = pd.read_csv(dir, index_col=0)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        categorical_cols = ["in_season", "Growth_Stage"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df
