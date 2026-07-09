import pandas as pd
import numpy as np


def merge_with_master(master, *other_dfs):
    master_dates = master["Timestamp"]
    master_df = master.copy()

    for df in other_dfs:
        df_filtered = df[df["Timestamp"].isin(master_dates)].copy()
        master_df = pd.merge(master_df, df_filtered, on="Timestamp", how="inner")
    return master_df


def add_date_info(df):
    df["year"] = pd.DatetimeIndex(df["Timestamp"]).year
    df["doy"] = pd.DatetimeIndex(df["Timestamp"]).dayofyear
    return df
