import pandas as pd
import numpy as np


def merge_with_master(master, *other_dfs):
    on_cols = ["cell_id", "Timestamp"]
    master_df = master.copy()
    for df in other_dfs:
        master_df = pd.merge(master_df, df, on=on_cols, how="inner")
    return master_df


def add_date_info(df):
    df["year"] = pd.DatetimeIndex(df["Timestamp"]).year
    df["doy"] = pd.DatetimeIndex(df["Timestamp"]).dayofyear
    return df
