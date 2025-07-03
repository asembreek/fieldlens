import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_index_heatmap(df, index):
    monthly_i_df = df.groupby(["Year", "Month"])[index].mean().reset_index()
    heatmap_data = monthly_i_df.pivot(index="Month", columns="Year", values=index)
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="YlGn")
    plt.tight_layout()
    plt.show()


def plot_index_barchart(df, index):
    df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df = df.sort_values("date")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=df["date"], y=df[index])
    plt.tight_layout()
    plt.show()
