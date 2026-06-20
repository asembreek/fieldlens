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


def plot_index_full_barchart(df, index):
    df = df.sort_values("Timestamp")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=df["Timestamp"], y=df[index])
    plt.tight_layout()
    plt.show()


def plot_index_doy_barchart(df, index):
    if f"{index}_mean" in df.columns:
        y_column = f"{index}_mean"
    else:
        y_column = index

    plot_df = df.copy()
    plot_df["DOY"] = df["Timestamp"].dt.dayofyear
    plot_df["Year"] = pd.DatetimeIndex(df["Timestamp"]).year
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=plot_df,
        x="DOY",
        y=y_column,
        hue="Year",
        palette="tab10",
    )
    plt.title(f"{index} Mean per Day of Year (DOY) by Year")
    plt.xlabel("Day of Year (DOY)")
    plt.ylabel(f"{index} Mean")
    plt.legend(title="Year")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
