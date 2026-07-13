import numpy as np
import pandas as pd
from models import NDVIClimatologyGAM as shifter
from data import utils
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed


class PhenologyFeatures:
    x_ticks = np.arange(1, 366, 30)
    doys = np.arange(1, 366)

    def __init__(
        self,
        start_of_season,
        end_of_season,
        base_temp=10,
        growth_stages=True,
        cumulative_gdd=True,
        season_markers=True,
    ):
        self.start_of_season = start_of_season
        self.end_of_season = end_of_season

        self.base_temp = base_temp
        self.growth_stages = growth_stages
        self.cumulative_gdd = cumulative_gdd
        self.season_markers = season_markers

    def fit(self, X):
        return self

    def transform(self, X):
        required = {"Timestamp", "temperature_2m_min", "temperature_2m_max"}
        missing = required - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = X.copy()

        date_cols = ["doy", "year"]
        contains_dates = True

        if not all(col in X.columns for col in date_cols):
            df = utils.add_date_info(df)
            contains_dates = False
        df["shifted_doy"] = shifter.shift_doy(df["doy"], self.start_of_season)

        df = self._add_growth_season_markers(df)
        df = self._add_gdd(df)
        df = self._add_cumulative_gdd(df)
        df = self._add_growth_stages(df)

        if not contains_dates:
            df = df.drop(["doy", "year"], axis=1)
        df = df.drop("shifted_doy", axis=1)
        return df

    def fit_transform(self, X):
        return self.transform(X)

    def plot_growth(
        self, X, year=None, growth_stages=True, shifted=True, all_years=True
    ):
        if not all_years and year is None:
            raise TypeError("Missing `year` parameter if `all_years=False`.")

        requires_transform = {"Ag_year", "cumulative_GDD"}

        if growth_stages:
            requires_transform.add("Growth_Stage")

        self._check_cols(cols=X.columns, required=requires_transform, transform=True)
        self._check_cols(cols=X.columns, required={"NDVI_mean"}, transform=False)

        plot_df = X.copy()
        plot_df = utils.add_date_info(plot_df)
        if all_years:
            interact(
                self._plot_growth,
                plot_year=np.unique(plot_df["Ag_year"]),
                data=fixed(plot_df),
                shift=fixed(self.start_of_season if shifted else 0),
                growth_stages=fixed(growth_stages),
            )
        else:
            self._plot_growth(
                data=plot_df,
                plot_year=year,
                shift=self.start_of_season if shifted else 0,
                growth_stages=growth_stages,
            )

    def _add_growth_season_markers(self, X):
        df = X.copy()
        shifted_sos = shifter.shift_doy(self.start_of_season, self.start_of_season)
        shifted_eos = shifter.shift_doy(self.end_of_season, self.start_of_season)
        print(shifted_sos)
        print(shifted_eos)
        df["Ag_year"] = np.where(
            df["doy"] >= self.start_of_season, df["year"], df["year"] - 1
        )
        df["in_season"] = (
            ((df["shifted_doy"] >= shifted_sos) & (df["shifted_doy"] <= shifted_eos))
            .astype(int)
            .astype("category")
        )
        df = df.sort_values(by=["Ag_year", "shifted_doy"])

        return df

    def _add_gdd(self, X, tmin_col="temperature_2m_min", tmax_col="temperature_2m_max"):
        df = X.copy()

        df["GDD"] = self._calculate_gdd(df[tmin_col], df[tmax_col])

        return df

    def _calculate_gdd(self, tmin, tmax):
        tmax_conv = tmax - 273.15
        tmin_conv = tmin - 273.15

        gdd = (tmax_conv + tmin_conv) / 2 - self.base_temp
        return np.maximum(gdd, 0)

    def _add_cumulative_gdd(self, X):
        df = X.copy()
        # Only sum GDD observations that are in season
        df["GDD_in_season"] = df["GDD"].where(df["in_season"] == 1, 0)
        df = df.sort_values(by=["Ag_year", "shifted_doy"])
        df["cumulative_GDD"] = df.groupby("Ag_year")["GDD_in_season"].cumsum()
        df["cumulative_GDD"] = df["cumulative_GDD"].where(df["in_season"] == 1, 0)

        return df.drop("GDD_in_season", axis=1)

    def _add_growth_stages(self, X):
        df = X.copy()

        stages = [
            "R6",
            "R5",
            "R4",
            "R3",
            "R2",
            "R1",
            "VT",
            "V16+",
            "V13",
            "V12",
            "V8-V9",
            "V6",
            "V4",
            "V2",
            "VE",
            "SoS",
        ]

        conditions = [
            df["cumulative_GDD"] >= 1500,
            df["cumulative_GDD"] >= 1278,
            df["cumulative_GDD"] >= 1083,
            df["cumulative_GDD"] >= 1042,
            df["cumulative_GDD"] >= 944,
            df["cumulative_GDD"] >= 833,
            df["cumulative_GDD"] >= 750,
            df["cumulative_GDD"] >= 639,
            df["cumulative_GDD"] >= 531,
            df["cumulative_GDD"] >= 500,
            df["cumulative_GDD"] >= 378,
            df["cumulative_GDD"] >= 289,
            df["cumulative_GDD"] >= 200,
            df["cumulative_GDD"] >= 111,
            df["cumulative_GDD"] >= 56,
            df["cumulative_GDD"] > 0,
        ]

        df["Growth_Stage"] = np.select(conditions, stages, default="Off-Season")
        df["Growth_Stage"] = np.where(
            df["in_season"] == 0, "Off-Season", df["Growth_Stage"]
        )
        return df

    def _check_cols(self, cols, required, transform=False):
        missing = required - set(cols)
        if missing:
            ERR_MSG = (
                f"Missing required columns: {missing}."
                if not transform
                else f"Missing required columns: {missing}. Run `transform(X)` first."
            )
            raise ValueError(ERR_MSG)
        return

    def _plot_growth(self, data, plot_year, shift, growth_stages):
        dfy = data[data["Ag_year"] == plot_year].copy()

        dfy["shifted_doy"] = shifter.shift_doy(dfy["doy"], shift)
        dfy = dfy.sort_values("shifted_doy").reset_index(drop=True)

        x_labels = shifter.inv_shift_doy(self.x_ticks, shift)

        fig, ax1 = plt.subplots(figsize=(12, 6.5))
        mask = dfy["NDVI_mean"].notna()

        ax1.plot(
            dfy.loc[mask, "shifted_doy"],
            dfy.loc[mask, "NDVI_mean"],
            color="forestgreen",
            label="NDVI",
            marker="o",
            lw=2.5,
            markersize=4,
        )
        ax1.set_xticks(self.x_ticks, labels=x_labels)
        ax1.set_xlabel("Day of Year")
        ax1.set_ylabel("NDVI")
        ax1.set_ylim(0, 1)
        ax1.grid(axis="y", alpha=0.5)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Cumulative GDD")
        ax2.plot(
            dfy["shifted_doy"],
            dfy["cumulative_GDD"],
            lw=2.5,
            linestyle="--",
            color="dimgrey",
            label="Cumulative GDD",
        )
        if growth_stages:
            self._plot_growth_stages(dfy, ax1)
        plt.title(
            f"NDVI + GDD progression (Ag_year {plot_year}), with Growth Labels",
            fontweight="semibold",
        )

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        fig.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper center",
            ncol=3,
            frameon=False,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        plt.show()

    def _plot_growth_stages(self, dfy, ax):
        # summary dataframe for growth stage transitions
        transition_dates = (
            dfy.groupby("Growth_Stage")["shifted_doy"].min().reset_index()
        )
        transition_dates = transition_dates.sort_values(by="shifted_doy")

        stage_positions = transition_dates["shifted_doy"].tolist()
        stage_labels = transition_dates["Growth_Stage"].tolist()

        ax.vlines(
            x=stage_positions,
            ymin=0,
            ymax=1,
            linestyle="-.",
            lw=1.2,
            color="blue",
            alpha=0.3,
            label="Growth Stages",
        )

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(stage_positions)
        ax_top.set_xticklabels(
            stage_labels,
            rotation=45,
            ha="left",
            fontweight="bold",
            color="black",
            fontsize=9,
        )
        ax_top.spines["top"].set_visible(False)
        ax_top.tick_params(top=False)
