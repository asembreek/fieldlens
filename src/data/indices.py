import ee
import pandas as pd

from data import utils

NDVI = "NDVI"  # Normalized Difference Vegetation Index
NDRE = "NDRE"  # Normalized Difference Red Edge Index
NDWI = "NDWI"  # Normalized Difference Water Index


def get_index_df(s2, index, reducer_fn):
    """Uses a Sentinel-2 ImageCollection to create a DataFrame containing the computed values of the given spectral index, reduced over a specified region.
    Args:
        s2 (ImageCollection): Cloud-masked Sentinel-2 ImageCollection.
        index (string): The spectral index to be computed.
        reducer_fn (function): The reducer function used to map over the aoi to find the mean of the spectral index.
    """
    index_fn = _get_index_fn(index)
    index_calc_data = s2.map(index_fn)
    index_df = utils.im_to_df(
        index_calc_data, index, reducer_fn, filt=ee.Filter.notNull([index])
    )
    return index_df


def apply_groupby_date(i_df, *other_dfs):
    """Groups spectral index calculations by timestamp. This compensates for the fact that Sentinel-2 sometimes passes the same region more than once a day.
    Args:
        i_df (DataFrame): DataFrame returned by get_index_df(...). Contains spectral index calculations.
        *other_dfs: Any other DataFrames with a valid "Timestamp" column to be grouped.
    """
    all_dfs = (i_df,) + other_dfs
    grouped_results = []
    for df in all_dfs:
        col_name = df.columns[0]
        grouped = (
            df.groupby(df["Timestamp"].dt.date)
            .agg({df.columns[0]: ["mean", "std"]})
            .fillna(0)
            .reset_index()
        )
        grouped.columns = ["Timestamp", f"{col_name}_mean", f"{col_name}_std"]
        grouped["Timestamp"] = pd.to_datetime(grouped["Timestamp"])
        grouped_results.append(grouped)
    return grouped_results


def _get_index_fn(index):
    """Function factory that returns the mapping function that calculates the given spectral index using the Sentinel-2 spectral bands.
    Args:
        index (string): The spectral index to be calculated.
    """
    match index:
        case "NDVI":
            return _append_ndvi
        case "NDRE":
            return _append_ndre
        case "NDWI":
            return _append_ndwi
        case _:
            raise ValueError(f"Unknown index: {index}")


def _append_ndvi(image):
    """Mapping function used to calculate Normalized Difference Vegetation Index.
    Args:
        image(ImageCollection): Cloud-masked Sentinel-2 ImageCollection over a valid region.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def _append_ndre(image):
    """Mapping function used to calculate Normalized Difference Red Edge Index.
    Args:
        image(ImageCollection): Cloud-masked Sentinel-2 ImageCollection over a valid region.
    """

    ndre = image.normalizedDifference(["B8", "B5"]).rename("NDRE")
    return image.addBands(ndre)


def _append_ndwi(image):
    """Mapping function used to calculate Normalized Difference Water Index.
    Args:
        image(ImageCollection): Cloud-masked Sentinel-2 ImageCollection over a valid region.
    """
    ndwi = image.normalizedDifference(["B8", "B12"]).rename("NDWI")
    return image.addBands(ndwi)
