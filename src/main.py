import importlib
import os

import ee
import numpy as np
import pandas as pd
import seaborn as sns

ee.Authenticate()
ee.Initialize(project="agriculture-drought-assesment")

from data import indices, loadS2, utils
from visualisation import index_plotter

importlib.reload(indices)
importlib.reload(loadS2)
importlib.reload(utils)
importlib.reload(index_plotter)


def main():

    region = utils.get_region()
    s2_clouded = loadS2.get_s2_data(region, "2019-01-01", "2024-12-31")
    s2_clear_sky = s2_clouded.map(loadS2.s2_clear_sky)
    reduce_fn = utils.create_reducer(region, scale=10)

    ndvi_df = get_index_df(s2_clear_sky, reduce_fn, "NDVI")
    index_plotter.plot_index_barchart(ndvi_df, "NDVI")


def get_index_df(s2, reducer, index):
    index_fn = indices.get_index_fn(index)
    s2_i = s2.map(index_fn).select(index)

    features = ee.FeatureCollection(s2_i.map(reducer)).filter(
        ee.Filter.notNull([index])
    )

    index_dict = utils.fc_to_dict(features).getInfo()
    index_df = pd.DataFrame(index_dict)
    index_df = utils.add_date_info(index_df)
    index_df = index_df.drop(columns=["millis", "system:index"])
    return index_df


main()
