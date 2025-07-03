import importlib

import ee
import pandas as pd
import seaborn as sns

ee.Authenticate()
ee.Initialize(project="agriculture-drought-assesment")

from data import indices, load_ERA5, loadS2, utils
from visualisation import index_plotter

importlib.reload(indices)
importlib.reload(loadS2)
importlib.reload(load_ERA5)
importlib.reload(utils)
importlib.reload(index_plotter)


def main():

    region = utils.get_region()
    reducer_fn = utils.create_reducer(region, scale=10)

    s2_clouded = loadS2.get_s2_data(region, "2024-01-01", "2024-12-31")
    era5_data = load_ERA5.get_era5_data(region, "2024-01-01", "2024-12-31")

    s2_clear_sky = s2_clouded.map(loadS2.s2_clear_sky)
    index_fn = indices.get_index_fn("NDVI")
    index_calc_data = s2_clear_sky.map(index_fn)

    soil_temp_df = utils.fc_to_df(era5_data, "soil_temperature_level_1", reducer_fn)
    index_df = utils.fc_to_df(
        index_calc_data, "NDVI", reducer_fn, filt=ee.Filter.notNull(["NDVI"])
    )


main()
