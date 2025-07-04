import importlib

import ee
import pandas as pd
import seaborn as sns

print("Authenticating with Google Earth Engine...")
ee.Authenticate()

print("Initializing project...")
ee.Initialize(project="agriculture-drought-assesment")

from data import indices, load_ERA5, load_S2, utils
from visualisation import index_plotter

importlib.reload(indices)
importlib.reload(load_S2)
importlib.reload(load_ERA5)
importlib.reload(utils)
importlib.reload(index_plotter)


def main():

    region = utils.get_region()
    i_date = "2024-01-01"
    f_date = "2024-12-31"
    print(f"Loading Sentinel-2 dataset for periods {i_date} to {f_date}")

    s2_clouded = load_S2.get_s2_data(region, i_date, f_date)
    print(f"Loading ERA5 Land data for periods {i_date} to {f_date}")

    era5_data = load_ERA5.get_era5_data(region, i_date, f_date)

    # s2_clear_sky = s2_clouded.map(load_S2.s2_clear_sky)
    # index_fn = indices.get_index_fn("NDVI")
    # index_calc_data = s2_clear_sky.map(index_fn)

    reducer_fn = utils.create_reducer(region, scale=20)
    soil_temp_df = load_ERA5.get_soil_levels_df(era5_data, reducer_fn, "water")

    # index_df = utils.im_to_df(
    #    index_calc_data, "NDVI", reducer_fn, filt=ee.Filter.notNull(["NDVI"])
    # )

    print(soil_temp_df)


main()
