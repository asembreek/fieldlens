import ee
import numpy as np
import pandas as pd

from data import utils


def get_era5_data(aoi, i_date, f_date):
    era5_land = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(aoi)
        .filter(ee.Filter.date(i_date, f_date))
    )
    return era5_land


def get_level_i_soil_temp(image, i):
    return image.select("soil_temperature_level_" + str(i))
