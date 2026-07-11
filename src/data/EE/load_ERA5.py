import ee
import numpy as np
import pandas as pd

import ee_utils as utils

TEMP_2M = "temperature_2m"
TEMP_2M_MIN = "temperature_2m_min"
TEMP_2M_MAX = "temperature_2m_max"
SOIL_TEMP_LEVEL = "soil_temperature_level_"
VOL_SOIL_WATER_LAYER = "volumetric_soil_water_layer_"
SSRD = "surface_solar_radiation_downwards_sum"


def get_era5_data(aoi, i_date, f_date):
    era5_land = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(aoi)
        .filter(ee.Filter.date(i_date, f_date))
    )
    return era5_land


def get_soil_levels_df(image, reducer_fn, variable):
    sl_df = pd.DataFrame()

    if variable == "temp":
        fc_selection_pref = SOIL_TEMP_LEVEL
    elif variable == "water":
        fc_selection_pref = VOL_SOIL_WATER_LAYER
    else:
        raise ValueError(f"Unknown Soil Level variable: {variable}")

    for i in range(4):
        fc_selection = fc_selection_pref + str(i + 1)
        sl_i_df = utils.im_to_df(image, fc_selection, reducer_fn)

        if i == 0:
            sl_df["Timestamp"] = sl_i_df["Timestamp"]

        sl_df[fc_selection] = sl_i_df[fc_selection]
    print(f"Concatinated {variable} levels DataFrame.")
    return sl_df


def get_temp_2m_df(image, reducer_fn):
    temp_2m_df = utils.im_to_df(image, TEMP_2M, reducer_fn)
    # print(utils.im_to_df(image, TEMP_2M, reducer_fn))

    temp_2m_df[TEMP_2M_MIN] = utils.im_to_df(image, TEMP_2M_MIN, reducer_fn)[
        TEMP_2M_MIN
    ]
    temp_2m_df[TEMP_2M_MAX] = utils.im_to_df(image, TEMP_2M_MAX, reducer_fn)[
        TEMP_2M_MAX
    ]
    return temp_2m_df


# Surface solar radiation downwards
def get_ssrd(image, reducer_fn):
    ssrd_df = utils.im_to_df(image, SSRD, reducer_fn)
    return ssrd_df
