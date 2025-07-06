import importlib

import ee

print("Authenticating with Google Earth Engine...")
ee.Authenticate()

print("Initializing project...")
ee.Initialize(project="agriculture-drought-assesment")

from data import indices, load_CHIRPS, load_ERA5, load_S2, utils
from visualisation import index_plotter

importlib.reload(indices)
importlib.reload(load_S2)
importlib.reload(load_ERA5)
importlib.reload(load_CHIRPS)
importlib.reload(utils)
importlib.reload(index_plotter)


def main():

    region = utils.get_region()
    i_date = "2019-01-01"
    f_date = "2023-12-31"
    master_df = get_master_df(region, i_date, f_date)


def get_master_df(region, i_date, f_date):

    print(f"Loading Sentinel-2 dataset for periods {i_date} to {f_date}...")
    s2_clouded = load_S2.get_s2_data(region, i_date, f_date)

    print(f"Loading ERA5 Land data for periods {i_date} to {f_date}...")
    era5_data = load_ERA5.get_era5_data(region, i_date, f_date)

    print(f"Loading CHIRPS Climate Data for periods {i_date} to {f_date}...")
    chirps_data = load_CHIRPS.get_chirps_data(region, i_date, f_date)

    s2_clear_sky = s2_clouded.map(load_S2.s2_clear_sky)

    reducer_fn = utils.create_reducer(region, scale=10)
    ndvi_df = indices.get_index_df(s2_clear_sky, indices.NDVI, reducer_fn)
    ndre_df = indices.get_index_df(s2_clear_sky, indices.NDRE, reducer_fn)
    ndwi_df = indices.get_index_df(s2_clear_sky, indices.NDWI, reducer_fn)

    reducer_fn = utils.create_reducer(region, scale=20)
    slvw_df = load_ERA5.get_soil_levels_df(era5_data, reducer_fn, "water")
    slt_df = load_ERA5.get_soil_levels_df(era5_data, reducer_fn, "temp")
    temp_2m_df = load_ERA5.get_temp_2m_df(era5_data, reducer_fn)
    ssrd_df = load_ERA5.get_ssrd(era5_data, reducer_fn)

    reducer_fn = utils.create_reducer(region, scale=5566)
    perc_df = load_CHIRPS.get_precipitation_df(chirps_data, reducer_fn)

    grouped = indices.apply_groupby_date(ndwi_df, ndvi_df, ndre_df)
    ndwi_df_grouped = grouped[0]
    ndvi_df_grouped = grouped[1]
    ndre_df_grouped = grouped[2]

    master_df = utils.merge_with_master(
        ndvi_df_grouped,
        ndwi_df_grouped,
        ndre_df_grouped,
        slvw_df,
        slt_df,
        temp_2m_df,
        ssrd_df,
        perc_df,
    )
    print("Done!")
    return master_df


main()
