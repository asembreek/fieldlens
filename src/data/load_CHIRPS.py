import ee

from data import utils


def get_chirps_data(aoi, i_date, f_date):
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filter(ee.Filter.date(i_date, f_date))
    )
    return chirps


def get_precipitation_df(chirps, reducer_fn):
    precipitation_df = utils.im_to_df(chirps, "precipitation", reducer_fn)
    return precipitation_df
