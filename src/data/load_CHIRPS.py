import ee


def get_precipitation_data(aoi, i_date, f_date):
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filter(ee.Filter.date(i_date, f_date))
    )
    precipitation = chirps.select("precipitation")
    return precipitation
