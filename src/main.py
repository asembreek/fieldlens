import json

import ee
from data.load_sentinel2 import Sentinel_2
from visualisation.beta_ndvi import Beta


def main():
    ee.Authenticate()
    ee.Initialize(project="agriculture-drought-assesment")
    i_date = "2024-01-01"
    f_date = "2024-12-31"
    gjson = "./data/map.geojson"
    with open(gjson, "r") as f:
        geoJSON = json.load(f)

    s2 = Sentinel_2(i_date, f_date)
    s2.set_region(geoJSON)
    region = s2.get_region()
    s2_data = s2.get_sent2_data()
    ndvi_data = s2.get_ndvi_data(s2_data)

    beta = Beta(ndvi_data, region)
    beta.draw_raw_fit()


main()
