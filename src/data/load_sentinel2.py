import json

import ee

# %matplotlib inline


class Sentinel_2:
    def __init__(self, i_date, f_date):
        self.i_date = i_date
        self.f_date = f_date
        self.region = ""


    def set_region(self, gjson):
        coords = gjson["features"][0]["geometry"]["coordinates"]
        self.region = ee.Geometry.Polygon(coords)


    def get_region(self):
        return self.region


    def get_sent2_data(self):

        s2 = ee.Image(
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(self.i_date, self.f_date)
            .filterBounds(self.region)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .first()
            .clip(self.region)
        )
        return s2


    def get_ndvi_data(self, s2):
        R = "B4"
        NIR = "B8"
        ndvi = s2.normalizedDifference([NIR, R])
        return ndvi
