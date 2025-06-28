import ee


def get_s2_data(aoi, idate, fdate):
    s2_sr_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(idate, fdate)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    return s2_sr_col


def s2_clear_sky(image):
    scl = image.select("SCL")
    clear_sky_pixels = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11))
    masked = image.updateMask(clear_sky_pixels).divide(10000)
    return masked.copyProperties(image, ["system:time_start"])
