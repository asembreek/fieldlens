import ee


def get_index_fn(index):
    match index:
        case "NDVI":
            return _append_ndvi
        case "NDRE":
            return _append_ndre
        case "NDWI":
            return _append_ndwi
        case _:
            raise ValueError(f"Unknown index: {index}")


# Normaliezd Difference Vegetation Index
def _append_ndvi(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


# Normalized Difference Red Edge Index: Chlorophyll content change detection caused by nutrient deficiency, fertilization issues or disease stress
def _append_ndre(image):
    ndre = image.normalizedDifference(["B8", "B5"]).rename("NDRE")
    return image.addBands(ndre)


# Normalized Difference Water Index: Canopy water content, used for drought stress.
def _append_ndwi(image):
    ndwi = image.normalizedDifference(["B8", "B12"]).rename("NDWI")
    return image.addBands(ndwi)
