import json

import ee
import pandas as pd


def create_reducer(
    region, scale, reducer=ee.Reducer.mean(), maxPixels=1e13, tileScale=4
):
    def reduce_region_function(image):
        reduced = image.reduceRegion(
            reducer=reducer,
            geometry=region,
            scale=scale,
            maxPixels=maxPixels,
            tileScale=tileScale,
        )
        return ee.Feature(region, reduced).set({"millis": image.date().millis()})

    return reduce_region_function


def get_region():
    with open("map.geojson", "r") as f:
        gj = json.load(f)
    coords = gj["features"][0]["geometry"]["coordinates"]
    region = ee.Geometry.Polygon(coords)

    return region


def fc_to_df(image, fc_selection, reducer_fn, filt=None):
    fc_data = image.select(fc_selection)
    if not filt:
        features = ee.FeatureCollection(fc_data.map(reducer_fn))
    else:
        features = ee.FeatureCollection(fc_data.map(reducer_fn)).filter(filt)

    select_dict = _fc_to_dict(features).getInfo()
    select_df = pd.DataFrame(select_dict)
    select_df = _add_date_info(select_df)
    return select_df


def _fc_to_dict(fc):
    property_name = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(property_name.size()),
        selectors=property_name,
    ).get("list")

    return ee.Dictionary.fromLists(property_name, prop_lists)


def _add_date_info(df):
    df["Timestamp"] = pd.to_datetime(df["millis"], unit="ms")
    df["Year"] = pd.DatetimeIndex(df["Timestamp"]).year
    df["Month"] = pd.DatetimeIndex(df["Timestamp"]).month
    df["Day"] = pd.DatetimeIndex(df["Timestamp"]).day
    df["DOY"] = pd.DatetimeIndex(df["Timestamp"]).dayofyear
    return df
