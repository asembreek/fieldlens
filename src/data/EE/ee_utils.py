import json

import ee
import pandas as pd

def make_grid(region, cell_size_m, epsg, full_cells_only=True):
    # represent locations as projected coordinates in meters
    proj = ee.Projection(epsg).atScale(cell_size_m)
    grid = region.coveringGrid(proj)

    if full_cells_only: # Filter by grid cells that are full-sized
        def mark_full(f):
            return f.set("inside", f.geometry().within(region, 1))

        grid = grid.map(mark_full).filter(ee.Filter.eq("inside", 1))

    features = grid.getInfo()["features"]

    def sorter(f):
        coords = f["geometry"]["coordinates"][0]
        sum_lat = 0
        sum_long = 0
        for p in coords:
            sum_lat += p[1]
            sum_long += p[0]

        return (sum_lat / len(coords), sum_long / len(coords))

    features = sorted(features, key=sorter)
    cells = []

    for i, f in enumerate(features):
        cell_id = f"cell_{i:03d}"
        geometry = ee.Geometry(f["geometry"])
        cells.append((cell_id, geometry))

    return cells

def stack_cells(cells, build_df_fn):
    dfs = []
    for cell_id, geom in cells:
        df = build_df_fn(geom)
        df.insert(0, "cell_id", cell_id)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

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
    with open("../data/external/map.geojson", "r") as f:
        gj = json.load(f)
    coords = gj["features"][0]["geometry"]["coordinates"]
    region = ee.Geometry.Polygon(coords)

    return region


def im_to_df(image, fc_selection, reducer_fn, filt=None):
    print(f"Creating {fc_selection} DataFrame...")
    fc_data = image.select(fc_selection)
    if not filt:
        features = ee.FeatureCollection(fc_data.map(reducer_fn))
    else:
        features = ee.FeatureCollection(fc_data.map(reducer_fn)).filter(filt)

    select_dict = _fc_to_dict(features).getInfo()
    select_df = pd.DataFrame(select_dict)
    select_df = _add_timestamp(select_df)
    return select_df





def _fc_to_dict(fc):
    property_name = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(property_name.size()),
        selectors=property_name,
    ).get("list")

    return ee.Dictionary.fromLists(property_name, prop_lists)


def _add_timestamp(df):
    df["Timestamp"] = pd.to_datetime(df["millis"], unit="ms")
    f_df = df.drop(columns=["millis", "system:index"])
    return f_df
