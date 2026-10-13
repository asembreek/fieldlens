import json

import ee
import pandas as pd

MAX_ROWS_PER_REQUEST = 3000


def make_grid(region, cell_size_m, epsg, full_cells_only=True):
    # represent locations as projected coordinates in meters
    proj = ee.Projection(epsg).atScale(cell_size_m)
    grid = region.coveringGrid(proj)

    if full_cells_only:  # Filter by grid cells that are full-sized

        def mark_full(f):
            return f.set("inside", f.geometry().containedIn(region, maxError=1))

        grid = grid.map(mark_full).filter(ee.Filter.eq("inside", True))

    grid_list = grid.toList(grid.size())
    indices = ee.List.sequence(0, grid_list.size().subtract(1))

    indexed_list = indices.map(
        lambda idx: ee.Feature(grid_list.get(idx)).set(
            "cell_id", ee.String("cell_").cat(ee.Number(idx).format("%03d"))
        )
    )
    indexed_grid = ee.FeatureCollection(indexed_list)

    return indexed_grid


def create_reducer(cells_fc, scale, n_cells, reducer=ee.Reducer.mean(), tileScale=4):
    def reduce_regions_function(image):
        band = ee.String(image.bandNames().get(0))
        reduced = image.reduceRegions(
            collection=cells_fc, reducer=reducer, scale=scale, tileScale=tileScale
        )
        millis = image.date().millis()
        return reduced.map(
            lambda feature: feature.setGeometry(None)
            .set(band, feature.get("mean"))
            .set("millis", millis)
        )

    reduce_regions_function.n_cells = n_cells
    return reduce_regions_function


def get_region():
    with open("../data/external/map.geojson", "r") as f:
        gj = json.load(f)
    coords = gj["features"][0]["geometry"]["coordinates"]
    region = ee.Geometry.Polygon(coords)

    return region


def im_to_df(image, fc_selection, reducer_fn, filt=None):
    print(f"Creating {fc_selection} DataFrame...")
    fc_data = image.select(fc_selection).sort("system:time_start")

    n_images = fc_data.size().getInfo()
    img_list = fc_data.toList(n_images)

    # divide request into date chunks to avoid requesting too many elements
    per_chunk = max(1, MAX_ROWS_PER_REQUEST // reducer_fn.n_cells)

    frames = []
    for a in range(0, n_images, per_chunk):
        chunk = ee.ImageCollection(img_list.slice(a, a + per_chunk))
        features = chunk.map(reducer_fn).flatten()
        if filt:
            features = features.filter(filt)
        if features.size().getInfo() == 0:
            # if cloud-masked
            continue
        frames.append(pd.DataFrame(_fc_to_dict(features).getInfo()))

    df = pd.concat(frames, ignore_index=True)
    return _add_timestamp(df, fc_selection)


def _fc_to_dict(fc):
    property_name = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(property_name.size()),
        selectors=property_name,
    ).get("list")

    return ee.Dictionary.fromLists(property_name, prop_lists)


def _add_timestamp(df, fc_selection):
    df["Timestamp"] = pd.to_datetime(df["millis"], unit="ms")
    df = df.drop(columns=["millis", "system:index"], errors="ignore")
    # value column first (apply_groupby_date relies on that), rows in a stable order
    df = df[[fc_selection, "cell_id", "Timestamp"]]
    return df.sort_values(["cell_id", "Timestamp"]).reset_index(drop=True)
