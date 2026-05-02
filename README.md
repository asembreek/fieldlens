# Satellite-Based Crop Health Modelling

This project explores the relationship between normalised vegetation indices and environmental/climate variables to analyse crop health in the maize-growing region of the Free State.

It combines multiple sources of geospatial data to engineer a dataset for machine learning models aimed at classifying vegetation stress conditions, given current climate and farm management conditions.

## Data Sources
From **Google Earth Engine**:

* Sentinel-2 satellite imagery 
* CHIRPS precipitation data
* ERA5 temperature, soil temperature, volumetric soil water layer and solar radiation. 

## Current Focus

The project is currently in the feature engineering and EDA phase.

Work completed so far includes:

* Extraction and preprocessing of multi-source geospatial datasets.
* Computation of vegetation indices (including NDVI)
* Integration of climate variables with satellite observations
* Development of time-series feature representations for vegetation and weather interactions
* Initial exploratory analysis and visualization of temporal patterns

## Notebooks

* Time Series Visualisation Notebook:
  [View notebook](./notebooks/time_series_visualisation.ipynb)

This notebook contains:

* Time-series plots of vegetation indices and climate variables
* Preliminary feature relationships and trend analysis
* Early-stage data exploration and validation of extracted features

## Project Status

This project is in development.
The current focus is on strengthening feature engineering and constructing a dataset for subsequent model training and evaluation.

