
# fieldlens

A probabilistic crop health monitoring system, featuring anomaly detection and short-horizon NDVI forecasting from remote sensing and weather data;  built around a delay-embedded Gaussian Mixture Model adapted from _"Gaussian Mixture Models for Time Series Modelling, Forecasting, and Interpolation"_ by _Emil Eirola_ and _Amaury Lendasse_.

## What this is
A reusable, modular Python implenetation for modelling and analysing environmental time series data using Gaussian Mixture Models (GMMs). The package includes modules for feature engineering, anomaly detection and time-series forecasting.

The feature-engineering component uses a Climatology (cyclic) Generalised Additive Model (GAM) as a foundation for the building blocks used to train the GMMs. In particular, the GAM allows for an approximation of when the growth-season starts/ends, which allows for easier classification of growth stages and calculating cumulative Growing Degree Days (GDD).

The anomaly detection component focuses on separation of anomalous years from the training data and selecting a subset of features that best detect whenever a given environmental observation deviates from the expected behaviour. 

The forecasting part of the project embeds observations into a higher-dimensional feature space, whereafter the conditional distribution is estimated from the trained model. This follows the general framework of  GMM-based time-series forecasting described by Eirola and Lendasse.


## What this isn't
This is **not** an intrinsic time-series model. By definition, the GMM assumes independent observations and as such, Stochastic Processes and Time Series Models are not used to model temporal structures. Instead, this structure is modeled by delay-embedding the series into fixed-length overlapping windows, making use of the Expectation Step of the EM-Algorithm to forecast the response variables.

Additionally, this project is **not** made to rival state-of-the-art models typically used in Time Series Analysis. It is primarily a proof of concept and a deep-dive into a model I found most interesting in my studies and later expanded upon in the paper [...]. The goal is therefore the implementation and understanding of these ideas. 

## Notebooks

| **Notebook**                          | **Purpose**                                                                                                                                                                                                             |   |
|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
|  0. Fetch, Load and Merge data        |  Download and preprocess satellite data from GEE. Functions here require GEE Authorization, therefore does not form part of the API.  Raw example data is instead provided in the form of a `load_example()` function.  |   |
|  1. Intro and Initial EDA             | Explanation of the Fieldlens project and the source and explanation of the data. Contains limited initial Exploratory Data Analysis to lay the groundwork for feature engineering.                                      |   |
|  2. Feature Engineering               | Contains all feature engineering approaches used in this project, such as fitting the Climatology GAM, calculating Growing Degree Days and implementing Growth Stage Classification.                                    |   |
|  3. EDA with engineered features      | Further Exploratory Data Analysis, but with the newly engineered features. This part specifically focuses on identifying anomalous years in the dataset.                                                                |   |
|  4. Gaussian Mixture Model            | Fitting of the unsupervised anomaly detecting Gaussian Mixture Model. Evaluated on both anomalous data and "normal" data (i.e. data that does not deviate far from approximately 0 log-likelihood)                      |   |
| 5. Forecasting Gaussian Mixture Model | Implementation of the research paper by Eirola and Lendasse (2013). Focuses on reducing overfitting and evaluating prediction accuracy.                                                                                                         |   |
## Usage/Examples

```python
//TODO
pass
```

## Evaluation
TODO

##  Limitations
The largest limitation this model faces is _data_. More specifically,
- Sentinel-2 only passes an area every x amount of days. If a specific pass is flagged by the cloud-mask, we lose that observation. This means that, across an interval of multiple years, we will only have a fraction of observations available to calculate the Normalised Difference indices. The missing observations are accounted for by using cubic-spline interpolation, and could theoretically be predicted using a climatology model.
- The delay-embedded GMM struggles with large amounts of multi-colinearity. Even when trained on the interpolated data, the sheer amount of features produced by delay-embedding results in a highly overfit model. This is mostly mitigated by forcing a "tied" covariance matrix across each cluster, but the ideal situation (i.e. tens-of-thousands of observations) would rather have a unique one for each.

## Project Structure

```
fieldlens/
├── data/
│   ├── external/       # External geographic/reference data
│   ├── raw/            # Raw datasets
│   └── processed/      # Cleaned and engineered datasets
├── notebooks/
│   ├── 0. Fetch, Load and Merge data.ipynb
│   ├── 1. Intro and Initial EDA.ipynb
│   ├── 2. Feature Engineering.ipynb
│   ├── 3. EDA with engineered features.ipynb
│   ├── 4. Gaussian Mixture Model.ipynb
│   └── 5. Forecasting Gaussian Mixture Model.ipynb
├── src/
│   ├── data/            # Data loading and Earth Engine utilities
│   ├── features/        # Feature engineering pipelines
│   └── models/          # GMM and forecasting models
├── README.md
└── requirements.txt
```

## Roadmap

- Create algorithm that splits .geojson into grids rather than averaging over entire space

- Add sections in README:
    - Screenshots
    - Usage
    - Installation

 - Implement an online, live-updating model, focusing on a specific area in the Free State.  


## References

Eirola, E. and Lendasse, A. (2013). Gaussian Mixture Models for Time Series Modelling, Forecasting, and Interpolation. [online] Available at: https://research.cs.aalto.fi/aml/Publications/Publication204.pdf
    - Installation

