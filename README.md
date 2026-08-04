
# fieldlens

A probabilistic crop health monitoring system, featuring anomaly detection and short-horizon NDVI forecasting from remote sensing and weather data;  built around a delay-embedded Gaussian Mixture Model adapted from [...].







## What this is

## What this isn't
An intrinsic time-series model. By definition, the GMM assumes independent observations and as such, Stochastic Processes and Time Series Models are not being considered outright in the traditional sense. Of course, this assumption trivially posesses the Markov Property, and in a sense a GMM is a restricted Hidden Markov Model, but this is not a note-worthy aspect of the implementation.

Additionally, this project is not made to rival state-of-the-art models typically used in Time Series Analysis. Instead it is a proof of concept and a deep-dive into a model I found most interesting during one of my courses, as well as peaked curiosity from the paper [...]. 
## Optimizations

optimisations


## Project Structure
## Usage/Examples

```python
pass
```


## Evaluation
##  Limitations
The largest limitation this model faces is _data_. More specifically,
- Sentinel-2 only passes an area every x amount of days. If a specific pass is flagged by the cloud-mask, we lose that observation. This means that, across an interval of multiple years, we will only have a fraction of observations available to calculate the Normalised Difference indices. The missing observations are accounted for by using cubic-spline interpolation, and could theoretically be predicted using a climatology model.
- The delay-embedded GMM struggles with large amounts of multi-colinearity. Even when trained on the interpolated data, the sheer amount of features produced by delay-embedding results in a highly overfit model. This is mostly mitigated by forcing a "tied" covariance matrix across each cluster, but the ideal situation (i.e. tens-of-thousands of observations) would rather have a unique one for each. 
## Installation


```python
pip #maybe?
```
    
## Notebooks

List of jupyter Notebooks


## Screenshots
