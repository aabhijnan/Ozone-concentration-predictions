# Tropospheric Ozone Concentration Prediction

Predicting annual average ground-level ozone at 5,577 monitoring stations worldwide
from station metadata and land-use variables — no time-series inputs, no chemistry
model. Built on the AQ-Bench dataset.

## The problem

Each row is one air-quality monitoring station. The inputs describe *where the station
is*: coordinates, altitude, climate zone, surrounding land cover within 25 km,
population density, night-light intensity, and NOx / NO2 levels. The target is
`o3_average_values` — that station's annual mean ozone in ppb.

The question the model answers: **can you estimate a station's ozone level from its
geography alone?**

## Data

`AQbench_dataset.csv` — 5,577 stations, 53 columns.

| Split | Stations |
|---|---|
| train | 3,348 |
| val | 1,114 |
| test | 1,115 |

The split is fixed by the `dataset` column in the CSV, so results are directly
comparable between runs. 70 countries are represented; 22 of them have 2 stations
or fewer.

Source: the AQ-Bench benchmark dataset (Betancourt et al.). Check the original
publication for the exact citation and licence terms before redistributing the CSV.

## Two things that will bite you

**1. Fourteen of the columns are the target in disguise.** `o3_daytime_avg`,
`o3_median`, `o3_perc25`, `o3_dma8eu`, `o3_aot40` and the rest are all different
aggregations of the same ozone measurements. Leave any of them in the features and
the model scores R² = 1.0 — it is reading the answer off the input, not predicting.
All 14 are dropped before training.

**2. `pd.get_dummies` changed behaviour in pandas 2.0.** It used to return `uint8`;
it now returns `bool`, and `select_dtypes(include=np.number)` does not match `bool`.
On pandas ≥ 2.0 the one-hot columns silently disappear, which then makes the
`dataset_*` drop raise `KeyError` — and because `drop` is atomic, the target never
gets removed either. The fix is `pd.get_dummies(df, dtype=np.uint8)`.

## Results

R² on the held-out splits, same features and same split throughout:

| Model | Validation | Test |
|---|---|---|
| Linear regression | 0.535 | 0.548 |
| Linear regression + engineered features | 0.542 | 0.580 |
| Lasso (alpha=0.001) + engineered features | 0.645 | 0.594 |
| **Neural network** (Keras, 1024-unit dense, 174 inputs) | **0.686** | **0.625** |

The engineered features are interaction terms between latitude, altitude, NO2,
land cover and urban density — the physical intuition being that ozone forms from
NOx under sunlight, so the product of the two should matter more than either alone.

Strongest individual predictors by correlation: altitude, relative altitude,
night-light intensity, urban land cover, and NO2 column.

## Repository layout

```
Ozone_conc_Pred.ipynb    main pipeline: load, clean, engineer, train, evaluate, map
AQbench_dataset.csv      the dataset
README.md
.gitignore
```

## Running it

Requires Python 3.11+, with:

```
pandas  numpy  scikit-learn  matplotlib  seaborn  folium  geopandas
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium geopandas
jupyter notebook Ozone_conc_Pred.ipynb
```

Update the `CSV` path at the top of the notebook to point at your copy, then
Run All. The final cells render a folium heatmap of prediction error by station
location.

## Notes on method

- Features are standardised before the interaction terms are built. For plain OLS
  this does not change R² at all, but the engineered features are nonlinear, so
  the scaling step affects them.
- Longitude is encoded as `cos(lon)`. Note that cosine alone maps +90° and −90° to
  the same value, which erases the east/west distinction — using both `sin` and
  `cos` is the standard fix, though on this dataset it did not improve the score.
- The `country` one-hot columns (70 of them, against 3,348 training rows) hurt
  generalisation. Dropping them and relying on `htap_region` and `climatic_zone`
  scored better on both validation and test.
