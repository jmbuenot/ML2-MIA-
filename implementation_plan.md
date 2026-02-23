# Implementation Plan — ML2 Online Learning Project (Single Notebook)

## Project: Concept Drift Detection with Historical Hourly Weather Data (River)

---

## Notebook-Only Constraint

This project is implemented entirely inside a single Jupyter notebook (in this repository: `ml2_online_learing_project.ipynb`).

- No `src/` folder
- No separate `.py` modules
- No package-style architecture
- All logic is implemented in notebook cells
- Notebook must execute top-to-bottom without manual fixes

---

## Assumptions & Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **ML Task** | **Regression** (predict temperature) | Continuous target with strong seasonal patterns makes concept drift easy to demonstrate and visualize. |
| **Target Variable** | `temperature` (Kelvin -> Celsius) | Natural continuous target with strong seasonal drift signal. |
| **Global Model** | Single model across all 36 cities | Required by task; city is an encoded feature. |
| **Data Scope** | All cities, full temporal range | Maximizes drift visibility across geographies and years. |
| **Imbalance** | Not applicable to regression; `weather_description` is imbalanced | Included as required context. |

---

## Required Notebook Header Order (Strict Rubric Order)

1. `# Problem Definition`
2. `# Dataset Justification`
3. `# Data Preparation`
4. `# Offline (Batch) Learning`
5. `# Online (Stream) Learning`
6. `# Concept Drift Detection`
7. `# Offline vs Online Comparison`
8. `# Visualization`
9. `# Results and Conclusions`

Keep optional helper subsections inside the same notebook:

- `## Imports and Setup`
- `## Reproducibility`
- `## Validation Checkpoints`

---

# Problem Definition

## Implementation Steps

1. Describe the real-world problem in non-ML terms.
2. Define task as regression.
3. State global-model scope (all cities together).
4. Explain why concept drift is expected:
   - seasonal weather changes
   - geographic diversity
5. Define metrics and motivation:
   - MAE (interpretable error in C)
   - RMSE (penalizes large errors)
   - R2 (explained variance)
6. Document assumptions:
   - chronological data arrival
   - no future leakage
   - city-wise forward-fill for small gaps

**Validation checkpoint:** Problem, assumptions, and metric rationale are explicitly documented before modeling.

---

# Dataset Justification

## Implementation Steps

1. Describe source and all files used:
   - `humidity.csv`, `pressure.csv`, `temperature.csv`, `weather_description.csv`, `wind_direction.csv`, `wind_speed.csv`, `city_attributes.csv`
2. Explain wide format (cities are columns).
3. Justify wide-to-long transformation for stream setup.
4. Report initial diagnostics:
   - date range
   - number of cities
   - key variables
   - missingness overview
5. Justify stream-learning suitability:
   - temporal structure
   - hourly cadence
   - multiyear horizon with expected drift

**Validation checkpoint:** Dataset suitability for online learning and drift analysis is clear.

---

# Data Preparation

## Wide -> Long, Merge, and Cleaning

1. Implement notebook function:

```python
def melt_wide_csv(filepath: str, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    return df.melt(id_vars="datetime", var_name="city", value_name=value_name)
```

2. Melt each weather table into long format.
3. Merge all long tables on `(datetime, city)` with outer joins.
4. Join city metadata from `city_attributes.csv` (`country`, `latitude`, `longitude`).
5. Convert temperature K -> C (only if values look like Kelvin) and keep Celsius as the target.
6. Apply **city-wise LOCF (forward fill)** for the **target only** (`temperature`), after sorting by `city`, then `datetime`.
7. Drop unresolved target NaN rows (leading NaNs per city that remain after LOCF).
8. Sort final prepared table by `datetime`, then `city` for stream simulation.

### Target imputation (stream-safe) validation (inside notebook)

- Report NaN counts in `temperature`:
  - before LOCF
  - after LOCF (before drop)
  - after dropping unresolved target NaNs
- Report total rows dropped and top-10 cities by dropped rows
- Confirm final `temperature` has **zero NaNs**

## Feature Engineering (Inside Notebook)

1. Temporal features: `hour`, `day_of_week`, `month`, `day_of_year`, `year`.
2. Cyclical encodings:
   - `hour_sin/cos`
   - `month_sin/cos`
   - `doy_sin/cos`
   - `wind_dir_sin/cos`
3. Feature groups:
   - Numerical: humidity, pressure, wind_speed, lat/lon, cyclical features
   - Categorical: city, weather_description, country
4. Group rare `weather_description` values to `other` (<1% frequency).

**Validation checkpoint:** Final DataFrame is chronological, target-complete, and feature-complete.

---

# Offline (Batch) Learning

## Implementation Steps

1. Use temporal split (not random):

```python
split_date = df["datetime"].quantile(0.8)
train_df = df[df["datetime"] <= split_date]
test_df = df[df["datetime"] > split_date]
```

2. Preprocess in notebook:
   - one-hot encode categoricals
   - scale numericals (fit on train only)
3. Train at least one baseline; implement three:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. Evaluate with MAE, RMSE, R2.
5. Compute monthly test metrics to reveal seasonal performance shifts.
6. Optional: `TimeSeriesSplit` for time-aware validation.

**Validation checkpoint:** Offline results table is complete and leakage-free.

---

# Online (Stream) Learning

## Real-Time Simulation

1. Keep data sorted by `datetime`, `city`.
2. Create stream generator:

```python
from river import stream

def create_stream(df, target="temperature"):
    x_cols = [c for c in df.columns if c not in [target, "datetime"]]
    for x, y in stream.iter_pandas(df[x_cols], df[target]):
        yield x, y
```

## Integrated River Pipelines (Preprocessing + Model)

Use a single integrated preprocessor in notebook cells:

```python
from river import compose, preprocessing, linear_model, tree

def build_preprocessor():
    num = compose.Select(
        "humidity", "pressure", "wind_speed", "latitude", "longitude",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
        "wind_dir_sin", "wind_dir_cos"
    ) | preprocessing.StandardScaler()
    cat = compose.Select("city", "weather_description", "country") | preprocessing.OneHotEncoder()
    return num + cat
```

Implement at least three online models:

1. `build_preprocessor() | linear_model.LinearRegression()`
2. `build_preprocessor() | tree.HoeffdingTreeRegressor(...)` (required)
3. `build_preprocessor() | tree.HoeffdingAdaptiveTreeRegressor(...)`

## Prequential Evaluation Loop

- predict first
- update MAE/RMSE/R2
- update drift detectors on absolute error
- then learn (`learn_one`)

Track:

- cumulative metrics
- rolling-window metrics (e.g., 5000)
- periodic snapshots (e.g., every 1000 samples)

**Validation checkpoint:** All online pipelines run successfully end-to-end on chronological stream.

---

# Concept Drift Detection

## Detector 1: ADWIN

```python
from river.drift import ADWIN
adwin = ADWIN(delta=0.002)
```

## Detector 2: PageHinkley

```python
from river.drift import PageHinkley
ph = PageHinkley(delta=0.005, threshold=50, direction="up")
```

## Integration Steps

1. Feed `abs(y - y_pred)` to both detectors each sample.
2. Store drift indices and timestamps.
3. Map drift to month/season.
4. Tune detector parameters if detections are too sparse/noisy.

**Validation checkpoint:** Both detectors produce plausible, explainable drift events.

---

# Offline vs Online Comparison

## Implementation Steps

1. Build unified comparison table with MAE/RMSE/R2 across all models.
2. Compare:
   - offline static test metrics
   - online rolling/cumulative metrics over time
3. Contrast Hoeffding Tree vs Adaptive Hoeffding Tree near detected drift.
4. Link detector events to error spikes and adaptation behavior.

**Validation checkpoint:** Comparison explicitly connects performance behavior to drift evidence.

---

# Visualization

## Required Plots

1. Rolling MAE over time (all online models)
2. Drift timeline (error curve + detector event lines)
3. Offline vs online metric bar chart
4. Monthly performance heatmap
5. Temperature timeline for selected cities with drift annotations
6. Cumulative metric trajectories
7. Drift density histogram by month
8. Learning curves (error vs seen samples)

## Execution Notes

- Keep plotting utilities as notebook cells (not external modules).
- Use consistent color and labeling.
- Add short interpretation under each plot.

**Validation checkpoint:** Plots clearly support the drift narrative and model comparison.

---

# Results and Conclusions

## Implementation Steps

1. Summarize all model results in a final table.
2. Conclude whether concept drift is demonstrated and why.
3. Interpret detector agreement/disagreement.
4. Compare offline and online adaptability.
5. Discuss limitations:
   - forward-fill assumptions
   - rare-category grouping
   - single-target setup
6. Propose future work:
   - ensemble online methods
   - additional drift-aware adaptation
   - multi-target forecasting

**Validation checkpoint:** Conclusions are evidence-based and directly tied to metrics, drift events, and visualizations.

---

## End-to-End Execution Order (Single Notebook)

1. Problem Definition
2. Dataset Justification
3. Data Preparation
4. Offline (Batch) Learning
5. Online (Stream) Learning
6. Concept Drift Detection
7. Offline vs Online Comparison
8. Visualization
9. Results and Conclusions

This sequence must run top-to-bottom in `notebook.ipynb`.

---

## Potential Pitfalls and Mitigations

| Pitfall | Mitigation |
|---|---|
| Memory pressure from large one-hot expansions | Use sparse encodings and avoid unnecessary DataFrame copies. |
| Wind direction circularity | Use sine/cosine encoding. |
| Drift detector over/under sensitivity | Tune `delta`/`threshold`; compare ADWIN vs PageHinkley. |
| Slow online loop on 1M+ rows | Log metrics every N steps; avoid expensive per-step plotting. |
| Rare weather category explosion | Group rare categories into `other`. |
| Offline leakage risk | Use strict temporal split; never random shuffle. |
