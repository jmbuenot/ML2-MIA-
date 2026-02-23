# Machine Learning 2

## Lab Practice: Online Learning

**Course 2025–2026**

---

# Online Learning Project – Final Task Description

## General Objective

The main goal of this project is to **demonstrate concept drift** using a real-world temporal dataset.

To achieve this, you must:

- Develop an **online (stream) model** using the River library that simulates real-time data arrival.
- Develop an **offline (batch) model** to serve as a baseline.
- Compare online and offline results in order to analyze and demonstrate the presence (or absence) of concept drift.
- Implement **data preprocessing, model training, and evaluation as a single integrated pipeline** in the online setting.

This laboratory exercise is weighted at **4 points** in the final course grade (laboratory part).

---

## Dataset

You must use the **Historical Hourly Weather Data** dataset.

**Source:**  
https://www.kaggle.com/selfishgene/historical-hourly-weather-data

This dataset contains hourly weather measurements for 36 cities, including:

- Geographical attributes (latitude, longitude)
- Humidity (%)
- Pressure (hPa)
- Temperature (K)
- Weather descriptions (categorical)
- Wind direction (degrees)
- Wind speed (m/s)

### Dataset Structure & Samples

The dataset consists of several thematic files covering 36 cities. Below are samples of the data structure (first 3 rows):

- **City Attributes:** Includes geographical coordinates.
  - City,Country,Latitude,Longitude
  - Vancouver,Canada,49.24966,-123.119339
  - Portland,United States,45.523449,-122.676208

- **Humidity (%):** Hourly humidity levels.
  - datetime,Vancouver,Portland,San Francisco,Seattle,Los Angeles,San Diego,Las Vegas,Phoenix,Albuquerque,Denver,San Antonio,Dallas,Houston,Kansas City,Minneapolis,Saint Louis,Chicago,Nashville,Indianapolis,Atlanta,Detroit,Jacksonville,Charlotte,Miami,Pittsburgh,Toronto,Philadelphia,New York,Montreal,Boston,Beersheba,Tel Aviv District,Eilat,Haifa,Nahariyya,Jerusalem
  - 2012-10-01 12:00:00,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,25.0,,,
  - 2012-10-01 13:00:00,76.0,81.0,88.0,81.0,88.0,82.0,22.0,23.0,50.0,62.0,93.0,87.0,93.0,71.0,67.0,71.0,71.0,100.0,76.0,94.0,76.0,88.0,87.0,83.0,93.0,82.0,71.0,58.0,93.0,68.0,50.0,63.0,22.0,51.0,51.0,50.0

- **Pressure (hPa):** Hourly atmospheric pressure.
  - datetime,Vancouver,Portland,San Francisco,Seattle,Los Angeles,San Diego,Las Vegas,Phoenix,Albuquerque,Denver,San Antonio,Dallas,Houston,Kansas City,Minneapolis,Saint Louis,Chicago,Nashville,Indianapolis,Atlanta,Detroit,Jacksonville,Charlotte,Miami,Pittsburgh,Toronto,Philadelphia,New York,Montreal,Boston,Beersheba,Tel Aviv District,Eilat,Haifa,Nahariyya,Jerusalem
  - 2012-10-01 12:00:00,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,1011.0,,,
  - 2012-10-01 13:00:00,,1024.0,1009.0,1027.0,1013.0,1013.0,1018.0,1013.0,1024.0,1028.0,1014.0,1011.0,1009.0,1011.0,1012.0,1010.0,1014.0,1005.0,1011.0,1006.0,1016.0,1009.0,1012.0,1011.0,1015.0,1012.0,1014.0,1012.0,1001.0,1014.0,984.0,1012.0,1010.0,1013.0,1013.0,990.0

- **Temperature (K):** Hourly temperature in Kelvin.
  - datetime,Vancouver,Portland,San Francisco,Seattle,Los Angeles,San Diego,Las Vegas,Phoenix,Albuquerque,Denver,San Antonio,Dallas,Houston,Kansas City,Minneapolis,Saint Louis,Chicago,Nashville,Indianapolis,Atlanta,Detroit,Jacksonville,Charlotte,Miami,Pittsburgh,Toronto,Philadelphia,New York,Montreal,Boston,Beersheba,Tel Aviv District,Eilat,Haifa,Nahariyya,Jerusalem
  - 2012-10-01 12:00:00,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,309.1,,,
  - 2012-10-01 13:00:00,284.63,282.08,289.48,281.8,291.87,291.53,293.41,296.6,285.12,284.61,289.29,289.74,288.27,289.98,286.87,286.18,284.01,287.41,283.85,294.03,284.03,298.17,288.65,299.72,281.0,286.26,285.63,288.22,285.83,287.17,307.59,305.47,310.58,304.4,304.4,303.5

- **Weather Description:** Textual categorical descriptions.
  - datetime,Vancouver,Portland,San Francisco,Seattle,Los Angeles,San Diego,Las Vegas,Phoenix,Albuquerque,Denver,San Antonio,Dallas,Houston,Kansas City,Minneapolis,Saint Louis,Chicago,Nashville,Indianapolis,Atlanta,Detroit,Jacksonville,Charlotte,Miami,Pittsburgh,Toronto,Philadelphia,New York,Montreal,Boston,Beersheba,Tel Aviv District,Eilat,Haifa,Nahariyya,Jerusalem
  - 2012-10-01 12:00:00,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,haze,,,
  - 2012-10-01 13:00:00,mist,scattered clouds,light rain,sky is clear,mist,sky is clear,sky is clear,sky is clear,sky is clear,light rain,sky is clear,mist,sky is clear,sky is clear,broken clouds,sky is clear,overcast clouds,mist,overcast clouds,light rain,sky is clear,scattered clouds,mist,light intensity drizzle,mist,sky is clear,broken clouds,few clouds,overcast clouds,sky is clear,sky is clear,sky is clear,haze,sky is clear,sky is clear,sky is clear

- **Wind Direction (degrees):**
  - datetime,Vancouver,Portland,San Francisco,Seattle,Los Angeles,San Diego,Las Vegas,Phoenix,Albuquerque,Denver,San Antonio,Dallas,Houston,Kansas City,Minneapolis,Saint Louis,Chicago,Nashville,Indianapolis,Atlanta,Detroit,Jacksonville,Charlotte,Miami,Pittsburgh,Toronto,Philadelphia,New York,Montreal,Boston,Beersheba,Tel Aviv District,Eilat,Haifa,Nahariyya,Jerusalem
  - 2012-10-01 12:00:00,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,360.0,,,
  - 2012-10-01 13:00:00,0.0,0.0,150.0,0.0,0.0,0.0,0.0,10.0,360.0,20.0,0.0,340.0,270.0,0.0,330.0,40.0,0.0,70.0,40.0,110.0,0.0,180.0,70.0,200.0,0.0,240.0,270.0,260.0,230.0,60.0,135.0,101.0,30.0,336.0,336.0,329.0

- **Wind Speed (m/s):**
  - datetime,Vancouver,Portland,San Francisco,Seattle,Los Angeles,San Diego,Las Vegas,Phoenix,Albuquerque,Denver,San Antonio,Dallas,Houston,Kansas City,Minneapolis,Saint Louis,Chicago,Nashville,Indianapolis,Atlanta,Detroit,Jacksonville,Charlotte,Miami,Pittsburgh,Toronto,Philadelphia,New York,Montreal,Boston,Beersheba,Tel Aviv District,Eilat,Haifa,Nahariyya,Jerusalem
  - 2012-10-01 12:00:00,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,8.0,,,
  - 2012-10-01 13:00:00,0.0,0.0,2.0,0.0,0.0,0.0,0.0,2.0,4.0,4.0,0.0,3.0,1.0,0.0,3.0,4.0,0.0,4.0,4.0,3.0,0.0,3.0,4.0,3.0,0.0,3.0,4.0,7.0,4.0,3.0,1.0,0.0,8.0,2.0,2.0,2.0

The dataset has a clear **temporal structure**, making it suitable for stream learning and concept drift analysis.

### Modeling Strategy

The project will implement a **general model that takes all locations into account**.

Instead of training separate models per city, a single unified model will be trained using data from all available cities. The `city` attribute will be treated as a categorical feature and included in the model through appropriate encoding.

## This design choice increases the complexity of the problem and allows the model to learn shared patterns across different geographical regions, while also enabling the analysis of potential geographical and temporal concept drift.

## Project Requirements

Your notebook must include the following components:

---

### 1. Problem Definition

- Provide an informal description of the problem (in non-ML terms).
- Clearly define the ML task:
  - Is it classification, regression, or clustering?
  - Is the dataset imbalanced?
  - Could it be affected by concept drift?
- Define appropriate evaluation metrics.
- State any relevant assumptions.

### Scope of the Model

The model will be trained on data from **all cities simultaneously**, forming a global predictive model.

This means:

- The city identifier will be included as an input feature.
- The model must generalize across different climatic regions.
- Concept drift may arise not only from seasonal changes, but also from differences between geographical locations.

## This setup aligns with the objective of demonstrating concept drift in a realistic multi-source streaming scenario.

### 2. Dataset Justification

- Explain why the selected dataset is suitable for stream learning.
- Describe any preparation required before modeling.

---

### 3. Data Preparation

- Describe necessary data type conversions for River.
- Perform normalization/standardization if required and justify it.
- Apply one-hot encoding if needed (for nominal or multiclass features).
- Define any new engineered features if necessary.
- Clearly explain preprocessing steps.

In the **online setting**, preprocessing must be integrated inside a **single River pipeline** together with the model and evaluation.

---

### 4. Concept Drift Detection

- Implement at least **two concept drift detectors**.
- Justify why these detectors were selected.
- Clearly demonstrate concept drift using experimental results and visualizations.

---

### 5. Offline (Batch) Learning

- Perform a proper train/test split (stratified if required).
- Train at least one baseline model.
- Optionally tune hyperparameters.
- Ensure no test data leakage.
- Compare multiple models if appropriate.
- Use cross-validation if relevant.

Offline results must later be compared with online results to analyze drift behavior.

---

### 6. Online (Stream) Learning

- Implement at least **three stream learning pipelines**.
- One of the models must be a **Hoeffding Tree**.
- Use appropriate performance metrics from River.
- Clearly simulate real-time data arrival.
- Ensure preprocessing, model, and evaluation are implemented as a single integrated pipeline.

---

### 7. Visualization

Include plots that:

- Show model performance over time.
- Illustrate detected concept drifts (if present).
- Compare offline vs. online performance.

---

### 8. Results and Conclusions

- Clearly interpret results.
- Explicitly analyze concept drift behavior.
- Compare offline and online models.
- Discuss limitations and possible future work.

---

## Evaluation Criteria

The project will be evaluated based on:

- Clarity of problem formulation
- Correct implementation of stream learning
- Proper concept drift detection
- Quality of pipelines
- Correct comparison between offline and online models
- Quality of visualizations
- Soundness of conclusions
- Organization and clarity of the notebook
- Oral presentation quality
