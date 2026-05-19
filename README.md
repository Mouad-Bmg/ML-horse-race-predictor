# Horse Race Outcome Predictor

A machine learning pipeline that predicts whether a horse will finish in the top 3 of a PMU race, trained on a full season of French racing data and queryable against live race fields via the official PMU API.

## What it does

Given an upcoming race, the system:

- Fetches live participant data from the PMU API (starting odds, post position, weight handicap, age, sex, blinker equipment, race number)
- Preprocesses and encodes the feature set using a trained pipeline
- Outputs a Top 3 / Not Top 3 prediction for each horse along with a confidence score
- Returns a ranked table sorted by predicted probability

The model was trained on historical 2023 race data and validated on a held-out test set.

## Stack

- **Language:** Python
- **ML:** scikit-learn (Logistic Regression with class balancing)
- **Data source:** PMU public API + local CSV dataset (2023 season)
- **Preprocessing:** StandardScaler for numerics, OneHotEncoder for categoricals, SimpleImputer for missing values
- **Serialization:** joblib

## Features used

| Feature | Type | Description |
|---|---|---|
| `cote` | Numeric | Starting odds |
| `corde` | Categorical | Post position (rail distance) |
| `poids_hand` | Numeric | Weight handicap |
| `course` | Numeric | Race number within the meeting |
| `age` | Numeric | Horse age |
| `sexe` | Categorical | Sex of the horse |
| `oeilleres` | Categorical | Blinker equipment |

## Sample output

```
predict_top3('20042024', 2, 1)

   Cheval          Cote    Prediction    Reliability
   KEANU           2.7     Top 3         76.97%
   HELLO JEEBY     4.6     Top 3         73.12%
   REUX            5.7     Top 3         72.51%
   O SOLE MIO      8.5     Top 3         64.70%
   ...
```

## Limitations and next steps

Logistic regression establishes a solid baseline but captures only linear relationships in the data. Planned improvements include gradient boosting models (XGBoost, LightGBM), addition of jockey and trainer historical performance features, track condition and weather variables, and a proper backtesting framework to evaluate betting simulation returns.

## Usage

```python
# Predict top 3 finishers for reunion 2, race 1 on April 20 2024
predict_top3('20042024', 2, 1)
```
