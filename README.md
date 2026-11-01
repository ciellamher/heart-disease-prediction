# Heart Disease Prediction

An interpretable machine-learning project that predicts heart-disease risk from clinical indicators.

## Problem

Heart-disease risk is often spotted late, and black-box models are hard for clinicians to trust.

## Solution

An interpretable model that predicts the presence of heart disease from routine clinical measurements (age, blood pressure, cholesterol, max heart rate, ECG and stress-test results), with clear explanations of which features drive each prediction.

## Planned features

- [x] Project setup and data dictionary *(Week 1)*
- [x] Initial data audit: shape, types, missing values, duplicates *(Week 1)*
- [ ] Data cleaning and exploratory data analysis with pandas
- [ ] SQL queries on a SQLite copy of the dataset
- [ ] Baseline models with accuracy, recall and ROC-AUC
- [ ] Feature-importance charts
- [ ] Simple results dashboard

## Dataset

[UCI Heart Disease, Cleveland subset](https://archive.ics.uci.edu/dataset/45/heart+disease): 303 patients, 13 clinical features, plus a diagnosis column. See [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) for every column.

## Project structure

```
heart-disease-prediction/
├── data/
│   ├── raw/               # original, untouched data
│   ├── processed/         # cleaned data (generated, not committed)
│   └── DATA_DICTIONARY.md
├── notebooks/
│   └── 01_data_overview.ipynb
├── src/
│   └── data_loading.py    # reusable loading helpers
├── reports/figures/
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/ciellamher/heart-disease-prediction.git
cd heart-disease-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_data_overview.ipynb
```

## Tools

Python · pandas · SQL / SQLite · scikit-learn · Matplotlib · Git

## Dataset citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X (CC BY 4.0)
