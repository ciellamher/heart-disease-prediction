# Data dictionary

Dataset: UCI Heart Disease, Cleveland subset (`data/raw/processed.cleveland.data`), 303 rows × 14 columns.

| Column | Meaning | Type | Values / units |
|---|---|---|---|
| `age` | Age | Numeric (int) | Years |
| `sex` | Sex | Categorical (binary) | 1 = male, 0 = female |
| `cp` | Chest pain type | Categorical | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic |
| `trestbps` | Resting blood pressure on admission | Numeric | mm Hg |
| `chol` | Serum cholesterol | Numeric | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | Categorical (binary) | 1 = true, 0 = false |
| `restecg` | Resting ECG result | Categorical | 0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved (exercise test) | Numeric | Beats per minute |
| `exang` | Exercise-induced angina | Categorical (binary) | 1 = yes, 0 = no |
| `oldpeak` | ST depression induced by exercise relative to rest | Numeric (float) | mm |
| `slope` | Slope of the peak exercise ST segment | Categorical (ordinal) | 1 = upsloping, 2 = flat, 3 = downsloping |
| `ca` | Number of major vessels colored by fluoroscopy | Numeric (ordinal) | 0–3; **4 missing** (`?`) |
| `thal` | Thallium stress test result | Categorical | 3 = normal, 6 = fixed defect, 7 = reversible defect; **2 missing** (`?`) |
| `num` | Diagnosis of heart disease (angiographic) | Ordinal | 0 = < 50% vessel narrowing (no disease), 1–4 = > 50% narrowing (increasing severity) |

## Target

**`target`** (derived, binary) = `1` if `num` > 0, else `0`.
This follows the standard practice for this dataset: published work focuses on presence (1–4) vs. absence (0) of disease rather than predicting the severity level.

## Notes

- All values in the raw file are stored as floats (e.g. `63.0`); categorical codes are cast to integers during cleaning.
- `?` marks missing values and is read as `NaN` via `na_values="?"`.
- Column names are not in the raw file; they are assigned in `src/data_loading.py`.
