-- Exploratory SQL queries on heart.db
--
-- Run against the database built by `python -m src.database`:
--     sqlite3 data/processed/heart.db < sql/analysis.sql
--
-- Each query starts with a "-- name:" line so notebooks/03_eda.ipynb can load
-- them one at a time. "Risk" = share of patients with heart disease (target = 1).


-- name: overall_disease_rate
-- Baseline: how common is heart disease in the cleaned data?
SELECT COUNT(*)                     AS patients,
       SUM(target)                  AS with_disease,
       ROUND(AVG(target) * 100, 1)  AS disease_pct
FROM patients;


-- name: risk_by_age_group
-- Does risk rise with age? Ten-year bands, with the small extremes merged.
SELECT CASE
           WHEN age < 45 THEN '1) under 45'
           WHEN age < 55 THEN '2) 45-54'
           WHEN age < 65 THEN '3) 55-64'
           ELSE               '4) 65+'
       END                          AS age_group,
       COUNT(*)                     AS patients,
       SUM(target)                  AS with_disease,
       ROUND(AVG(target) * 100, 1)  AS disease_pct
FROM patients
GROUP BY age_group
ORDER BY age_group;


-- name: risk_by_sex
SELECT sex,
       COUNT(*)                     AS patients,
       SUM(target)                  AS with_disease,
       ROUND(AVG(target) * 100, 1)  AS disease_pct
FROM patients_labeled
GROUP BY sex
ORDER BY disease_pct DESC;


-- name: risk_by_cholesterol_range
-- Bands follow the NCEP ATP III guideline for total cholesterol (mg/dl).
SELECT CASE
           WHEN chol < 200 THEN '1) desirable (<200)'
           WHEN chol < 240 THEN '2) borderline (200-239)'
           ELSE                 '3) high (240+)'
       END                          AS cholesterol_range,
       COUNT(*)                     AS patients,
       ROUND(AVG(chol), 0)          AS avg_chol,
       ROUND(AVG(target) * 100, 1)  AS disease_pct
FROM patients
GROUP BY cholesterol_range
ORDER BY cholesterol_range;


-- name: risk_by_chest_pain_type
-- Joins the lookup table directly (instead of the view) to show the foreign key.
SELECT c.label                        AS chest_pain_type,
       COUNT(*)                       AS patients,
       ROUND(AVG(p.target) * 100, 1)  AS disease_pct
FROM patients p
JOIN chest_pain_types c ON c.code = p.cp
GROUP BY c.code, c.label
ORDER BY disease_pct DESC;


-- name: risk_by_sex_and_chest_pain
-- Does the chest-pain pattern hold for both sexes? HAVING hides tiny groups.
SELECT sex,
       chest_pain_type,
       COUNT(*)                     AS patients,
       ROUND(AVG(target) * 100, 1)  AS disease_pct
FROM patients_labeled
GROUP BY sex, chest_pain_type
HAVING COUNT(*) >= 10
ORDER BY sex, disease_pct DESC;


-- name: exercise_test_by_outcome
-- Exercise-test results side by side for patients with and without disease.
SELECT CASE target WHEN 1 THEN 'disease' ELSE 'no disease' END AS outcome,
       COUNT(*)                                 AS patients,
       ROUND(AVG(thalach), 1)                   AS avg_max_heart_rate,
       ROUND(AVG(oldpeak), 2)                   AS avg_st_depression,
       ROUND(AVG(exang) * 100, 1)               AS exercise_angina_pct,
       ROUND(AVG(ca), 2)                        AS avg_vessels_colored
FROM patients
GROUP BY target
ORDER BY target;


-- name: risk_by_vessels_colored
-- Number of major vessels seen on fluoroscopy (0-3) vs. share with disease.
SELECT ca                           AS vessels_colored,
       COUNT(*)                     AS patients,
       ROUND(AVG(target) * 100, 1)  AS disease_pct
FROM patients
GROUP BY ca
ORDER BY ca;
