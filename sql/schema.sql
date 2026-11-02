-- Schema for heart.db (SQLite)
--
-- Design:
--   * One row per patient in `patients` (the fact table). Every measurement
--     depends only on patient_id, so the table is in third normal form.
--   * Multi-class category codes (cp, restecg, slope, thal) are foreign keys
--     into small lookup tables that hold the human-readable labels. The codes
--     stay as the UCI integers so the data matches the source file.
--   * Binary flags (sex, fbs, exang) use CHECK (x IN (0, 1)) instead of a
--     lookup table; their meaning is spelled out in the `patients_labeled` view.
--   * `target` is a generated column computed from `num`, so the derived
--     value can never disagree with the diagnosis it comes from.
--   * CHECK constraints reject impossible values at insert time.

PRAGMA foreign_keys = ON;

DROP VIEW  IF EXISTS patients_labeled;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS chest_pain_types;
DROP TABLE IF EXISTS resting_ecg_results;
DROP TABLE IF EXISTS st_slopes;
DROP TABLE IF EXISTS thal_results;

-- Lookup tables --------------------------------------------------------------

CREATE TABLE chest_pain_types (
    code  INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE
);

CREATE TABLE resting_ecg_results (
    code  INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE
);

CREATE TABLE st_slopes (
    code  INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE
);

CREATE TABLE thal_results (
    code  INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE
);

-- Fact table -----------------------------------------------------------------

CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,  -- line number in the raw UCI file
    age        INTEGER NOT NULL CHECK (age BETWEEN 0 AND 120),
    sex        INTEGER NOT NULL CHECK (sex IN (0, 1)),            -- 1 = male
    cp         INTEGER NOT NULL REFERENCES chest_pain_types (code),
    trestbps   INTEGER NOT NULL CHECK (trestbps > 0),             -- mm Hg
    chol       INTEGER NOT NULL CHECK (chol > 0),                 -- mg/dl
    fbs        INTEGER NOT NULL CHECK (fbs IN (0, 1)),            -- 1 = > 120 mg/dl
    restecg    INTEGER NOT NULL REFERENCES resting_ecg_results (code),
    thalach    INTEGER NOT NULL CHECK (thalach > 0),              -- bpm
    exang      INTEGER NOT NULL CHECK (exang IN (0, 1)),          -- 1 = yes
    oldpeak    REAL    NOT NULL CHECK (oldpeak >= 0),             -- mm
    slope      INTEGER NOT NULL REFERENCES st_slopes (code),
    ca         INTEGER NOT NULL CHECK (ca BETWEEN 0 AND 3),
    thal       INTEGER NOT NULL REFERENCES thal_results (code),
    num        INTEGER NOT NULL CHECK (num BETWEEN 0 AND 4),
    target     INTEGER GENERATED ALWAYS AS (CASE WHEN num > 0 THEN 1 ELSE 0 END) VIRTUAL
);

CREATE INDEX idx_patients_target ON patients (target);

-- Readable view --------------------------------------------------------------

CREATE VIEW patients_labeled AS
SELECT
    p.patient_id,
    p.age,
    CASE p.sex   WHEN 1 THEN 'male' ELSE 'female' END AS sex,
    cp.label                                         AS chest_pain_type,
    p.trestbps,
    p.chol,
    CASE p.fbs   WHEN 1 THEN 'true' ELSE 'false' END AS fasting_blood_sugar_gt_120,
    ecg.label                                        AS resting_ecg,
    p.thalach,
    CASE p.exang WHEN 1 THEN 'yes' ELSE 'no' END     AS exercise_angina,
    p.oldpeak,
    sl.label                                         AS st_slope,
    p.ca,
    th.label                                         AS thal,
    p.num,
    p.target
FROM patients p
JOIN chest_pain_types    cp  ON cp.code  = p.cp
JOIN resting_ecg_results ecg ON ecg.code = p.restecg
JOIN st_slopes           sl  ON sl.code  = p.slope
JOIN thal_results        th  ON th.code  = p.thal;
