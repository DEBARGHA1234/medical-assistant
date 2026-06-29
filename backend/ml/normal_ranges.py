# Normal Reference Ranges for Medical Lab Parameters

NORMAL_RANGES = {
    "hemoglobin": {
        "male": (13.5, 17.5),
        "female": (12.0, 15.5),
        "unit": "g/dL"
    },
    "WBC": {
        "all": (4500, 11000),
        "unit": "cells/mcL"
    },
    "RBC": {
        "male": (4.5, 5.9),
        "female": (4.1, 5.1),
        "unit": "million cells/mcL"
    },
    "platelets": {
        "all": (150000, 400000),
        "unit": "/mcL"
    },
    "MCV": {
        "all": (80, 100),
        "unit": "fL"
    },
    "MCH": {
        "all": (27, 33),
        "unit": "pg"
    },
    "MCHC": {
        "all": (32, 36),
        "unit": "g/dL"
    },
    "blood_sugar_fasting": {
        "all": (70, 99),
        "unit": "mg/dL"
    },
    "HbA1c": {
        "all": (0.0, 5.6),
        "unit": "%"
    },
    "creatinine": {
        "male": (0.7, 1.3),
        "female": (0.6, 1.1),
        "unit": "mg/dL"
    },
    "uric_acid": {
        "male": (3.5, 7.2),
        "female": (2.6, 6.0),
        "unit": "mg/dL"
    },
    "albumin": {
        "all": (3.4, 5.4),
        "unit": "g/dL"
    },
    "cholesterol_total": {
        "all": (0.0, 199.0),
        "unit": "mg/dL"
    },
    "triglycerides": {
        "all": (0.0, 149.0),
        "unit": "mg/dL"
    },
    "TSH": {
        "all": (0.4, 4.0),
        "unit": "mIU/L"
    },
    "T4": {
        "all": (4.6, 12.0),
        "unit": "ug/dL"
    },
    "FTI": {
        "all": (4.5, 12.5),
        "unit": "ug/dL"
    },
    "ALT": {
        "all": (7, 56),
        "unit": "U/L"
    },
    "AST": {
        "all": (10, 40),
        "unit": "U/L"
    },
    "bilirubin_total": {
        "all": (0.1, 1.2),
        "unit": "mg/dL"
    },
    "vitamin_D": {
        "all": (20.0, 50.0),
        "unit": "ng/mL"
    },
    "vitamin_B12": {
        "all": (200.0, 900.0),
        "unit": "pg/mL"
    },
    "BP_systolic": {
        "all": (0, 119),
        "unit": "mmHg"
    },
    "BP_diastolic": {
        "all": (0, 79),
        "unit": "mmHg"
    }
}


def check_value(name: str, value: float, gender: str = "male") -> dict:
    """
    Checks if a given value is low, high, or normal.
    Returns a dict with 'status' (normal, low, high) and the reference range.
    """
    gender = gender.lower() if gender else "male"
    if gender not in ["male", "female"]:
        gender = "male"

    if name not in NORMAL_RANGES:
        return {"status": "normal", "range": "N/A", "unit": ""}

    ranges = NORMAL_RANGES[name]
    ref_range = ranges.get(gender, ranges.get("all"))
    unit = ranges.get("unit", "")

    if not ref_range or value == -1:
        return {"status": "normal", "range": "N/A", "unit": unit}

    low, high = ref_range
    if value < low:
        return {"status": "low", "range": f"{low} - {high}", "unit": unit}
    elif value > high:
        return {"status": "high", "range": f"{low} - {high}", "unit": unit}
    else:
        return {"status": "normal", "range": f"{low} - {high}", "unit": unit}
