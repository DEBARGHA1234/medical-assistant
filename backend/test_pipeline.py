import os
import json
import sqlite3
from datetime import datetime

from ml.normal_ranges import check_value, NORMAL_RANGES
from ml.pipeline import extract_lab_values, run_ml_pipeline, load_models
from database import engine_sqlite, Base, User, HealthProfile, Report, SessionLocalSQLite

def test_regex_extraction():
    print("Testing Regex Lab Value Extractor...")
    report_text = """
    LABORATORY TEST REPORT
    Patient: John Doe
    Date: 2026-06-29
    
    Hemoglobin: 11.2 g/dL
    Blood Sugar (Fasting): 145 mg/dL
    HbA1c: 7.2 %
    Creatinine: 1.1 mg/dL
    TSH: 3.1 mIU/L
    BP: 142/95 mmHg
    """
    
    vals = extract_lab_values(report_text)
    
    assert vals["hemoglobin"] == 11.2, f"Expected 11.2, got {vals['hemoglobin']}"
    assert vals["blood_sugar_fasting"] == 145.0, f"Expected 145.0, got {vals['blood_sugar_fasting']}"
    assert vals["HbA1c"] == 7.2, f"Expected 7.2, got {vals['HbA1c']}"
    assert vals["creatinine"] == 1.1, f"Expected 1.1, got {vals['creatinine']}"
    assert vals["TSH"] == 3.1, f"Expected 3.1, got {vals['TSH']}"
    assert vals["BP_systolic"] == 142.0, f"Expected 142.0, got {vals['BP_systolic']}"
    assert vals["BP_diastolic"] == 95.0, f"Expected 95.0, got {vals['BP_diastolic']}"
    
    print("[OK] Regex Lab Value Extractor passed successfully.")

def test_ml_pipeline_execution():
    print("Testing ML Pipeline Inference and Safety Constraints...")
    
    # Check that models load correctly
    load_models()
    
    report_text = """
    Patient details: Male, 42 years old.
    Fasting Blood Sugar: 165 mg/dL, HbA1c: 7.8 %, Hemoglobin: 10.5 g/dL
    Notes: Patient complains of fatigue and polyuria.
    """
    
    profile = {
        "age": 42,
        "gender": "male",
        "chronic_conditions": ["hypertension"]
    }
    
    diagnosis = run_ml_pipeline(report_text, profile, previous_reports=[])
    
    # Safety Constraint: Never show a confidence score above 89%
    for condition in diagnosis["top_conditions"]:
        assert condition["confidence"] <= 0.89, f"Capping failed! Got {condition['confidence']} for {condition['name']}"
        
    print(f"Severity predicted: {diagnosis['severity']}")
    assert diagnosis["severity"] in ["mild", "moderate", "serious"]
    
    # Verify abnormal values list
    abnormals = diagnosis["abnormal_values"]
    assert len(abnormals) > 0, "Expected abnormal values to be flagged"
    
    # Check fasting sugar flag
    sugar_flag = next((v for v in abnormals if v["name"] == "blood_sugar_fasting"), None)
    assert sugar_flag is not None, "Fasting sugar should be flagged"
    assert sugar_flag["status"] == "high", f"Expected high status, got {sugar_flag['status']}"
    
    print("[OK] ML Pipeline and safety constraints passed successfully.")

def test_sqlite_database_sync():
    print("Testing local SQLite database operations...")
    
    # Initialize DB session
    db = SessionLocalSQLite()
    
    # Clear test rows if exist
    db.query(Report).filter(Report.user_id == "test_user_id").delete()
    db.query(User).filter(User.id == "test_user_id").delete()
    db.commit()
    
    # Add test user
    user = User(
        id="test_user_id",
        email="test@mediai.in",
        name="Test Patient",
        subscription_tier="pro"
    )
    db.add(user)
    db.commit()
    
    # Add test report
    report = Report(
        id="test_report_id",
        user_id="test_user_id",
        file_url="http://supabase.com/test.pdf",
        raw_text="Hemoglobin 11.2",
        extracted_values=json.dumps([{"name": "hemoglobin", "value": 11.2}]),
        ml_diagnosis=json.dumps({"severity": "moderate"}),
        severity="moderate"
    )
    db.add(report)
    db.commit()
    
    # Query check
    retrieved_user = db.query(User).filter_by(id="test_user_id").first()
    retrieved_report = db.query(Report).filter_by(id="test_report_id").first()
    
    assert retrieved_user.name == "Test Patient"
    assert retrieved_report.severity == "moderate"
    
    # Clean up
    db.query(Report).filter(Report.user_id == "test_user_id").delete()
    db.query(User).filter(User.id == "test_user_id").delete()
    db.commit()
    db.close()
    
    print("[OK] SQLite Database operations passed successfully.")

if __name__ == "__main__":
    print("--------------------------------------------------")
    print("RUNNING DIAGNOSTIC TEST SUITE FOR MEDIAI BACKEND")
    print("--------------------------------------------------")
    test_regex_extraction()
    test_ml_pipeline_execution()
    test_sqlite_database_sync()
    print("--------------------------------------------------")
    print("ALL TESTS PASSED SUCCESSFULLY! Ready for deployment.")
    print("--------------------------------------------------")
