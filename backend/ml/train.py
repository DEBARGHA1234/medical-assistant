import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Make directories if they don't exist
os.makedirs("ml/models", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# List of 132 symptoms for Model 1 (from Kaggle disease-prediction-using-machine-learning)
SYMPTOMS_LIST = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain",
    "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss",
    "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes",
    "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", "dark_urine",
    "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea",
    "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs",
    "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs",
    "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "excessive_hunger", "extra_marital_contacts", "drying_of_eyes_and_mouth", "hip_joint_pain", "muscle_weakness",
    "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness",
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine",
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_typhoid",
    "depression", "irritability", "muscle_pain", "altered_sensory_details", "red_spots_over_body", "belly_pain",
    "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", "increased_appetite", "polyuria", "family_history",
    "mucoid_sputum", "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload_2", "blood_in_sputum", "prominent_veins_on_calf",
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurrying", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister", "red_sore_around_nose",
    "yellow_crust_ooze"
]

# List of 42 diseases for Model 1
DISEASES_LIST = [
    "Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction", "Peptic ulcer diseae", "AIDS",
    "Diabetes", "Gastroenteritis", "Bronchial Asthma", "Hypertension", "Migraine", "Cervical spondylosis",
    "Paralysis (brain hemorrhage)", "Jaundice", "Malaria", "Chicken pox", "Dengue", "Typhoid", "hepatitis A",
    "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis", "Tuberculosis",
    "Common Cold", "Pneumonia", "Dimorphic hemmorhoids(piles)", "Heart attack", "Varicose veins", "Hypothyroidism",
    "Hyperthyroidism", "Hypoglycemia", "Osteoarthristis", "Arthritis", "(vertigo) Paroymsal  Positional Vertigo",
    "Acne", "Urinary tract infection", "Psoriasis", "Impetigo"
]

print("Starting training script. Generating synthetic data fallbacks...")

# =========================================================================
# MODEL 1 — General Disease Classifier
# =========================================================================
print("Training Model 1: General Disease Classifier...")
# Create synthetic symptom data
num_samples = 1000
X1 = np.random.choice([0, 1], size=(num_samples, len(SYMPTOMS_LIST)), p=[0.95, 0.05])
y1 = np.random.choice(DISEASES_LIST, size=num_samples)

# Make sure symptoms align slightly with diseases (mock logic)
for i in range(num_samples):
    # e.g., if disease is Diabetes, set irregular_sugar_level, excessive_hunger to 1
    if y1[i] == "Diabetes":
        X1[i, SYMPTOMS_LIST.index("irregular_sugar_level")] = 1
        X1[i, SYMPTOMS_LIST.index("excessive_hunger")] = 1
    elif y1[i] == "Hypothyroidism":
        X1[i, SYMPTOMS_LIST.index("weight_gain")] = 1
        X1[i, SYMPTOMS_LIST.index("lethargy")] = 1

model1 = RandomForestClassifier(n_estimators=50, random_state=42)
model1.fit(X1, y1)

with open("ml/models/general_disease_model.pkl", "wb") as f:
    pickle.dump({"model": model1, "symptoms": SYMPTOMS_LIST, "diseases": DISEASES_LIST}, f)


# =========================================================================
# MODEL 2 — Blood Report Analyzer
# =========================================================================
print("Training Model 2: Blood Report Analyzer...")
# Inputs: hemoglobin, WBC, RBC, platelets, MCV, MCH, MCHC
# Outputs: anemia type + blood disorder flags
# Categories: "Normal", "Iron Deficiency Anemia", "Megaloblastic Anemia", "Thalassemia", "Infection/Leukopenia"
blood_diseases = ["Normal", "Iron Deficiency Anemia", "Megaloblastic Anemia", "Thalassemia", "Infection/Leukopenia"]
X2_data = []
y2_data = []

# Generate normal blood reports
for _ in range(200):
    X2_data.append([np.random.uniform(12.0, 16.5), np.random.uniform(4500, 11000), np.random.uniform(4.1, 5.5), np.random.uniform(150000, 400000), np.random.uniform(80, 100), np.random.uniform(27, 33), np.random.uniform(32, 36)])
    y2_data.append("Normal")

# Iron Deficiency (Low hemoglobin, low MCV, low MCH, low RBC)
for _ in range(100):
    X2_data.append([np.random.uniform(8.0, 11.5), np.random.uniform(4500, 11000), np.random.uniform(3.0, 4.0), np.random.uniform(150000, 450000), np.random.uniform(60, 79), np.random.uniform(18, 26), np.random.uniform(28, 31)])
    y2_data.append("Iron Deficiency Anemia")

# Megaloblastic Anemia (Low hemoglobin, high MCV, normal MCH, low RBC)
for _ in range(100):
    X2_data.append([np.random.uniform(7.5, 11.0), np.random.uniform(3500, 9000), np.random.uniform(2.5, 3.8), np.random.uniform(120000, 350000), np.random.uniform(101, 125), np.random.uniform(28, 35), np.random.uniform(31, 35)])
    y2_data.append("Megaloblastic Anemia")

# Thalassemia (Low hemoglobin, extremely low MCV, normal MCH, high/normal RBC)
for _ in range(100):
    X2_data.append([np.random.uniform(8.5, 11.0), np.random.uniform(4500, 11000), np.random.uniform(4.5, 6.0), np.random.uniform(150000, 400000), np.random.uniform(55, 75), np.random.uniform(18, 25), np.random.uniform(29, 32)])
    y2_data.append("Thalassemia")

# Infection/Leukopenia (Abnormal WBC)
for _ in range(100):
    X2_data.append([np.random.uniform(11.0, 15.0), np.random.uniform(15000, 35000), np.random.uniform(4.0, 5.0), np.random.uniform(150000, 400000), np.random.uniform(80, 100), np.random.uniform(27, 33), np.random.uniform(32, 36)])
    y2_data.append("Infection/Leukopenia")

X2 = np.array(X2_data)
y2 = np.array(y2_data)

scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

model2 = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
model2.fit(X2_scaled, y2)

with open("ml/models/blood_report_model.pkl", "wb") as f:
    pickle.dump({"model": model2, "scaler": scaler2, "features": ["hemoglobin", "WBC", "RBC", "platelets", "MCV", "MCH", "MCHC"], "classes": blood_diseases}, f)


# =========================================================================
# MODEL 3 — Cardiac Risk Model
# =========================================================================
print("Training Model 3: Cardiac Risk Model...")
# Inputs: cholesterol_total, BP_systolic, BP_diastolic, age, gender (0=female, 1=male), chest_pain_type (0-3), ECG (0-2)
# Output: Cardiac risk category ("Low", "Medium", "High")
cardiac_classes = ["Low", "Medium", "High"]
X3_data = []
y3_data = []

for _ in range(600):
    age = np.random.uniform(25, 80)
    gender = np.random.choice([0, 1])
    chol = np.random.uniform(150, 320)
    bps = np.random.uniform(95, 180)
    bpd = np.random.uniform(60, 110)
    cp = np.random.choice([0, 1, 2, 3])
    ecg = np.random.choice([0, 1, 2])
    
    # Simple risk logic
    risk_score = 0
    if age > 55: risk_score += 2
    if chol > 240: risk_score += 2
    if bps > 140 or bpd > 90: risk_score += 2
    if cp > 0: risk_score += 3
    if ecg > 0: risk_score += 1
    
    if risk_score <= 3:
        label = "Low"
    elif risk_score <= 6:
        label = "Medium"
    else:
        label = "High"
        
    X3_data.append([chol, bps, bpd, age, gender, cp, ecg])
    y3_data.append(label)

X3 = np.array(X3_data)
y3 = np.array(y3_data)

scaler3 = StandardScaler()
X3_scaled = scaler3.fit_transform(X3)

model3 = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
model3.fit(X3_scaled, y3)

with open("ml/models/cardiac_model.pkl", "wb") as f:
    pickle.dump({"model": model3, "scaler": scaler3, "features": ["cholesterol_total", "BP_systolic", "BP_diastolic", "age", "gender", "chest_pain_type", "ECG"], "classes": cardiac_classes}, f)


# =========================================================================
# MODEL 4 — Stroke Risk Model
# =========================================================================
print("Training Model 4: Stroke Risk Model...")
# Inputs: age, hypertension, heart_disease, blood_sugar_fasting, BMI, smoking (0=no, 1=yes)
# Output: Stroke probability (classifier: "No Stroke Risk", "Stroke Risk")
stroke_classes = ["No Stroke Risk", "Stroke Risk"]
X4_data = []
y4_data = []

for _ in range(500):
    age = np.random.uniform(20, 85)
    hyper = np.random.choice([0, 1], p=[0.8, 0.2])
    hd = np.random.choice([0, 1], p=[0.9, 0.1])
    glucose = np.random.uniform(70, 250)
    bmi = np.random.uniform(18, 40)
    smoking = np.random.choice([0, 1], p=[0.7, 0.3])
    
    # Stroke logic
    score = 0
    if age > 60: score += 3
    if hyper: score += 2
    if hd: score += 2
    if glucose > 140: score += 2
    if bmi > 30: score += 1
    if smoking: score += 1
    
    label = "Stroke Risk" if score >= 5 else "No Stroke Risk"
    X4_data.append([age, hyper, hd, glucose, bmi, smoking])
    y4_data.append(label)

X4 = np.array(X4_data)
y4 = np.array(y4_data)

scaler4 = StandardScaler()
X4_scaled = scaler4.fit_transform(X4)

model4 = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
model4.fit(X4_scaled, y4)

with open("ml/models/stroke_model.pkl", "wb") as f:
    pickle.dump({"model": model4, "scaler": scaler4, "features": ["age", "hypertension", "heart_disease", "blood_sugar_fasting", "BMI", "smoking"], "classes": stroke_classes}, f)


# =========================================================================
# MODEL 5 — Diabetes & Metabolic Model
# =========================================================================
print("Training Model 5: Diabetes & Metabolic Model...")
# Inputs: HbA1c, blood_sugar_fasting, BMI, age, hypertension
# Output: "Normal", "Pre-diabetic", "Diabetic"
diab_classes = ["Normal", "Pre-diabetic", "Diabetic"]
X5_data = []
y5_data = []

for _ in range(600):
    hba1c = np.random.uniform(4.5, 9.5)
    glucose = np.random.uniform(60, 280)
    bmi = np.random.uniform(17, 42)
    age = np.random.uniform(18, 80)
    hyper = np.random.choice([0, 1], p=[0.75, 0.25])
    
    if hba1c >= 6.5 or glucose >= 126:
        label = "Diabetic"
    elif 5.7 <= hba1c < 6.5 or 100 <= glucose < 126:
        label = "Pre-diabetic"
    else:
        label = "Normal"
        
    X5_data.append([hba1c, glucose, bmi, age, hyper])
    y5_data.append(label)

X5 = np.array(X5_data)
y5 = np.array(y5_data)

scaler5 = StandardScaler()
X5_scaled = scaler5.fit_transform(X5)

model5 = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
model5.fit(X5_scaled, y5)

with open("ml/models/diabetes_model.pkl", "wb") as f:
    pickle.dump({"model": model5, "scaler": scaler5, "features": ["HbA1c", "blood_sugar_fasting", "BMI", "age", "hypertension"], "classes": diab_classes}, f)


# =========================================================================
# MODEL 6 — Thyroid Model
# =========================================================================
print("Training Model 6: Thyroid Model...")
# Inputs: TSH, T4, FTI, T3 (simulated), age, gender
# Output: "Normal", "Hypothyroidism", "Hyperthyroidism"
thyroid_classes = ["Normal", "Hypothyroidism", "Hyperthyroidism"]
X6_data = []
y6_data = []

for _ in range(500):
    tsh = np.random.uniform(0.1, 15.0)
    t4 = np.random.uniform(2.0, 18.0)
    fti = np.random.uniform(2.0, 18.0)
    t3 = np.random.uniform(0.5, 4.5)
    age = np.random.uniform(18, 75)
    gender = np.random.choice([0, 1])
    
    if tsh > 4.0 and (t4 < 4.6 or fti < 4.5):
        label = "Hypothyroidism"
    elif tsh < 0.4 and (t4 > 12.0 or fti > 12.5):
        label = "Hyperthyroidism"
    else:
        label = "Normal"
        
    X6_data.append([tsh, t4, fti, t3, age, gender])
    y6_data.append(label)

X6 = np.array(X6_data)
y6 = np.array(y6_data)

scaler6 = StandardScaler()
X6_scaled = scaler6.fit_transform(X6)

model6 = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
model6.fit(X6_scaled, y6)

with open("ml/models/thyroid_model.pkl", "wb") as f:
    pickle.dump({"model": model6, "scaler": scaler6, "features": ["TSH", "T4", "FTI", "T3", "age", "gender"], "classes": thyroid_classes}, f)


# =========================================================================
# MODEL 7 — Kidney & Liver Model
# =========================================================================
print("Training Model 7: Kidney & Liver Model...")
# Inputs: creatinine, uric_acid, albumin, ALT, AST, bilirubin, blood_urea, sodium, potassium
# Output: "Normal", "Kidney Disease (CKD)", "Liver Disease"
organs_classes = ["Normal", "Kidney Disease (CKD)", "Liver Disease"]
X7_data = []
y7_data = []

for _ in range(600):
    creat = np.random.uniform(0.4, 4.5)
    uric = np.random.uniform(2.0, 10.0)
    alb = np.random.uniform(2.0, 6.0)
    alt = np.random.uniform(5, 150)
    ast = np.random.uniform(5, 150)
    bili = np.random.uniform(0.1, 4.0)
    urea = np.random.uniform(10, 80)
    sod = np.random.uniform(130, 150)
    pot = np.random.uniform(3.0, 6.0)
    
    # Risk logic
    if creat > 1.4 or uric > 8.0:
        label = "Kidney Disease (CKD)"
    elif alt > 60 or ast > 50 or bili > 1.5:
        label = "Liver Disease"
    else:
        label = "Normal"
        
    X7_data.append([creat, uric, alb, alt, ast, bili, urea, sod, pot])
    y7_data.append(label)

X7 = np.array(X7_data)
y7 = np.array(y7_data)

scaler7 = StandardScaler()
X7_scaled = scaler7.fit_transform(X7)

model7 = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
model7.fit(X7_scaled, y7)

with open("ml/models/kidney_liver_model.pkl", "wb") as f:
    pickle.dump({"model": model7, "scaler": scaler7, "features": ["creatinine", "uric_acid", "albumin", "ALT", "AST", "bilirubin_total", "blood_urea", "sodium", "potassium"], "classes": organs_classes}, f)


# =========================================================================
# MODEL 8 — Severity & Triage Classifier
# =========================================================================
print("Training Model 8: Severity & Triage Classifier...")
# Inputs: all extracted values + outputs of Models 1-7 (which are mapped as categorical indices or confidence scores)
# To simplify, we'll feed a feature vector:
# [hemoglobin, blood_sugar_fasting, HbA1c, creatinine, TSH, ALT, AST, BP_systolic, BP_diastolic, age, chronic_disease_count]
# Output: "mild", "moderate", "serious"
severity_classes = ["mild", "moderate", "serious"]
X8_data = []
y8_data = []

for _ in range(800):
    hemo = np.random.uniform(8.0, 16.0)
    sugar = np.random.uniform(70, 240)
    hba1c = np.random.uniform(4.5, 9.0)
    creat = np.random.uniform(0.5, 3.5)
    tsh = np.random.uniform(0.2, 12.0)
    alt = np.random.uniform(10, 120)
    ast = np.random.uniform(10, 120)
    bps = np.random.uniform(90, 180)
    bpd = np.random.uniform(60, 110)
    age = np.random.uniform(18, 80)
    chronic = np.random.choice([0, 1, 2, 3])
    
    # Severity logic
    score = 0
    if hemo < 10.0 or hemo > 18.0: score += 2
    if sugar > 180: score += 2
    if hba1c > 8.0: score += 2
    if creat > 1.8: score += 3
    if alt > 80 or ast > 80: score += 2
    if bps > 160 or bpd > 100: score += 3
    if chronic >= 2: score += 1
    
    if score >= 5:
        label = "serious"
    elif score >= 2:
        label = "moderate"
    else:
        label = "mild"
        
    X8_data.append([hemo, sugar, hba1c, creat, tsh, alt, ast, bps, bpd, age, chronic])
    y8_data.append(label)

X8 = np.array(X8_data)
y8 = np.array(y8_data)

scaler8 = StandardScaler()
X8_scaled = scaler8.fit_transform(X8)

model8 = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
model8.fit(X8_scaled, y8)

with open("ml/models/severity_model.pkl", "wb") as f:
    pickle.dump({"model": model8, "scaler": scaler8, "features": ["hemoglobin", "blood_sugar_fasting", "HbA1c", "creatinine", "TSH", "ALT", "AST", "BP_systolic", "BP_diastolic", "age", "chronic"], "classes": severity_classes}, f)

print("All 8 specialized models trained and saved to ml/models/ successfully!")
