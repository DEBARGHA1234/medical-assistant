-- Supabase PostgreSQL Schema for MediAI

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  subscription_tier TEXT DEFAULT 'free',
  stripe_customer_id TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Doctors Table
CREATE TABLE IF NOT EXISTS doctors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  specialty TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  available BOOLEAN DEFAULT TRUE,
  rating FLOAT DEFAULT 5.0,
  total_consultations INTEGER DEFAULT 0
);

-- 3. Health Profiles Table
CREATE TABLE IF NOT EXISTS health_profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  age INTEGER NOT NULL,
  gender TEXT NOT NULL,
  blood_group TEXT NOT NULL,
  chronic_conditions TEXT[] DEFAULT '{}',
  medications TEXT,
  allergies TEXT,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Reports Table
CREATE TABLE IF NOT EXISTS reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  file_url TEXT,
  raw_text TEXT,
  extracted_values JSONB DEFAULT '{}'::jsonb,
  ml_diagnosis JSONB DEFAULT '{}'::jsonb,
  severity TEXT NOT NULL, -- 'mild', 'moderate', 'serious'
  doctor_reviewed BOOLEAN DEFAULT FALSE,
  doctor_notes TEXT
);

-- 5. Appointments Table
CREATE TABLE IF NOT EXISTS appointments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  doctor_id UUID NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
  report_id UUID REFERENCES reports(id) ON DELETE SET NULL,
  type TEXT NOT NULL, -- 'chat', 'video'
  status TEXT DEFAULT 'pending', -- 'pending', 'accepted', 'completed', 'cancelled'
  video_room_url TEXT,
  scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
  is_free BOOLEAN DEFAULT FALSE
);

-- 6. Subscriptions Table
CREATE TABLE IF NOT EXISTS subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  stripe_subscription_id TEXT UNIQUE,
  plan TEXT NOT NULL,
  status TEXT NOT NULL,
  current_period_start TIMESTAMP WITH TIME ZONE,
  current_period_end TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_reports_user_id ON reports(user_id);
CREATE INDEX IF NOT EXISTS idx_health_profiles_user_id ON health_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_appointments_user_id ON appointments(user_id);
CREATE INDEX IF NOT EXISTS idx_appointments_doctor_id ON appointments(doctor_id);

-- Trigger to update health_profiles.updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_health_profiles_updated_at
BEFORE UPDATE ON health_profiles
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


-- =========================================================================
-- SEED DATA
-- =========================================================================

-- Seed Doctors
INSERT INTO doctors (id, name, specialty, email, available, rating, total_consultations) VALUES
  ('a1111111-1111-1111-1111-111111111111', 'Dr. Anjali Sharma', 'General Physician', 'anjali.sharma@mediai.in', TRUE, 4.8, 124),
  ('a2222222-2222-2222-2222-222222222222', 'Dr. Rohan Mehta', 'Cardiologist', 'rohan.mehta@mediai.in', TRUE, 4.9, 98),
  ('a3333333-3333-3333-3333-333333333333', 'Dr. Priya Nair', 'Endocrinologist', 'priya.nair@mediai.in', FALSE, 4.7, 150),
  ('a4444444-4444-4444-4444-444444444444', 'Dr. Vikram Bose', 'Pulmonologist', 'vikram.bose@mediai.in', TRUE, 4.6, 67),
  ('a5555555-5555-5555-5555-555555555555', 'Dr. Sneha Iyer', 'Nephrologist', 'sneha.iyer@mediai.in', TRUE, 4.9, 89)
ON CONFLICT (email) DO UPDATE SET
  name = EXCLUDED.name,
  specialty = EXCLUDED.specialty,
  available = EXCLUDED.available,
  rating = EXCLUDED.rating,
  total_consultations = EXCLUDED.total_consultations;

-- Seed Demo User (Note: In Supabase, credentials auth is handled in auth.users.
-- We seed public.users for the app state)
INSERT INTO users (id, email, name, subscription_tier, stripe_customer_id, created_at) VALUES
  ('d0000000-0000-0000-0000-000000000000', 'demo@mediai.in', 'Demo Account', 'basic', 'cus_mock_demo123', NOW() - INTERVAL '30 days')
ON CONFLICT (email) DO UPDATE SET
  name = EXCLUDED.name,
  subscription_tier = EXCLUDED.subscription_tier;

-- Seed Demo User Health Profile
INSERT INTO health_profiles (id, user_id, age, gender, blood_group, chronic_conditions, medications, allergies) VALUES
  ('d1111111-1111-1111-1111-111111111111', 'd0000000-0000-0000-0000-000000000000', 42, 'male', 'O-positive', '{"diabetes", "hypertension"}', 'Metformin 500mg, Lisinopril 10mg', 'Penicillin')
ON CONFLICT (id) DO NOTHING;

-- Seed 3 Historical Reports (1 month ago, 2 weeks ago, 3 days ago) to enable trend charting out of the box
INSERT INTO reports (id, user_id, uploaded_at, file_url, raw_text, extracted_values, ml_diagnosis, severity) VALUES
  (
    'r0000000-0000-0000-0000-000000000001',
    'd0000000-0000-0000-0000-000000000000',
    NOW() - INTERVAL '30 days',
    'https://supabase.com/mock/report1.pdf',
    'Blood Test Report: Hemoglobin 12.5 g/dL, Blood Sugar Fasting 135 mg/dL, HbA1c 6.8%, TSH 3.2 mIU/L, Creatinine 0.9 mg/dL',
    '{"hemoglobin": 12.5, "blood_sugar_fasting": 135, "HbA1c": 6.8, "TSH": 3.2, "creatinine": 0.9}'::jsonb,
    '{"top_conditions": [{"name": "Diabetes Mellitus", "confidence": 0.85, "description": "Elevated blood sugar and HbA1c indicative of diabetes."}], "recommended_action": "Consult endocrinologist"}'::jsonb,
    'moderate'
  ),
  (
    'r0000000-0000-0000-0000-000000000002',
    'd0000000-0000-0000-0000-000000000000',
    NOW() - INTERVAL '14 days',
    'https://supabase.com/mock/report2.pdf',
    'Follow up Blood Test: Hemoglobin 12.1 g/dL, Blood Sugar Fasting 128 mg/dL, HbA1c 6.6%, TSH 3.0 mIU/L, Creatinine 1.0 mg/dL',
    '{"hemoglobin": 12.1, "blood_sugar_fasting": 128, "HbA1c": 6.6, "TSH": 3.0, "creatinine": 1.0}'::jsonb,
    '{"top_conditions": [{"name": "Diabetes Mellitus", "confidence": 0.82, "description": "Slight improvement but fasting sugar is still high."}], "recommended_action": "Continue Metformin, limit sugar"}'::jsonb,
    'moderate'
  ),
  (
    'r0000000-0000-0000-0000-000000000003',
    'd0000000-0000-0000-0000-000000000000',
    NOW() - INTERVAL '3 days',
    'https://supabase.com/mock/report3.pdf',
    'Latest Lab Report: Hemoglobin 11.2 g/dL, Blood Sugar Fasting 142 mg/dL, HbA1c 7.1%, TSH 3.1 mIU/L, Creatinine 1.1 mg/dL',
    '{"hemoglobin": 11.2, "blood_sugar_fasting": 142, "HbA1c": 7.1, "TSH": 3.1, "creatinine": 1.1}'::jsonb,
    '{"top_conditions": [{"name": "Diabetes Mellitus", "confidence": 0.88, "description": "Blood sugar levels are rising again. Anemia flags are active due to dropping hemoglobin."}], "recommended_action": "Urgent medical checkup required"}'::jsonb,
    'serious'
  )
ON CONFLICT (id) DO NOTHING;
