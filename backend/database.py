import os
from datetime import datetime
from typing import List
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.types import TypeDecorator

# Load environment variables
load_dotenv()

# Base model class
Base = declarative_base()

# Custom JSON type helper to support both SQLite (Text/JSON) and Postgres (JSONB)
import json
class SQLiteJSON(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value)

# Custom Array type helper to support SQLite (stores comma-separated or JSON) and Postgres (ARRAY)
class SQLiteArray(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        return json.loads(value)

# Helper function to get appropriate column type depending on dialect (Postgres vs SQLite)
def get_json_type(is_sqlite=False):
    return SQLiteJSON() if is_sqlite else JSONB

def get_array_type(is_sqlite=False):
    return SQLiteArray() if is_sqlite else ARRAY(String)

# Database models definition
class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True)  # uuid string
    email = Column(String, unique=True, nullable=False)
    name = Column(String)
    subscription_tier = Column(String, default='free')
    stripe_customer_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Doctor(Base):
    __tablename__ = 'doctors'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    specialty = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    available = Column(Boolean, default=True)
    rating = Column(Float, default=5.0)
    total_consultations = Column(Integer, default=0)

class HealthProfile(Base):
    __tablename__ = 'health_profiles'
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    blood_group = Column(String, nullable=False)
    # Define fields below dynamically in subclass or keep schema clean
    chronic_conditions = Column(Text)  # JSON-serialized array for SQLite, ARRAY(String) for PG
    medications = Column(Text)
    allergies = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Report(Base):
    __tablename__ = 'reports'
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    file_url = Column(String)
    raw_text = Column(Text)
    extracted_values = Column(Text)  # JSONB for Postgres, JSON-serialized string for SQLite
    ml_diagnosis = Column(Text)      # JSONB for Postgres, JSON-serialized string for SQLite
    severity = Column(String, nullable=False)
    doctor_reviewed = Column(Boolean, default=False)
    doctor_notes = Column(Text)

class Appointment(Base):
    __tablename__ = 'appointments'
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    doctor_id = Column(String, ForeignKey('doctors.id', ondelete='CASCADE'), nullable=False)
    report_id = Column(String, ForeignKey('reports.id', ondelete='SET NULL'))
    type = Column(String, nullable=False)  # 'chat', 'video'
    status = Column(String, default='pending')  # 'pending', 'accepted', 'completed', 'cancelled'
    video_room_url = Column(String)
    scheduled_at = Column(DateTime, nullable=False)
    is_free = Column(Boolean, default=False)

class Subscription(Base):
    __tablename__ = 'subscriptions'
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    stripe_subscription_id = Column(String, unique=True)
    plan = Column(String, nullable=False)
    status = Column(String, nullable=False)
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)

# Database engines setup
DATABASE_URL = os.getenv("DATABASE_URL")
SQLITE_URL = "sqlite:///medical_history.db"

# Remote Postgres engine
engine_pg = None
SessionLocalPostgres = None
if DATABASE_URL:
    try:
        # Handle Supabase pooler / direct connection parameters
        engine_pg = create_engine(
            DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        SessionLocalPostgres = sessionmaker(autocommit=False, autoflush=False, bind=engine_pg)
        print("Connected to remote Supabase database successfully.")
    except Exception as e:
        print(f"Error initializing Supabase connection: {e}")

# Local SQLite engine
engine_sqlite = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
SessionLocalSQLite = sessionmaker(autocommit=False, autoflush=False, bind=engine_sqlite)

# Create local SQLite tables
Base.metadata.create_all(bind=engine_sqlite)
print("Initialized local SQLite database.")

# Dependency injection for endpoints
def get_db_pg():
    if SessionLocalPostgres is None:
        raise Exception("Supabase Database connection is not configured or failed.")
    db = SessionLocalPostgres()
    try:
        yield db
    finally:
        db.close()

def get_db_lite():
    db = SessionLocalSQLite()
    try:
        yield db
    finally:
        db.close()
