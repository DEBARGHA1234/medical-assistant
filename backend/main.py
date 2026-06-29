import os
import json
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import stripe
import httpx
from sqlalchemy.orm import Session

from database import (
    get_db_pg, get_db_lite, User, HealthProfile, Report, Doctor, Appointment, Subscription
)
from ml.pipeline import run_ml_pipeline, parse_report_file
from ml.normal_ranges import NORMAL_RANGES

# Load environment variables
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "sk_test_mock")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_mock")
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")

stripe.api_key = STRIPE_SECRET_KEY

app = FastAPI(title="MediAI Backend API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory rate limiting: user_id -> List of upload timestamps
upload_limits = {}

def check_rate_limit(user_id: str):
    """
    Constraint: Rate limit the upload endpoint: max 10 uploads/hour per user.
    """
    now = datetime.utcnow()
    one_hour_ago = now - timedelta(hours=1)
    
    if user_id not in upload_limits:
        upload_limits[user_id] = []
        
    # Clean up old timestamps
    upload_limits[user_id] = [t for t in upload_limits[user_id] if t > one_hour_ago]
    
    if len(upload_limits[user_id]) >= 10:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 10 report uploads per hour."
        )
        
    upload_limits[user_id].append(now)

# Helper to sync write operations to both Supabase and SQLite
def sync_save(db_pg: Optional[Session], db_lite: Session, model_instance):
    # 1. Save to local SQLite
    try:
        # Check if record already exists in SQLite to avoid duplication on seed/sync
        existing = db_lite.query(model_instance.__class__).filter_by(id=model_instance.id).first()
        if existing:
            # Update attributes
            for key, val in model_instance.__dict__.items():
                if not key.startswith('_'):
                    setattr(existing, key, val)
        else:
            db_lite.add(model_instance)
        db_lite.commit()
    except Exception as e:
        db_lite.rollback()
        print(f"SQLite Write Error: {e}")
        
    # 2. Save to remote Supabase (Postgres) if available
    if db_pg is not None:
        try:
            # Recreate object for PG session since session bindings differ
            existing_pg = db_pg.query(model_instance.__class__).filter_by(id=model_instance.id).first()
            if existing_pg:
                for key, val in model_instance.__dict__.items():
                    if not key.startswith('_') and key != 'id':
                        setattr(existing_pg, key, val)
            else:
                db_pg.add(model_instance)
            db_pg.commit()
        except Exception as e:
            db_pg.rollback()
            print(f"Supabase PostgreSQL Write Error: {e}")

# Database session resolver that falls back gracefully to SQLite if PostgreSQL fails
def get_db(request: Request):
    db_lite = next(get_db_lite())
    db_pg = None
    try:
        db_pg = next(get_db_pg())
    except Exception as e:
        print(f"Warning: Remote Supabase database is not accessible: {e}. Falling back to SQLite.")
    return db_pg, db_lite

# =========================================================================
# API ROUTES
# =========================================================================

# Onboarding request schema
class OnboardingRequest(BaseModel):
    user_id: str
    email: str
    name: str
    age: int
    gender: str
    blood_group: str
    chronic_conditions: List[str]
    medications: str
    allergies: str

@app.post("/api/auth/onboarding")
def user_onboarding(data: OnboardingRequest, dbs: tuple = Depends(get_db)):
    db_pg, db_lite = dbs
    
    # 1. Create or verify User
    user = User(
        id=data.user_id,
        email=data.email,
        name=data.name,
        subscription_tier="free"
    )
    sync_save(db_pg, db_lite, user)
    
    # 2. Create Health Profile
    profile = HealthProfile(
        id=data.user_id, # Match user_id for simplicity
        user_id=data.user_id,
        age=data.age,
        gender=data.gender,
        blood_group=data.blood_group,
        chronic_conditions=json.dumps(data.chronic_conditions), # serialize for portability
        medications=data.medications,
        allergies=data.allergies
    )
    sync_save(db_pg, db_lite, profile)
    
    return {"status": "success", "message": "Onboarding complete"}

@app.post("/api/reports/upload")
async def upload_report(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    dbs: tuple = Depends(get_db)
):
    db_pg, db_lite = dbs
    
    # Rate limit check
    check_rate_limit(user_id)
    
    # Check user subscription limit
    # FREE: 1 upload/month, BASIC: 5, PRO/PREMIUM: Unlimited
    user = db_lite.query(User).filter_by(id=user_id).first()
    if not user:
        # Implicitly register user if not onboarding
        user = User(id=user_id, email="unregistered@mediai.in", name="User", subscription_tier="free")
        sync_save(db_pg, db_lite, user)
        
    tier = user.subscription_tier or "free"
    
    # Simple count check for uploads this month
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    upload_count = db_lite.query(Report).filter(
        Report.user_id == user_id,
        Report.uploaded_at > thirty_days_ago
    ).count()
    
    if tier == "free" and upload_count >= 1:
        raise HTTPException(status_code=403, detail="Free tier limit reached (1 upload per month). Please upgrade your subscription.")
    elif tier == "basic" and upload_count >= 5:
        raise HTTPException(status_code=403, detail="Basic tier limit reached (5 uploads per month). Please upgrade your subscription.")

    # Save file locally
    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/{user_id}_{datetime.utcnow().timestamp()}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    # Step 1: Parse report
    raw_text = parse_report_file(file_path)
    
    # Fetch user health profile for ML context
    profile = db_lite.query(HealthProfile).filter_by(user_id=user_id).first()
    profile_dict = {}
    if profile:
        profile_dict = {
            "age": profile.age,
            "gender": profile.gender,
            "chronic_conditions": json.loads(profile.chronic_conditions) if profile.chronic_conditions else []
        }
    else:
        profile_dict = {"age": 35, "gender": "male", "chronic_conditions": []}

    # Fetch last 3 reports for trend analysis comparison
    prev_reports_query = db_lite.query(Report).filter_by(user_id=user_id).order_by(Report.uploaded_at.desc()).limit(3).all()
    prev_reports = []
    for r in prev_reports_query:
        prev_reports.append({
            "uploaded_at": r.uploaded_at,
            "extracted_values": r.extracted_values
        })
        
    # Step 2-6: Run pipeline
    diagnosis = run_ml_pipeline(raw_text, profile_dict, prev_reports)
    
    # Store in DB
    report_id = f"rep_{int(datetime.utcnow().timestamp())}"
    report_model = Report(
        id=report_id,
        user_id=user_id,
        file_url=f"/api/reports/file/{os.path.basename(file_path)}",
        raw_text=raw_text[:2000], # Keep database clean
        extracted_values=json.dumps(diagnosis["abnormal_values"]),
        ml_diagnosis=json.dumps(diagnosis),
        severity=diagnosis["severity"],
        doctor_reviewed=False
    )
    sync_save(db_pg, db_lite, report_model)
    
    return diagnosis

@app.get("/api/reports/{user_id}")
def get_user_reports(user_id: str, limit: int = 10, offset: int = 0, dbs: tuple = Depends(get_db)):
    _, db_lite = dbs
    reports = db_lite.query(Report).filter_by(user_id=user_id).order_by(Report.uploaded_at.desc()).limit(limit).offset(offset).all()
    
    results = []
    for r in reports:
        # Safely parse JSON strings
        try:
            diag = json.loads(r.ml_diagnosis) if r.ml_diagnosis else {}
        except Exception:
            diag = {}
            
        results.append({
            "id": r.id,
            "uploaded_at": r.uploaded_at,
            "file_url": r.file_url,
            "severity": r.severity,
            "diagnosis": diag,
            "doctor_reviewed": r.doctor_reviewed,
            "doctor_notes": r.doctor_notes
        })
    return results

@app.get("/api/reports/result/{report_id}")
def get_report_details(report_id: str, dbs: tuple = Depends(get_db)):
    _, db_lite = dbs
    report = db_lite.query(Report).filter_by(id=report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    try:
        diag = json.loads(report.ml_diagnosis) if report.ml_diagnosis else {}
    except Exception:
        diag = {}
        
    return {
        "id": report.id,
        "uploaded_at": report.uploaded_at,
        "severity": report.severity,
        "diagnosis": diag,
        "doctor_reviewed": report.doctor_reviewed,
        "doctor_notes": report.doctor_notes
    }

# Appointments schema
class AppointmentBookRequest(BaseModel):
    user_id: str
    doctor_id: str
    report_id: Optional[str] = None
    scheduled_at: str # ISO string

@app.post("/api/appointments/book")
async def book_appointment(data: AppointmentBookRequest, dbs: tuple = Depends(get_db)):
    db_pg, db_lite = dbs
    
    # 1. Fetch user subscription tier
    user = db_lite.query(User).filter_by(id=data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    tier = user.subscription_tier or "free"
    
    # 2. Check if the booking is free
    # Serious severity = Free for all
    # Mild/Moderate = Free for Basic (1/mo), Pro (3/mo), Premium (Instant). Paid ($50) for Free tier.
    is_free = False
    report = None
    if data.report_id:
        report = db_lite.query(Report).filter_by(id=data.report_id).first()
        if report and report.severity == "serious":
            is_free = True
            
    # Check count of calls this month for basic/pro tiers
    if not is_free:
        if tier == "basic":
            calls_this_month = db_lite.query(Appointment).filter(
                Appointment.user_id == data.user_id,
                Appointment.scheduled_at > datetime.utcnow() - timedelta(days=30),
                Appointment.is_free == True
            ).count()
            if calls_this_month < 1:
                is_free = True
        elif tier in ["pro", "premium"]:
            limit = 3 if tier == "pro" else 999
            calls_this_month = db_lite.query(Appointment).filter(
                Appointment.user_id == data.user_id,
                Appointment.scheduled_at > datetime.utcnow() - timedelta(days=30),
                Appointment.is_free == True
            ).count()
            if calls_this_month < limit:
                is_free = True
                
    # 3. Create Daily.co Room
    video_room_url = f"https://mediai.daily.co/mock-call-{data.user_id}-{data.doctor_id}"
    if DAILY_API_KEY:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    "https://api.daily.co/v1/rooms",
                    headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
                    json={
                        "properties": {
                            "exp": int((datetime.utcnow() + timedelta(hours=2)).timestamp()),
                            "enable_screenshare": True
                        }
                    },
                    timeout=10.0
                )
                if res.status_code == 200:
                    video_room_url = res.json().get("url", video_room_url)
        except Exception as e:
            print(f"Daily.co API call failed: {e}. Fallback to mock room.")

    # 4. Save appointment
    appt_id = f"appt_{int(datetime.utcnow().timestamp())}"
    appt = Appointment(
        id=appt_id,
        user_id=data.user_id,
        doctor_id=data.doctor_id,
        report_id=data.report_id,
        type="video",
        status="pending",
        video_room_url=video_room_url,
        scheduled_at=datetime.fromisoformat(data.scheduled_at.replace("Z", "+00:00")),
        is_free=is_free
    )
    sync_save(db_pg, db_lite, appt)
    
    return {
        "status": "success",
        "appointment_id": appt_id,
        "video_room_url": video_room_url,
        "is_free": is_free
    }

@app.get("/api/appointments/{user_id}")
def get_user_appointments(user_id: str, dbs: tuple = Depends(get_db)):
    _, db_lite = dbs
    appts = db_lite.query(Appointment).filter_by(user_id=user_id).order_by(Appointment.scheduled_at.asc()).all()
    results = []
    for a in appts:
        doc = db_lite.query(Doctor).filter_by(id=a.doctor_id).first()
        results.append({
            "id": a.id,
            "doctor_name": doc.name if doc else "Specialist",
            "specialty": doc.specialty if doc else "Medical Advisor",
            "scheduled_at": a.scheduled_at,
            "status": a.status,
            "video_room_url": a.video_room_url,
            "is_free": a.is_free
        })
    return results

@app.get("/api/doctors")
def get_doctors(dbs: tuple = Depends(get_db)):
    _, db_lite = dbs
    return db_lite.query(Doctor).all()

# Doctor portal submissions
class DoctorNotesRequest(BaseModel):
    report_id: str
    doctor_notes: str

@app.post("/api/doctor/notes")
def add_doctor_notes(data: DoctorNotesRequest, dbs: tuple = Depends(get_db)):
    db_pg, db_lite = dbs
    report = db_lite.query(Report).filter_by(id=data.report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    report.doctor_notes = data.doctor_notes
    report.doctor_reviewed = True
    sync_save(db_pg, db_lite, report)
    
    return {"status": "success", "message": "Doctor notes updated"}

# Dashboard statistics
@app.get("/api/dashboard/{user_id}")
def get_dashboard_data(user_id: str, dbs: tuple = Depends(get_db)):
    _, db_lite = dbs
    
    # 1. Fetch user health profile
    profile = db_lite.query(HealthProfile).filter_by(user_id=user_id).first()
    
    # 2. Fetch reports for trends (last 5)
    reports = db_lite.query(Report).filter_by(user_id=user_id).order_by(Report.uploaded_at.desc()).limit(5).all()
    
    # Trend coordinates extraction
    trends = []
    health_score = 90 # default baseline
    abnormal_count = 0
    sugar_levels = []
    
    # Parse trends in ascending chronological order
    for r in reversed(reports):
        # We need to extract the raw metrics from report raw text if they are not stored cleanly
        # Let's run regex extraction on raw text directly to reconstruct trend lines
        metrics = {}
        # Simple extraction helper
        lines = r.raw_text.lower() if r.raw_text else ""
        
        hb_match = re.search(r"hemoglobin\s*:?\s*(\d+\.?\d*)", lines)
        sug_match = re.search(r"fasting\s*blood\s*sugar\s*:?\s*(\d+)", lines)
        hba_match = re.search(r"hba1c\s*:?\s*(\d+\.?\d*)", lines)
        cre_match = re.search(r"creatinine\s*:?\s*(\d+\.?\d*)", lines)
        tsh_match = re.search(r"tsh\s*:?\s*(\d+\.?\d*)", lines)
        
        trends.append({
            "date": r.uploaded_at.strftime("%b %d"),
            "hemoglobin": float(hb_match.group(1)) if hb_match else None,
            "blood_sugar_fasting": float(sug_match.group(1)) if sug_match else None,
            "HbA1c": float(hba_match.group(1)) if hba_match else None,
            "creatinine": float(cre_match.group(1)) if cre_match else None,
            "TSH": float(tsh_match.group(1)) if tsh_match else None,
        })
        
        if sug_match:
            sugar_levels.append(float(sug_match.group(1)))
            
    # Calculate health score: deduct points for abnormal values in latest report
    if reports:
        latest = reports[0]
        try:
            diag = json.loads(latest.ml_diagnosis) if latest.ml_diagnosis else {}
        except Exception:
            diag = {}
            
        abnormal_values = diag.get("abnormal_values", [])
        abnormal_count = len(abnormal_values)
        
        # Deduct 10 points per abnormal value, min score 45
        health_score = max(100 - (abnormal_count * 10), 45)
        if latest.severity == "serious":
            health_score = min(health_score, 60)
            
    # Summary card insight
    summary_insight = "All recent parameters processed are within biological limits. Continue your healthy routines!"
    if len(sugar_levels) >= 2:
        if any(x > 125 for x in sugar_levels):
            summary_insight = "Based on your last reports, your blood sugar is consistently elevated. Limit refined sugars."
        elif sugar_levels[-1] > sugar_levels[-2]:
            summary_insight = "Based on your last 3 reports, your blood sugar levels are on a rising trend. Watch your diet."
    elif abnormal_count > 0:
        summary_insight = f"You have {abnormal_count} flagged values outside normal reference ranges. Review them with our doctors."

    return {
        "health_score": health_score,
        "trends": trends,
        "abnormal_count": abnormal_count,
        "summary_insight": summary_insight,
        "total_reports": len(reports)
    }

# =========================================================================
# STRIPE SUBSCRIPTION ENDPOINTS
# =========================================================================

class CreateSubscriptionRequest(BaseModel):
    user_id: str
    email: str
    tier: str # 'basic', 'pro', 'premium'

@app.post("/api/stripe/create-subscription")
def create_subscription_checkout(data: CreateSubscriptionRequest, dbs: tuple = Depends(get_db)):
    db_pg, db_lite = dbs
    
    # Map tiers to Stripe price IDs
    # (In sandbox, mock these or replace with actual Stripe Price IDs)
    tier_prices = {
        "basic": "price_basic_123",
        "pro": "price_pro_123",
        "premium": "price_premium_123"
    }
    
    price_id = tier_prices.get(data.tier)
    if not price_id:
        raise HTTPException(status_code=400, detail="Invalid subscription tier")
        
    try:
        # Find or create Stripe Customer
        user = db_lite.query(User).filter_by(id=data.user_id).first()
        customer_id = user.stripe_customer_id if user else None
        
        if not customer_id or customer_id.startswith("cus_mock"):
            customer = stripe.Customer.create(
                email=data.email,
                metadata={"user_id": data.user_id}
            )
            customer_id = customer.id
            if user:
                user.stripe_customer_id = customer_id
                sync_save(db_pg, db_lite, user)
                
        # Create Checkout Session
        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url="http://localhost:5173/settings?payment_success=true",
            cancel_url="http://localhost:5173/settings?payment_cancelled=true",
            metadata={"user_id": data.user_id, "tier": data.tier}
        )
        
        return {"checkout_url": checkout_session.url}
    except Exception as e:
        # Fallback for testing: return a direct mock successful link updating the tier immediately
        print(f"Stripe error: {e}. Generating sandbox checkout redirect.")
        # If Stripe is not configured/auth fails, redirect user to success page directly with a query param
        return {"checkout_url": f"http://localhost:5173/settings?payment_success=true&tier={data.tier}"}

@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request, dbs: tuple = Depends(get_db)):
    db_pg, db_lite = dbs
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        # Direct webhook event manual trigger for sandbox testing when signature is missing
        try:
            event = json.loads(payload)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")
            
    event_type = event.get("type") if isinstance(event, dict) else event.type
    event_data = event.get("data") if isinstance(event, dict) else event.data
    
    if event_type == "checkout.session.completed":
        session = event_data.get("object") if isinstance(event_data, dict) else event_data.object
        user_id = session.metadata.get("user_id")
        tier = session.metadata.get("tier")
        cust_id = session.customer
        
        if user_id:
            user = db_lite.query(User).filter_by(id=user_id).first()
            if user:
                user.subscription_tier = tier
                user.stripe_customer_id = cust_id
                sync_save(db_pg, db_lite, user)
                
    elif event_type in ["customer.subscription.updated", "customer.subscription.deleted"]:
        sub = event_data.get("object") if isinstance(event_data, dict) else event_data.object
        cust_id = sub.customer
        
        # Resolve user
        user = db_lite.query(User).filter_by(stripe_customer_id=cust_id).first()
        if user:
            if event_type == "customer.subscription.deleted":
                user.subscription_tier = "free"
            else:
                # Update based on subscription status
                pass
            sync_save(db_pg, db_lite, user)
            
    return {"status": "success"}

# Serve static report uploads for viewing (mock endpoint)
from fastapi.responses import FileResponse
@app.get("/api/reports/file/{filename}")
def get_report_file(filename: str):
    file_path = f"data/uploads/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")
