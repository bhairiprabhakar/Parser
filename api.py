import os
import sys
import io
import time
import math
import json
import secrets
import fitz  # PyMuPDF
import psycopg2
import stripe  # 🚀 NEW: Stripe SDK
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks, Request, Depends, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

try:
    import pandas as pd
except ImportError:
    pd = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsers.universal_router import route_and_parse
from extractors.text_extractor import extract_raw_text

# ── CONFIGURATION ──
DB_CONFIG = {
    "dbname": "VSPB",
    "user": "postgres",
    "password": "101990",
    "host": "localhost",
    "port": "5432"
}

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "supersecretpassword"

# Model for the incoming Add Customer request
class CustomerCreate(BaseModel):
    customer_name: str
    plan: str
    credits: int = 0

# 🚀 NEW: Stripe Keys (Get this from your Stripe Dashboard)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_your_stripe_key_here")

# 🚀 NEW: Initialize Rate Limiter (Tracks by IP address)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Marg ERP Extraction API", version="1.0.0")

# Register the rate limiter with FastAPI
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS & BACKGROUND TASKS
# ══════════════════════════════════════════════════════════════════════════════

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")

def calculate_billable_pages(filename: str, file_bytes: bytes) -> int:
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return max(1, len(doc))
        except Exception:
            return 0
    elif ext in ['jpg', 'jpeg', 'png', 'bmp']:
        return 1
    elif ext == 'csv':
        try:
            lines = file_bytes.decode('utf-8', errors='ignore').strip().count('\n') + 1
            return max(1, math.ceil(lines / 50))
        except Exception:
            return 1
    elif ext in ['xlsx', 'xls']:
        try:
            if pd is not None:
                df_dict = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                return max(1, math.ceil(sum(len(df) for df in df_dict.values()) / 50))
            return 1
        except Exception:
            return 1
    return 0

def log_telemetry_and_bill_stripe(api_key: str, endpoint: str, file_name: str, pages: int, status_code: int, process_time: int, stripe_item_id: str, error: str = ""):
    """Logs to Postgres AND pings Stripe to charge the customer."""
    # 1. Log to PostgreSQL Database
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO api_requests_log (api_key, endpoint_hit, file_name, pages_billed, status_code, processing_time_ms, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (api_key, endpoint, file_name, pages, status_code, process_time, error))
        conn.commit()
    except Exception as e:
        print(f"DB Log Failed: {e}")
    finally:
        cursor.close()
        conn.close()

    # 2. Charge via Stripe (Only if successful and using pay-as-you-go)
    if status_code == 200 and stripe_item_id and stripe_item_id.startswith("si_"):
        try:
            stripe.SubscriptionItem.create_usage_record(
                stripe_item_id,
                quantity=pages,
                timestamp=int(time.time()),
                action='increment',
            )
        except Exception as e:
            print(f"Stripe Billing Failed for {stripe_item_id}: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  CORE EXTRACTION ENDPOINT 
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/extract")
@limiter.limit("20/minute") # 🚀 NEW: Security Rate Limit (Max 20 requests per minute)
async def extract_document(
    request: Request, # Required for limiter
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    start_time = time.time()
    page_count = 0
    temp_filepath = None
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        cursor.execute("SELECT * FROM api_users WHERE api_key = %s", (x_api_key,))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        file_bytes = await file.read()
        
        # 🚀 NEW: Security Check - Block files larger than 15MB
        if len(file_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 15MB.")

        page_count = calculate_billable_pages(file.filename, file_bytes)
        if page_count == 0:
            raise HTTPException(status_code=400, detail="Invalid or unsupported file format.")

        if user["plan"] == "prepaid" and user["credits_remaining"] < page_count:
            raise HTTPException(status_code=402, detail=f"Insufficient credits. Requires {page_count} pages.")

        temp_filepath = f"temp_{file.filename}"
        with open(temp_filepath, "wb") as f:
            f.write(file_bytes)

        raw_text = extract_raw_text(temp_filepath)
        extracted_data = route_and_parse(raw_text, source_file=file.filename)
        
        if user["plan"] == "prepaid":
            cursor.execute("UPDATE api_users SET credits_remaining = credits_remaining - %s WHERE api_key = %s", (page_count, x_api_key))
        else:
            cursor.execute("UPDATE api_users SET pages_processed_this_month = pages_processed_this_month + %s WHERE api_key = %s", (page_count, x_api_key))
        
        conn.commit()
        process_time_ms = int((time.time() - start_time) * 1000)

        # 🚀 Trigger God-Mode Telemetry & Stripe Billing
        background_tasks.add_task(
            log_telemetry_and_bill_stripe, 
            x_api_key, "/api/v1/extract", file.filename, page_count, 200, process_time_ms, user.get("stripe_sub_item_id", "")
        )

        extracted_data["Billing"] = {
            "Customer": user["customer_name"],
            "PagesBilled": page_count,
            "Plan": user["plan"],
            "ProcessingTimeMs": process_time_ms
        }
        return JSONResponse(content=extracted_data)

    except HTTPException as http_exc:
        process_time_ms = int((time.time() - start_time) * 1000)
        background_tasks.add_task(log_telemetry_and_bill_stripe, x_api_key, "/api/v1/extract", file.filename, page_count, http_exc.status_code, process_time_ms, "", http_exc.detail)
        raise
    except Exception as e:
        process_time_ms = int((time.time() - start_time) * 1000)
        background_tasks.add_task(log_telemetry_and_bill_stripe, x_api_key, "/api/v1/extract", file.filename, page_count, 500, process_time_ms, "", str(e))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        cursor.close()
        conn.close()
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN ACTIONS (SECURED)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/admin/customers")
async def add_new_customer(customer: CustomerCreate, current_admin: str = Depends(verify_admin)):
    """Generates a secure API key and adds a new customer to the database."""
    
    # Generate a cryptographically secure 32-byte API Key
    raw_token = secrets.token_urlsafe(32)
    new_api_key = f"sk_live_{raw_token}"
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO api_users (api_key, customer_name, plan, credits_remaining, pages_processed_this_month)
            VALUES (%s, %s, %s, %s, 0)
        """, (new_api_key, customer.customer_name, customer.plan, customer.credits))
        conn.commit()
        
        return {"status": "success", "message": "Customer created successfully!", "api_key": new_api_key}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()
        
# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN DASHBOARD & LOGOUT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard", response_class=HTMLResponse)
async def view_admin_dashboard(request: Request, current_admin: str = Depends(verify_admin)):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT * FROM api_users ORDER BY pages_processed_this_month DESC")
        users = cursor.fetchall()
        
        cursor.execute("SELECT SUM(pages_billed) as total_pages FROM api_requests_log")
        total_pages = cursor.fetchone()['total_pages'] or 0

        cursor.execute("SELECT AVG(processing_time_ms) as avg_latency FROM api_requests_log WHERE status_code = 200")
        avg_lat = cursor.fetchone()['avg_latency']
        avg_latency = int(avg_lat) if avg_lat else 0

        cursor.execute("SELECT COUNT(*) as total_reqs, SUM(CASE WHEN status_code > 399 THEN 1 ELSE 0 END) as total_errors FROM api_requests_log")
        error_stats = cursor.fetchone()
        error_rate = round((error_stats['total_errors'] / error_stats['total_reqs']) * 100, 2) if error_stats and error_stats['total_reqs'] > 0 else 0

        kpis = {"total_pages": total_pages, "avg_latency": avg_latency, "error_rate": error_rate}

        cursor.execute("SELECT created_at, status_code, file_name, error_message FROM api_requests_log WHERE status_code > 399 ORDER BY created_at DESC LIMIT 5")
        recent_errors = cursor.fetchall()

        cursor.execute("SELECT DATE(created_at) as date, SUM(pages_billed) as daily_pages FROM api_requests_log WHERE created_at >= CURRENT_DATE - INTERVAL '7 days' GROUP BY DATE(created_at) ORDER BY date ASC")
        chart_data = cursor.fetchall()
        
        return templates.TemplateResponse(request=request, name="admin.html", context={
            "admin_user": current_admin, "users": users, "kpis": kpis, "errors": recent_errors,
            "chart_labels": json.dumps([str(row['date']) for row in chart_data]),
            "chart_values": json.dumps([row['daily_pages'] for row in chart_data])
        })
    finally:
        cursor.close()
        conn.close()

@app.get("/logout", response_class=HTMLResponse)
async def logout_admin():
    return HTMLResponse(status_code=401, headers={"WWW-Authenticate": "Basic"}, content="""
        <html><head><title>Logged Out</title><style>body { font-family: Arial, sans-serif; text-align: center; padding-top: 100px; background-color: #f4f7f6; } .box { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; } a { color: #3498db; text-decoration: none; font-weight: bold; }</style></head><body><div class="box"><h2>🔒 Securely logged out.</h2><br><a href="/dashboard">Log back in</a></div></body></html>
    """)