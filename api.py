import os
import sys
import io
import time
import math
import json
import secrets
import uuid
import fitz  # PyMuPDF
import psycopg2
import stripe
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks, Request, Depends, status
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
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

# Stripe Keys (Get this from your Stripe Dashboard)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_your_stripe_key_here")

# Initialize Rate Limiter (Tracks by IP address)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Marg ERP Extraction API", version="1.0.0")

# Register the rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

# Ensure the uploads directory exists for persistent storage
os.makedirs("uploads", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════

class CustomerCreate(BaseModel):
    customer_name: str
    plan: str
    credits: int = 0

class CustomerUpdate(BaseModel):
    api_key: str
    plan: str
    credits: int

class KeyRegenerateRequest(BaseModel):
    old_api_key: str

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS & SECURITY
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

    if status_code == 200 and stripe_item_id and stripe_item_id.startswith("si_"):
        try:
            stripe.SubscriptionItem.create_usage_record(
                stripe_item_id, quantity=pages, timestamp=int(time.time()), action='increment'
            )
        except Exception as e:
            print(f"Stripe Billing Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  CORE EXTRACTION ENDPOINT 
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/extract")
@limiter.limit("20/minute") 
async def extract_document(
    request: Request, 
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    start_time = time.time()
    page_count = 0
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join("uploads", unique_filename)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        cursor.execute("SELECT * FROM api_users WHERE api_key = %s", (x_api_key,))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        file_bytes = await file.read()
        
        if len(file_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 15MB.")

        page_count = calculate_billable_pages(file.filename, file_bytes)
        if page_count == 0:
            raise HTTPException(status_code=400, detail="Invalid or unsupported file format.")

        if user["plan"] == "prepaid" and user["credits_remaining"] < page_count:
            raise HTTPException(status_code=402, detail=f"Insufficient credits. Requires {page_count} pages.")

        # Save File Permanently
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Run Extraction
        raw_text = extract_raw_text(file_path)
        extracted_data = route_and_parse(raw_text, source_file=file.filename)
        
        # Deduct Credits
        if user["plan"] == "prepaid":
            cursor.execute("UPDATE api_users SET credits_remaining = credits_remaining - %s WHERE api_key = %s", (page_count, x_api_key))
        else:
            cursor.execute("UPDATE api_users SET pages_processed_this_month = pages_processed_this_month + %s WHERE api_key = %s", (page_count, x_api_key))
        
        conn.commit()
        process_time_ms = int((time.time() - start_time) * 1000)

        # Trigger God-Mode Telemetry & Stripe Billing (Log the unique filename!)
        background_tasks.add_task(
            log_telemetry_and_bill_stripe, 
            x_api_key, "/api/v1/extract", unique_filename, page_count, 200, process_time_ms, user.get("stripe_sub_item_id", "")
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
        background_tasks.add_task(log_telemetry_and_bill_stripe, x_api_key, "/api/v1/extract", unique_filename, page_count, http_exc.status_code, process_time_ms, "", http_exc.detail)
        raise
    except Exception as e:
        process_time_ms = int((time.time() - start_time) * 1000)
        background_tasks.add_task(log_telemetry_and_bill_stripe, x_api_key, "/api/v1/extract", unique_filename, page_count, 500, process_time_ms, "", str(e))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN ACTIONS (SECURED)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/admin/customers")
async def add_new_customer(customer: CustomerCreate, current_admin: str = Depends(verify_admin)):
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
        return {"status": "success", "api_key": new_api_key}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.put("/admin/customers")
async def edit_customer(customer: CustomerUpdate, current_admin: str = Depends(verify_admin)):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE api_users SET plan = %s, credits_remaining = %s WHERE api_key = %s", (customer.plan, customer.credits, customer.api_key))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Customer not found.")
        conn.commit()
        return {"status": "success"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.post("/admin/customers/regenerate-key")
async def regenerate_api_key(request: KeyRegenerateRequest, current_admin: str = Depends(verify_admin)):
    new_api_key = f"sk_live_{secrets.token_urlsafe(32)}"
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE api_users SET api_key = %s WHERE api_key = %s", (new_api_key, request.old_api_key))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Old API Key not found.")
        cursor.execute("UPDATE api_requests_log SET api_key = %s WHERE api_key = %s", (new_api_key, request.old_api_key))
        conn.commit()
        return {"status": "success", "new_api_key": new_api_key}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/admin/files/{filename}")
async def get_uploaded_file(filename: str, current_admin: str = Depends(verify_admin)):
    """Serves the uploaded file securely to the Admin."""
    file_path = os.path.join("uploads", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server.")
    return FileResponse(file_path)

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD ROUTE
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

        cursor.execute("""
            SELECT u.customer_name, COUNT(l.log_id) as total_docs, SUM(l.pages_billed) as total_pages 
            FROM api_requests_log l JOIN api_users u ON l.api_key = u.api_key 
            WHERE l.status_code = 200 GROUP BY u.customer_name ORDER BY total_pages DESC
        """)
        company_stats = cursor.fetchall()

        cursor.execute("""
            SELECT u.customer_name, l.file_name, l.pages_billed, l.created_at, l.status_code 
            FROM api_requests_log l JOIN api_users u ON l.api_key = u.api_key 
            ORDER BY l.created_at DESC LIMIT 20
        """)
        file_logs = cursor.fetchall()
        
        return templates.TemplateResponse(request=request, name="admin.html", context={
            "admin_user": current_admin, "users": users, "kpis": kpis, "errors": recent_errors,
            "chart_labels": json.dumps([str(row['date']) for row in chart_data]),
            "chart_values": json.dumps([row['daily_pages'] for row in chart_data]),
            "company_stats": company_stats, "file_logs": file_logs
        })
    finally:
        cursor.close()
        conn.close()

@app.get("/logout", response_class=HTMLResponse)
async def logout_admin():
    return HTMLResponse(status_code=401, headers={"WWW-Authenticate": "Basic"}, content="""
        <html><head><title>Logged Out</title><style>body { font-family: Arial, sans-serif; text-align: center; padding-top: 100px; background-color: #f4f7f6; } .box { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; } a { color: #3498db; text-decoration: none; font-weight: bold; }</style></head><body><div class="box"><h2>🔒 Securely logged out.</h2><br><a href="/dashboard">Log back in</a></div></body></html>
    """)