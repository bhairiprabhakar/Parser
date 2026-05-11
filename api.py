import os
import sys
import io
import time
import math
import json
import secrets
import uuid
import shutil
import psycopg2
from datetime import datetime
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks, Request, Depends, status, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

try:
    import pandas as pd
except ImportError:
    pd = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractors.text_extractor import extract_raw_text
from parsers.universal_router import route_and_parse

# ── CONFIGURATION & DIRECTORIES ──
DB_CONFIG = {
    "dbname": "VSPB",
    "user": "postgres",
    "password": "101990",
    "host": "localhost",
    "port": "5432"
}

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "supersecretpassword"

# 🚀 NEW: Quarantine Vault Directory
QUARANTINE_DIR = "quarantine_vault"
os.makedirs(QUARANTINE_DIR, exist_ok=True)

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Extraction API", version="1.0.0")

# 🚀 CORS Middleware for Public SaaS APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

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
        import fitz
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return max(1, len(doc))
        except Exception:
            return 0
    elif ext in ['jpg', 'jpeg', 'png', 'bmp']:
        return 3 
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

def log_api_request_to_db(api_key: str, endpoint: str, file_name: str, file_url: str, pages: int, status_code: int, process_time: int, file_size_bytes: int = 0, error: str = "", quarantine_path: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO api_requests_log (api_key, endpoint_hit, file_name, file_url, pages_billed, status_code, processing_time_ms, file_size_bytes, error_message, quarantine_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (api_key, endpoint, file_name, file_url, pages, status_code, process_time, file_size_bytes, error, quarantine_path))
        conn.commit()
    except Exception as e:
        print(f"DB Log Failed: {e}")
    finally:
        cursor.close()
        conn.close()

# 🚀 NEW: Auto-Delete 72-Hour Janitor
def clean_quarantine_vault():
    """Deletes any quarantined files older than 72 hours to ensure privacy compliance."""
    if not os.path.exists(QUARANTINE_DIR):
        return
        
    now = time.time()
    for filename in os.listdir(QUARANTINE_DIR):
        filepath = os.path.join(QUARANTINE_DIR, filename)
        if os.path.isfile(filepath):
            file_age_hours = (now - os.path.getctime(filepath)) / 3600
            if file_age_hours > 72:
                try:
                    os.remove(filepath)
                except Exception:
                    pass

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
    file_size_bytes = 0
    drive_link = "Local Processing Only" 
    temp_filepath = None
    quarantine_path = None
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Trigger the 72-hour cleanup janitor in the background
    background_tasks.add_task(clean_quarantine_vault)

    try:
        cursor.execute("SELECT * FROM api_users WHERE api_key = %s", (x_api_key,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        file_bytes = await file.read()
        file_size_bytes = len(file_bytes)
        
        if file_size_bytes > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 15MB.")

        page_count = calculate_billable_pages(file.filename, file_bytes)
        if page_count == 0:
            raise HTTPException(status_code=400, detail="Invalid or unsupported file format.")

        if user["plan"] == "prepaid" and user["credits_remaining"] < page_count:
            raise HTTPException(status_code=402, detail=f"Insufficient credits. Requires {page_count} pages.")

        temp_filepath = f"temp_{uuid.uuid4()}_{file.filename}"
        with open(temp_filepath, "wb") as f:
            f.write(file_bytes)

        raw_text = extract_raw_text(temp_filepath)
        extracted_data = route_and_parse(raw_text, source_file=file.filename)
        
        if user["plan"] == "prepaid":
            cursor.execute("""
                UPDATE api_users 
                SET credits_remaining = credits_remaining - %s, 
                    pages_processed_this_month = pages_processed_this_month + %s 
                WHERE api_key = %s
            """, (page_count, page_count, x_api_key))
        else:
            cursor.execute("""
                UPDATE api_users 
                SET pages_processed_this_month = pages_processed_this_month + %s 
                WHERE api_key = %s
            """, (page_count, x_api_key))
        
        conn.commit()
        process_time_ms = int((time.time() - start_time) * 1000)

        # Successful extraction, no quarantine needed
        background_tasks.add_task(
            log_api_request_to_db, x_api_key, "/api/v1/extract", file.filename, drive_link, page_count, 200, process_time_ms, file_size_bytes, "", None
        )

        extracted_data["Billing"] = {
            "Customer": user["customer_name"],
            "PagesBilled": page_count,
            "Plan": user["plan"],
            "FileSizeKB": round(file_size_bytes / 1024, 2),
            "ProcessingTimeMs": process_time_ms
        }
        return JSONResponse(content=extracted_data)

    except HTTPException as http_exc:
        process_time_ms = int((time.time() - start_time) * 1000)
        # We generally do not quarantine for simple 400/401/402 HTTP Errors
        background_tasks.add_task(log_api_request_to_db, x_api_key, "/api/v1/extract", file.filename, drive_link, page_count, http_exc.status_code, process_time_ms, file_size_bytes, http_exc.detail, None)
        raise
        
    except Exception as e:
        process_time_ms = int((time.time() - start_time) * 1000)
        
        # 🚀 QUARANTINE VAULT LOGIC (For 500 Crashes)
        if temp_filepath and os.path.exists(temp_filepath):
            safe_filename = f"crash_{uuid.uuid4().hex[:8]}_{file.filename}"
            quarantine_path = os.path.join(QUARANTINE_DIR, safe_filename)
            shutil.copy(temp_filepath, quarantine_path) 
            
        background_tasks.add_task(log_api_request_to_db, x_api_key, "/api/v1/extract", file.filename, drive_link, page_count, 500, process_time_ms, file_size_bytes, str(e), quarantine_path)
        raise HTTPException(status_code=500, detail=f"Extraction failed. Engineers have been notified.")
        
    finally:
        cursor.close()
        conn.close()
        # 🚀 ALWAYS delete the original temporary file
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC SAAS ONBOARDING
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def public_landing_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.post("/request-access")
async def submit_api_request(company_name: str = Form(...), email: str = Form(...), phone: str = Form("")):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO api_leads (company_name, email, phone) VALUES (%s, %s, %s)", (company_name, email, phone))
        conn.commit()
        return RedirectResponse("/success", status_code=303)
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/success", response_class=HTMLResponse)
async def request_success(request: Request):
    return templates.TemplateResponse(request=request, name="success.html", context={})

# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN ACTIONS & NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/admin/api/notifications")
async def get_notifications(current_admin: str = Depends(verify_admin)):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM api_leads WHERE status = 'Pending'")
        pending_count = cursor.fetchone()[0]
        return {"pending_requests": pending_count}
    finally:
        cursor.close()
        conn.close()

# 🚀 NEW: Secure Download Endpoint for Quarantined Files
@app.get("/admin/api/download-quarantine/{filename}")
async def download_quarantined_file(filename: str, current_admin: str = Depends(verify_admin)):
    filepath = os.path.join(QUARANTINE_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File already deleted by 72-hour policy or does not exist.")
    return FileResponse(path=filepath, filename=filename, media_type='application/octet-stream')

@app.post("/admin/customers")
async def add_new_customer(customer: CustomerCreate, current_admin: str = Depends(verify_admin)):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM api_users WHERE customer_name = %s", (customer.customer_name,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Customer already exists.")

        raw_token = secrets.token_urlsafe(32)
        new_api_key = f"sk_live_{raw_token}"
        
        cursor.execute("""
            INSERT INTO api_users (api_key, customer_name, plan, credits_remaining, pages_processed_this_month)
            VALUES (%s, %s, %s, %s, 0)
        """, (new_api_key, customer.customer_name, customer.plan, customer.credits))
        
        cursor.execute("UPDATE api_leads SET status = 'Approved' WHERE company_name = %s", (customer.customer_name,))
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

@app.get("/admin/api/customer-insights/{api_key}")
async def get_customer_insights(api_key: str, current_admin: str = Depends(verify_admin)):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT customer_name, plan, credits_remaining, pages_processed_this_month FROM api_users WHERE api_key = %s", (api_key,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Customer not found")

        cursor.execute("""
            SELECT DATE(created_at) as date, SUM(pages_billed) as pages
            FROM api_requests_log
            WHERE api_key = %s AND status_code = 200
            GROUP BY DATE(created_at) ORDER BY date ASC
        """, (api_key,))
        usage_history = [{"date": str(row['date']), "pages": row['pages']} for row in cursor.fetchall()]

        cursor.execute("""
            SELECT created_at, error_message
            FROM api_requests_log
            WHERE api_key = %s AND status_code = 402
            ORDER BY created_at DESC LIMIT 10
        """, (api_key,))
        quota_errors = [{"time": row['created_at'].strftime('%Y-%m-%d %H:%M'), "msg": row['error_message']} for row in cursor.fetchall()]

        return {"user": user, "usage_history": usage_history, "quota_errors": quota_errors}
    finally:
        cursor.close()
        conn.close()

@app.get("/admin/api/customer-logs/{api_key}")
async def get_customer_logs(api_key: str, page: int = 1, limit: int = 10, current_admin: str = Depends(verify_admin)):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        offset = (page - 1) * limit
        cursor.execute("SELECT COUNT(*) as total FROM api_requests_log WHERE api_key = %s", (api_key,))
        total_records = cursor.fetchone()['total']

        cursor.execute("""
            SELECT created_at, file_name, pages_billed, status_code, processing_time_ms, file_size_bytes
            FROM api_requests_log
            WHERE api_key = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (api_key, limit, offset))
        
        logs = [{"time": log['created_at'].strftime('%Y-%m-%d %H:%M:%S'), 
                 "file_name": log['file_name'], 
                 "pages_billed": log['pages_billed'], 
                 "status_code": log['status_code'], 
                 "file_size_bytes": log['file_size_bytes'],
                 "processing_time_ms": log['processing_time_ms']} for log in cursor.fetchall()]

        total_pages = math.ceil(total_records / limit) if total_records > 0 else 1

        return {
            "total_records": total_records,
            "current_page": page,
            "total_pages": total_pages,
            "limit": limit,
            "logs": logs
        }
    finally:
        cursor.close()
        conn.close()


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

        # 🚀 Fetching quarantine_path for download links in HTML
        cursor.execute("SELECT created_at, status_code, file_name, error_message, quarantine_path FROM api_requests_log WHERE status_code > 399 ORDER BY created_at DESC LIMIT 10")
        recent_errors = cursor.fetchall()
        
        # Clean up path for frontend parsing
        for err in recent_errors:
            if err.get('quarantine_path'):
                err['q_filename'] = err['quarantine_path'].replace('\\', '/').split('/')[-1]
            else:
                err['q_filename'] = None

        cursor.execute("SELECT DATE(created_at) as date, SUM(pages_billed) as daily_pages FROM api_requests_log WHERE created_at >= CURRENT_DATE - INTERVAL '7 days' GROUP BY DATE(created_at) ORDER BY date ASC")
        chart_data = cursor.fetchall()

        cursor.execute("""
            SELECT u.customer_name, COUNT(l.log_id) as total_docs, SUM(l.pages_billed) as total_pages 
            FROM api_requests_log l JOIN api_users u ON l.api_key = u.api_key 
            WHERE l.status_code = 200 GROUP BY u.customer_name ORDER BY total_pages DESC
        """)
        company_stats = cursor.fetchall()

        cursor.execute("""
            SELECT u.customer_name, l.file_name, l.file_url, l.pages_billed, l.created_at, l.status_code, l.processing_time_ms, l.file_size_bytes 
            FROM api_requests_log l JOIN api_users u ON l.api_key = u.api_key 
            ORDER BY l.created_at DESC LIMIT 20
        """)
        file_logs = cursor.fetchall()
        
        cursor.execute("SELECT * FROM api_leads WHERE status = 'Pending' ORDER BY requested_at DESC")
        api_leads = cursor.fetchall()
        
        return templates.TemplateResponse(request=request, name="admin.html", context={
            "admin_user": current_admin, "users": users, "kpis": kpis, "errors": recent_errors,
            "chart_labels": json.dumps([str(row['date']) for row in chart_data]),
            "chart_values": json.dumps([row['daily_pages'] for row in chart_data]),
            "company_stats": company_stats, "file_logs": file_logs, "api_leads": api_leads
        })
    finally:
        cursor.close()
        conn.close()

@app.get("/logout", response_class=HTMLResponse)
async def logout_admin():
    return HTMLResponse(status_code=401, headers={"WWW-Authenticate": "Basic"}, content="""
        <html><head><title>Logged Out</title><style>body { font-family: Arial, sans-serif; text-align: center; padding-top: 100px; background-color: #f4f7f6; } .box { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; } a { color: #3498db; text-decoration: none; font-weight: bold; }</style></head><body><div class="box"><h2>🔒 Securely logged out.</h2><br><a href="/dashboard">Log back in</a></div></body></html>
    """)