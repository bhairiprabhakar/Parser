import os
import sys
import re
import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ── Engine imports (fail-fast with a helpful message) ─────────────────────────
try:
    import pdfplumber
except ImportError:
    log.warning("❌  pdfplumber not found.  Run:  pip install pdfplumber")

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    _PDFMINER_AVAILABLE = True
except ImportError:
    _PDFMINER_AVAILABLE = False

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

# 🚀 USER ADDON: Tesseract + OpenCV for Scanned PDFs
try:
    from pdf2image import convert_from_path
    import pytesseract
    import cv2
    import numpy as np
    _OCR_EXTRA_AVAILABLE = True
except ImportError:
    _OCR_EXTRA_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL TEXT EXTRACTION (NATIVE FIRST ENGINE)
# ══════════════════════════════════════════════════════════════════════════════

_PLUMBER_LAYOUT_KWARGS = {
    "layout": True,
    "use_text_flow": False,
    "x_tolerance": 3,
    "y_tolerance": 3
}

def _preprocess_image_cv2(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh

def _extract_with_pdfplumber(pdf_path: str) -> str:
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text(**_PLUMBER_LAYOUT_KWARGS) or ""
            except Exception as exc:
                log.warning("pdfplumber failed on page %d: %s", page_num, exc)
                text = page.extract_text() or ""   
            pages_text.append(text)
    return "\x0c".join(pages_text)

def _extract_spreadsheet_text(filepath: str) -> str:
    text_lines = []
    ext = Path(filepath).suffix.lower()
    try:
        if ext == '.csv':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    cells = [str(c).strip().replace('\n', ' ').replace('\r', ' ') for c in row if str(c).strip()]
                    if cells: text_lines.append(" ".join(cells))
                    
        elif ext in ['.xlsx', '.xls']:
            success = False
            if pd is not None:
                try:
                    df = pd.read_excel(filepath, header=None, dtype=str)
                    df = df.fillna("")
                    for _, row in df.iterrows():
                        cells = [str(c).strip().replace('\n', ' ').replace('\r', ' ') for c in row if str(c).strip() and str(c).strip() != 'nan']
                        if cells: text_lines.append(" ".join(cells))
                    success = True
                except Exception as e:
                    log.warning("Standard Excel engine failed. Attempting Universal HTML/TSV Fallback for Marg ERP...")
            
            if not success:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    if '<table' in content.lower():
                        content = re.sub(r'</tr>', '\n', content, flags=re.IGNORECASE)
                        content = re.sub(r'</td>', '\t', content, flags=re.IGNORECASE)
                        content = re.sub(r'<[^>]+>', '', content)
                        for line in content.split('\n'):
                            cells = [c.strip() for c in line.split('\t') if c.strip()]
                            if cells: text_lines.append(" ".join(cells))
                            
                    elif '\t' in content or ',' in content:
                        f.seek(0)
                        delim = '\t' if '\t' in content else ','
                        reader = csv.reader(f, delimiter=delim)
                        for row in reader:
                            cells = [str(c).strip() for c in row if str(c).strip()]
                            if cells: text_lines.append(" ".join(cells))
                            
    except Exception as e:
        log.error("Failed to read spreadsheet %s: %s", filepath, e)
        
    return "\n".join(text_lines)


# 🚀 B7 FIX: Thread-safe PaddleOCR Engine Loader
_PADDLE_ENGINE = None

def _get_paddle():
    global _PADDLE_ENGINE
    if _PADDLE_ENGINE is None:
        try:
            # 🚀 FIX: Prevent silent Windows crashes from threading!
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            from paddleocr import PaddleOCR
            import warnings
            os.environ['FLAGS_enable_pir_api'] = '0'
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            
            log.info("Initializing PaddleOCR Engine...")
            _PADDLE_ENGINE = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
        except ImportError:
            log.error("❌  paddleocr not found. Run: pip install paddlepaddle paddleocr")
    return _PADDLE_ENGINE


def _extract_image_text_paddle(filepath: str) -> str:
    try:
        from PIL import Image
        import tempfile
    except ImportError:
        log.error("❌ PIL not found. Run: pip install Pillow")
        return ""

    filepath_to_process = filepath
    temp_file_path = None
    
    try:
        with Image.open(filepath) as img:
            MAX_SIDE = 1536 
            if max(img.size) > MAX_SIDE:
                log.info("Resizing image to max %d pixels to prevent memory crash...", MAX_SIDE)
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
                
                fd, temp_file_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                img.convert('RGB').save(temp_file_path, quality=85)
                filepath_to_process = temp_file_path
    except Exception as e:
        log.warning("Could not auto-resize image, proceeding with original. Error: %s", e)

    paddle_eng = _get_paddle()
    if paddle_eng is None: return ""

    log.info("Running PaddleOCR on %s...", filepath_to_process)
    
    try:
        result_gen = paddle_eng.predict(filepath_to_process)
        result = list(result_gen)
    except Exception as e:
        log.error("PaddleOCR failed to process the image: %s", e)
        result = []
        
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except Exception:
            pass
    
    if not result or result[0] is None:
        return ""

    blocks = []
    page_data = result[0]
    
    if isinstance(page_data, dict) and 'dt_polys' in page_data:
        text_key = 'rec_texts' if 'rec_texts' in page_data else 'rec_text'
        if text_key not in page_data:
            text_key = next((k for k in page_data.keys() if 'text' in k.lower()), None)
            
        if text_key and text_key in page_data:
            for box, text in zip(page_data['dt_polys'], page_data[text_key]):
                lines_in_text = text.split('\n')
                y_min = min(pt[1] for pt in box)
                y_max = max(pt[1] for pt in box)
                x_min = min(pt[0] for pt in box)
                x_max = max(pt[0] for pt in box)
                line_height = (y_max - y_min) / max(1, len(lines_in_text))
                
                for i, sub_text in enumerate(lines_in_text):
                    sub_text = sub_text.strip()
                    if not sub_text: continue
                    sub_y_min = y_min + i * line_height
                    sub_y_max = sub_y_min + line_height
                    blocks.append({
                        'text': sub_text,
                        'center_y': (sub_y_min + sub_y_max) / 2.0,
                        'y_min': sub_y_min,
                        'y_max': sub_y_max,
                        'min_x': x_min,
                        'max_x': x_max,
                        'height': line_height
                    })
    else:
        for line in page_data:
            box = line[0]        
            text = line[1][0]   
            
            lines_in_text = text.split('\n')
            y_min = min(pt[1] for pt in box)
            y_max = max(pt[1] for pt in box)
            x_min = min(pt[0] for pt in box)
            x_max = max(pt[0] for pt in box)
            line_height = (y_max - y_min) / max(1, len(lines_in_text))
            
            for i, sub_text in enumerate(lines_in_text):
                sub_text = sub_text.strip()
                if not sub_text: continue
                sub_y_min = y_min + i * line_height
                sub_y_max = sub_y_min + line_height
                blocks.append({
                    'text': sub_text,
                    'center_y': (sub_y_min + sub_y_max) / 2.0,
                    'y_min': sub_y_min,
                    'y_max': sub_y_max,
                    'min_x': x_min,
                    'max_x': x_max,
                    'height': line_height
                })

    blocks.sort(key=lambda b: b['center_y'])
    lines = []
    current_line = []
    
    for block in blocks:
        if not current_line:
            current_line.append(block)
            continue
            
        avg_center_y = sum(b['center_y'] for b in current_line) / len(current_line)
        avg_height = sum(b['height'] for b in current_line) / len(current_line)
        
        if abs(block['center_y'] - avg_center_y) < (avg_height * 0.6):
            current_line.append(block)
        else:
            current_line.sort(key=lambda b: b['min_x'])
            lines.append(" ".join([b['text'] for b in current_line]))
            current_line = [block]
            
    if current_line:
        current_line.sort(key=lambda b: b['min_x'])
        lines.append(" ".join([b['text'] for b in current_line]))

    return "\n".join(lines)


def extract_raw_text(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext in ['.csv', '.xlsx', '.xls']:
        log.info("Extracting text from Spreadsheet/OCR output (%s) …", ext)
        return _extract_spreadsheet_text(filepath)
        
    elif ext == '.pdf':
        log.info("📄 Priority 1: Extracting via Native Layout Engine (pdfplumber)...")
        text = _extract_with_pdfplumber(filepath)
        
        if len(text.strip()) < 100:
            log.warning("⚠️ Native extraction yielded almost no text. This appears to be a SCANNED PDF.")
            if _OCR_EXTRA_AVAILABLE:
                log.info("📸 Engaging Heavy OCR (Tesseract) for scanned document...")
                try:
                    images = convert_from_path(filepath, dpi=300)
                    ocr_text = ""
                    for img in images:
                        processed = _preprocess_image_cv2(img)
                        ocr_text += pytesseract.image_to_string(processed, config='--oem 3 --psm 6') + "\n"
                    text = re.sub(r'[|]', ' ', ocr_text)
                    text = re.sub(r'\s+', ' ', text)
                    log.info("✅ Scanned PDF OCR extraction complete.")
                    return text
                except Exception as e:
                    log.error("❌ Tesseract processing failed: %s", e)
            else:
                log.warning("⚠️ Tesseract is not installed or available in PATH. Skipping Heavy OCR.")
            
            if _FITZ_AVAILABLE:
                log.info("Attempting PyMuPDF raw text fallback...")
                doc = fitz.open(filepath)
                text = "\n".join([page.get_text("text") for page in doc])
                doc.close()
            elif _PDFMINER_AVAILABLE:
                log.info("Attempting pdfminer raw text fallback...")
                text = pdfminer_extract(filepath)
                
        else:
            log.info("✅ Native text extraction successful (Extracted %d characters). Skipping heavy OCR.", len(text))

        return text
        
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        log.info("Extracting text via AI OCR (Paddle) …")
        return _extract_image_text_paddle(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")