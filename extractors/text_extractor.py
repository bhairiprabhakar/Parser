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
                    cells = [str(c).strip().replace('\n', ' ').replace('\r', ' ')
                             for c in row if str(c).strip()]
                    if cells:
                        text_lines.append("\t".join(cells))

        elif ext in ['.xlsx', '.xls']:
            success = False
            if pd is not None:
                try:
                    df = pd.read_excel(filepath, header=None, dtype=str)
                    df = df.fillna("")
                    for _, row in df.iterrows():
                        cells = [
                            str(c).strip().replace('\n', ' ').replace('\r', ' ')
                            for c in row
                            if str(c).strip() and str(c).strip() != 'nan'
                        ]
                        if cells:
                            text_lines.append("\t".join(cells))
                    success = True
                except Exception as e:
                    log.warning("Standard Excel engine failed (%s). Attempting Universal HTML/TSV Fallback...", e)

            if not success:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    if '<table' in content.lower():
                        content = re.sub(r'</tr>', '\n', content, flags=re.IGNORECASE)
                        content = re.sub(r'</td>', '\t', content, flags=re.IGNORECASE)
                        content = re.sub(r'<[^>]+>', '', content)
                        for line in content.split('\n'):
                            cells = [c.strip() for c in line.split('\t') if c.strip()]
                            if cells:
                                text_lines.append("\t".join(cells))

                    elif '\t' in content or ',' in content:
                        f.seek(0)
                        delim = '\t' if '\t' in content else ','
                        reader = csv.reader(f, delimiter=delim)
                        for row in reader:
                            cells = [str(c).strip() for c in row if str(c).strip()]
                            if cells:
                                text_lines.append("\t".join(cells))

    except Exception as e:
        log.error("Failed to read spreadsheet %s: %s", filepath, e)

    return "\n".join(text_lines)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTERPRISE HYBRID PDF EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

def analyze_page_quality(text: str) -> dict:
    if not text or not text.strip():
        return {"char_count": 0, "num_density": 0.0, "financial_rows": 0, "table_rows": 0}
    char_count = len(text)
    digits_count = sum(c.isdigit() for c in text)
    num_density = (digits_count / char_count) if char_count > 0 else 0
    financial_pattern = r'\b\d+[\.,]\d{2}\b'
    financial_rows = len(re.findall(financial_pattern, text))
    lines = text.split('\n')
    table_rows = sum(1 for line in lines if len(re.findall(r'\b\d+(?:\.\d+)?\b', line)) >= 3)
    return {"char_count": char_count, "num_density": num_density, "financial_rows": financial_rows, "table_rows": table_rows}


def is_weak_extraction(text: str, page_num: int) -> bool:
    metrics = analyze_page_quality(text)
    if metrics["char_count"] < 150:
        return True
    if metrics["num_density"] < 0.03:
        return True
    return False


def _extract_pdf_hybrid(filepath: str) -> str:
    final_pages_text = []
    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            log.info("PDF opened successfully. Total pages: %d", total_pages)
            for page_num_0_idx, page in enumerate(pdf.pages):
                page_num = page_num_0_idx + 1
                try:
                    native_text = page.extract_text(
                        layout=True, x_tolerance=3, y_tolerance=3) or ""
                except Exception as exc:
                    log.warning("pdfplumber failed on page %d: %s", page_num, exc)
                    native_text = ""
                if not is_weak_extraction(native_text, page_num):
                    final_pages_text.append(native_text)
                    continue
                log.info("Page %d is weak/image-based. Running PaddleOCR ...", page_num)
                ocr_text = ""
                try:
                    import tempfile
                    images = convert_from_path(
                        filepath, dpi=300,
                        first_page=page_num, last_page=page_num)
                    if images:
                        fd, temp_path = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        images[0].save(temp_path)
                        ocr_text = _extract_image_text_paddle(temp_path)
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                        log.info("Page %d PaddleOCR done. text_length=%d",
                                 page_num, len(ocr_text))
                except Exception as e:
                    log.error("PaddleOCR fallback failed for page %d: %s", page_num, e)
                final_pages_text.append(ocr_text if ocr_text.strip() else native_text)
    except Exception as e:
        log.error("Fatal error opening PDF %s: %s", filepath, e)
    return "\x0c".join(final_pages_text)


# 🚀 B7 FIX: Thread-safe PaddleOCR Engine Loader
_PADDLE_ENGINE = None

_MIN_CONF = 0.45
_MIN_TOLERANCE = 15.0


def _get_paddle():
    global _PADDLE_ENGINE
    if _PADDLE_ENGINE is None:
        try:
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


def _paddle_extract_blocks(page_data: list) -> list:
    """Convert PaddleOCR page result into block dicts with position, width & confidence."""
    blocks = []
    if isinstance(page_data, dict) and 'dt_polys' in page_data:
        text_key = 'rec_texts' if 'rec_texts' in page_data else 'rec_text'
        if text_key not in page_data:
            text_key = next((k for k in page_data.keys() if 'text' in k.lower()), None)
        if text_key and text_key in page_data:
            confs = page_data.get('rec_scores', [None] * len(page_data['dt_polys']))
            for box, text, conf in zip(page_data['dt_polys'], page_data[text_key], confs):
                conf_val = float(conf) if conf is not None else 0.5
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
                        'text': sub_text, 'confidence': conf_val,
                        'center_y': (sub_y_min + sub_y_max) / 2.0,
                        'y_min': sub_y_min, 'y_max': sub_y_max,
                        'min_x': x_min, 'max_x': x_max, 'height': line_height,
                        'char_width': (x_max - x_min) / max(1, len(sub_text)),
                    })
    else:
        for line in page_data:
            box = line[0]
            text = line[1][0]
            conf_val = float(line[1][1]) if len(line[1]) > 1 and line[1][1] is not None else 0.5
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
                    'text': sub_text, 'confidence': conf_val,
                    'center_y': (sub_y_min + sub_y_max) / 2.0,
                    'y_min': sub_y_min, 'y_max': sub_y_max,
                    'min_x': x_min, 'max_x': x_max, 'height': line_height,
                    'char_width': (x_max - x_min) / max(1, len(sub_text)),
                })
    return [b for b in blocks if b['confidence'] >= _MIN_CONF]


def _cluster_rows_anchor(blocks: list) -> list:
    """Anchor-based row clustering using median-height tolerance."""
    if not blocks:
        return []
    heights = [b["height"] for b in blocks]
    median_h = float(np.median(heights)) if heights else 20.0
    tolerance = max(median_h * 1.0, _MIN_TOLERANCE)
    sorted_blocks = sorted(blocks, key=lambda b: b["center_y"])
    row_anchors = []
    row_buckets = []
    for blk in sorted_blocks:
        cy = blk["center_y"]
        best_idx, best_dist = -1, float("inf")
        for i, anchor in enumerate(row_anchors):
            d = abs(cy - anchor)
            if d <= tolerance and d < best_dist:
                best_dist, best_idx = d, i
        if best_idx >= 0:
            row_buckets[best_idx].append(blk)
        else:
            row_anchors.append(cy)
            row_buckets.append([blk])
    return [sorted(bucket, key=lambda b: b["min_x"]) for bucket in row_buckets]


def _detect_column_boundaries(all_blocks: list) -> list:
    """Detect column boundaries via X-position histogram."""
    if not all_blocks:
        return []
    x_centres = [(b["min_x"] + b["max_x"]) / 2.0 for b in all_blocks]
    x_min_all = min(x_centres)
    x_max_all = max(x_centres)
    page_w = x_max_all - x_min_all
    if page_w < 10:
        return []
    buckets = 200
    hist = [0] * buckets
    for xc in x_centres:
        idx = min(int((xc - x_min_all) / page_w * buckets), buckets - 1)
        hist[idx] += 1
    smoothed = [0.0] * buckets
    for i in range(buckets):
        smoothed[i] = sum(hist[max(0, i - 1):i + 2]) / 3.0
    threshold = max(smoothed) * 0.20
    in_gap = False
    gap_starts = []
    for i, v in enumerate(smoothed):
        if v < threshold and not in_gap:
            in_gap = True
            gap_starts.append(i)
        elif v >= threshold:
            in_gap = False
    col_starts = sorted(set(x_min_all + (gs / buckets) * page_w for gs in gap_starts))
    min_sep = page_w * 0.02
    filtered = [col_starts[0]] if col_starts else []
    for x in col_starts[1:]:
        if x - filtered[-1] >= min_sep:
            filtered.append(x)
    log.debug("Detected %d column boundaries", len(filtered))
    return filtered


def _assign_column_slot(block: dict, col_boundaries: list) -> int:
    """Assign block to the best overlapping column."""
    if not col_boundaries:
        return 0
    tok_w = max(block["max_x"] - block["min_x"], 1.0)
    bounds = col_boundaries + [col_boundaries[-1] + tok_w * 3]
    best_col, best_overlap = 0, -1.0
    for i in range(len(bounds) - 1):
        col_x0, col_x1 = bounds[i], bounds[i + 1]
        overlap = min(block["max_x"], col_x1) - max(block["min_x"], col_x0)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = i
    return best_col


def _row_to_text(row: list, col_boundaries: list = None) -> str:
    """Assemble a row of blocks into a tab-separated text line (column-aware)."""
    if not row:
        return ""
    if col_boundaries and len(col_boundaries) >= 5:
        slots = {i: [] for i in range(len(col_boundaries))}
        for blk in row:
            slot = _assign_column_slot(blk, col_boundaries)
            slots[slot].append(blk["text"])
        parts = [" ".join(slots[i]).strip() for i in range(len(col_boundaries))]
        while parts and not parts[-1]:
            parts.pop()
        return "\t".join(parts)
    else:
        avg_cw = float(np.mean([b["char_width"] for b in row])) or 8.0
        parts = []
        for i, blk in enumerate(row):
            gap = (blk["min_x"] - row[i - 1]["max_x"]) if i > 0 else 0.0
            if gap < 0:
                sep = " "
            elif gap <= 1.0 * avg_cw:
                sep = ""
            elif gap <= 3.5 * avg_cw:
                sep = " "
            else:
                sep = "\t"
            if parts:
                parts.append(sep + blk["text"])
            else:
                parts.append(blk["text"])
        return "".join(parts)


def _run_paddle_on_image(img_path: str) -> tuple:
    """Run PaddleOCR on an image path; returns (text, block_count, score)."""
    paddle_eng = _get_paddle()
    if paddle_eng is None:
        return "", 0, 0.0
    try:
        result_gen = paddle_eng.predict(img_path)
        result = list(result_gen)
    except Exception as e:
        log.error("PaddleOCR failed on %s: %s", img_path, e)
        return "", 0, 0.0
    if not result or result[0] is None:
        return "", 0, 0.0
    page_data = result[0]
    blocks = _paddle_extract_blocks(page_data)
    if not blocks:
        return "", 0, 0.0
    rows = _cluster_rows_anchor(blocks)
    col_boundaries = _detect_column_boundaries(blocks)
    lines = [_row_to_text(r, col_boundaries) for r in rows if r]
    text = "\n".join(lines)
    score = len(blocks) * (1.0 + sum(b.get('confidence', 0.5) for b in blocks) / len(blocks))
    return text, len(blocks), score


def _extract_image_text_paddle(filepath: str) -> str:
    """Extract text from an image using PaddleOCR with multi-pass preprocessing."""
    import tempfile
    from PIL import Image

    passes = []
    try:
        with Image.open(filepath) as img:
            MAX_SIDE = 1536
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            # Pass 1: original (resized if needed)
            fd, p1 = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            img.convert('RGB').save(p1, quality=85)
            passes.append(("original", p1))
            # Pass 2: CLAHE enhanced
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            boosted = clahe.apply(gray)
            fd, p2 = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            cv2.imwrite(p2, cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR))
            passes.append(("clahe", p2))
    except Exception as e:
        log.warning("Multi-pass generation failed, falling back to original: %s", e)
        passes = [("original", filepath)]

    best_text, best_score = "", 0.0
    for name, pass_path in passes:
        try:
            text, block_count, score = _run_paddle_on_image(pass_path)
            if score > best_score:
                best_text, best_score = text, score
                log.info("Pass '%s' score=%.1f blocks=%d chars=%d",
                         name, score, block_count, len(text))
        finally:
            if name != "original" and pass_path != filepath and os.path.exists(pass_path):
                try:
                    os.remove(pass_path)
                except Exception:
                    pass
    # Clean up pass-1 temp file
    if passes and passes[0][0] == "original" and passes[0][1] != filepath:
        try:
            os.remove(passes[0][1])
        except Exception:
            pass
    return best_text


def extract_raw_text(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext in ['.csv', '.xlsx', '.xls']:
        log.info("Extracting text from Spreadsheet/OCR output (%s)", ext)
        return _extract_spreadsheet_text(filepath)
    elif ext == '.pdf':
        log.info("Launching Hybrid PDF Engine (pdfplumber + Enterprise OCR)")
        return _extract_pdf_hybrid(filepath)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        log.info("Running PaddleOCR on image ...")
        return _extract_image_text_paddle(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def extract_with_enterprise_ocr(filepath: str,
                                 dpi: int = 300) -> str:
    """Extract text using PaddleOCR (replaces legacy enterprise pipeline)."""
    return extract_raw_text(filepath)