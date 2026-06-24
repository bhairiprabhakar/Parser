import os
import re
import yaml
import json
import logging
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from parsers.erp_template_engine import AIFormatInferenceEngine, ERPTemplateEngine
    _SCHEMAS_PATH = os.path.join(os.path.dirname(__file__), '..', 'format_learned_schemas.json')
    _ai_format_inference = AIFormatInferenceEngine(schemas_path=_SCHEMAS_PATH)
    _erp_template_engine = ERPTemplateEngine(schemas_path=_SCHEMAS_PATH)
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False
    _ai_format_inference = None
    _erp_template_engine = None
    logger.info("ERP template engine not available, running in legacy mode")

# 🚀 1. BULLETPROOF MULTI-YAML LOADER (Scans the 'formats' folder)
FORMATS_REGISTRY = {}
_YAML_CONFIGS = []  # Keeping this for your legacy fallback logic

# Find the 'formats' folder (assuming it is one level up from the parsers folder)
current_dir = Path(__file__).parent
formats_dir = current_dir.parent / "formats"

if formats_dir.exists() and formats_dir.is_dir():
    for yaml_file in formats_dir.glob("*.yaml"):
        try:
            with open(yaml_file, "r", encoding="utf-8") as file:
                parsed_yaml = yaml.safe_load(file)
                
                if not parsed_yaml:
                    continue
                    
                # A: Load into New Dynamic Router (if it uses the 'formats' wrapper)
                if "formats" in parsed_yaml:
                    FORMATS_REGISTRY.update(parsed_yaml["formats"])
                    
                # B: Keep appending for your Legacy Router (skip formats_config.yaml which uses keyword matching)
                if "formats" not in parsed_yaml:
                    _YAML_CONFIGS.append(parsed_yaml)
                
            logger.info(f"✅ Loaded YAML config: {yaml_file.name}")
        except Exception as e:
            logger.error(f"Error reading YAML from {yaml_file.name}: {e}")
else:
    logger.warning(f"⚠️ 'formats' directory not found at {formats_dir}. Running purely in legacy mode.")

def identify_yaml_format(raw_text: str) -> str:
    """Scores text against strict YAML rules."""
    best_match = "UNKNOWN_FORMAT"
    highest_score = 0

    for format_name, rules in FORMATS_REGISTRY.items():
        score = sum(1 for kw in rules.get("keywords", []) if kw in raw_text)
        if score >= rules.get("min_matches", 999) and score > highest_score:
            highest_score = score
            best_match = format_name

    return best_match

# ── Absolute Import for Pipeline Transformers ──
from transformers.cleaners import preprocess_line

# ══════════════════════════════════════════════════════════════════════════════
#  SIGNATURE DETECTION UTILITIES & HEALERS
# ══════════════════════════════════════════════════════════════════════════════

def _heal_marg_csv_artifacts(raw_text: str) -> str:
    """Fixes Marg ERP's bug where CSV strings are printed onto PDFs, shattering rows vertically."""
    if '",' in raw_text or ',"' in raw_text or ',,' in raw_text:
        raw_text = re.sub(r'\r?\n\s*(?=[,"])', ' ', raw_text)
        raw_text = re.sub(r'(?<=[,"])\s*\r?\n', ' ', raw_text)
        raw_text = raw_text.replace('"', ' ')
        raw_text = re.sub(r',{2,}', ' ', raw_text)
        raw_text = re.sub(r'(?<!\d),(?!\d)', ' ', raw_text)
    return raw_text

def _compact_signature(text: str) -> str:
    return re.sub(r'[^A-Z0-9]+', '', str(text).upper())

def _header_views(lines: list, max_lines: int = 30) -> tuple:
    header_lines = [preprocess_line(str(line)) for line in lines[:max_lines]]
    joined = "\n".join(header_lines).upper()
    spaced = re.sub(r'[^A-Z0-9]+', ' ', joined)
    spaced = re.sub(r'\s{2,}', ' ', spaced).strip()
    compact = _compact_signature(joined)
    compact_lines = [_compact_signature(line) for line in header_lines if str(line).strip()]
    return spaced, compact, compact_lines

def _has_quarter_or_months(spaced: str) -> bool:
    words = set(re.findall(r'\b[A-Z0-9]+\b', spaced))
    month_count = len(words & {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"})
    if month_count >= 2: return True
    return bool(words & {"Q1", "Q2", "Q3", "Q4", "QTR", "QUARTERLY"})

def _line_order(compact_lines: list, *phrases: str) -> bool:
    needles = [_compact_signature(phrase) for phrase in phrases]
    for line in compact_lines:
        pos = -1
        matched = True
        for needle in needles:
            new_pos = line.find(needle, pos + 1)
            if new_pos < 0:
                matched = False
                break
            pos = new_pos
        if matched: return True
    return False

def _line_store_month_total_order(compact_lines: list, total_before_month: bool) -> bool:
    month_markers = ["MONTH", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    for line in compact_lines:
        store_pos = line.find("STORE")
        total_pos = line.find("TOTAL")
        month_positions = [line.find(marker) for marker in month_markers if line.find(marker) >= 0]
        if store_pos < 0 or total_pos < 0 or not month_positions: continue
        month_pos = min(month_positions)
        if total_before_month and store_pos < total_pos < month_pos: return True
        if not total_before_month and store_pos < month_pos < total_pos: return True
    return False

def detect_report_format(lines: list) -> dict:
    spaced, compact, compact_lines = _header_views(lines, max_lines=30)

    def found(fmt: str, parser_mode: str, reason: str) -> dict:
        return {"Format": fmt, "ReportType": parser_mode, "Reason": reason}

    for config in _YAML_CONFIGS:
        # Skip formats_config.yaml (has "formats:" wrapper, uses keyword matching)
        if "formats" in config:
            continue
        try:
            req_all = config.get("fingerprint_keywords", {}).get("require_all", [])
            req_any = config.get("fingerprint_keywords", {}).get("require_any", [])
            exc_any = config.get("fingerprint_keywords", {}).get("exclude_any", [])
            special = config.get("special_condition")
            
            if exc_any and any(_compact_signature(kw) in compact for kw in exc_any):
                continue
            
            has_all = True
            if req_all:
                needles = [_compact_signature(phrase) for phrase in req_all]
                has_all = any(all(needle in line for needle in needles) for line in compact_lines)
            
            has_any = True
            if req_any:
                has_any = any(_compact_signature(kw) in compact for kw in req_any)
            
            condition_met = True
            if special == "check_agency_hint_in_header" and has_all:
                no_ag = "NOTBETHERE" in compact or "WITHOUTAGENCY" in compact
                has_ag = any(re.search(r'\b(AGENCY|AGENCIES|DISTRIBUTORS?|TRADERS?|PHARMA|MEDICALS?|GSTIN|DL\s*NO)\b', l, re.IGNORECASE) for l in lines[:10]) and not no_ag
                fmt_id = "FORMAT_01" if has_ag else "FORMAT_02"
                return found(fmt_id, config.get("report_type", "UNKNOWN"), "YAML: " + config.get('format_id', 'UNKNOWN'))
                
            elif special == "line_order_amount_then_store":
                condition_met = _line_order(compact_lines, "AMOUNT", "STORE NAME")
            elif special == "store_month_total_order_false":
                condition_met = _line_store_month_total_order(compact_lines, False)
            elif special == "store_month_total_order_true":
                condition_met = _line_store_month_total_order(compact_lines, True)
            elif special == "has_quarter_or_months":
                condition_met = _has_quarter_or_months(spaced)
            elif special == "or_sales_book_alternative_layout":
                alt_met = ("SALESBOOK" in compact or "SALES BOOK" in spaced) and (all(_compact_signature(k) in compact for k in ["BILLNO", "PARTYNAME"]) or all(_compact_signature(k) in compact for k in ["BILLAMT", "TAXABLE"]))
                if not has_all and alt_met:
                    return found(config["format_id"], config.get("report_type", "UNKNOWN"), "YAML alt-layout")

            if has_all and has_any and condition_met:
                report_type = config.get("report_type", "UNKNOWN")
                
                # 🚀 FIX: Prevent FORMAT_18 from misclassifying Mfr/Customer reports as standard SUMMARY
                if report_type == "SUMMARY" and any(k in spaced for k in ["MFR/CUSTOMER", "MFR CUSTOMER", "CUSTOMER WISE SALES SUMMARY"]):
                    report_type = "MFR_CUSTOMER_SUMMARY"
                    
                return found(config.get("format_id", "UNKNOWN"), report_type, "YAML: " + config.get('format_id', 'UNKNOWN'))
        except Exception as e:
            logger.warning("Failed to parse YAML Config: %s", e)

    head_text_clean = spaced
    if any(kw in head_text_clean for kw in ["QUARTERLY", "QTR", "Q1", "Q2", "Q3", "Q4", "AMT OCT", "AMT NOV", "AMT DEC"]) or \
       ("JAN" in head_text_clean and "FEB" in head_text_clean) or \
       ("APR" in head_text_clean and "MAY" in head_text_clean) or \
       ("JUL" in head_text_clean and "AUG" in head_text_clean) or \
       ("OCT" in head_text_clean and "NOV" in head_text_clean) or _has_quarter_or_months(head_text_clean):
        return found("LEGACY_QUARTERLY_SUMMARY", "QUARTERLY_SUMMARY", "fallback quarter/month keywords")
        
    if "SALES ANALYSIS" in head_text_clean and "ALL PARTIES" in head_text_clean:
        return found("LEGACY_SALES_ANALYSIS_SUMMARY", "SALES_ANALYSIS_SUMMARY", "fallback sales analysis")
        
    if "COMPANY WISE CUSTOMER SALES" in head_text_clean:
        return found("LEGACY_SUMMARY", "SUMMARY", "fallback company wise")
        
    if "CUSTOMER COMPANY AND PRODUCT SALES" in head_text_clean:
        return found("LEGACY_CUSTOMER_PRODUCT_SALES", "CUSTOMER_PRODUCT_SALES", "fallback customer company product")
        
    if any(_compact_signature(kw) in compact for kw in ["CUSTOMER-WISE PRODUCT-WISE", "CUSTOMER WISE PRODUCT WISE", "CUSTOMER-PRODUCT WISE", "CUSTOMER PRODUCT WISE"]):
        return found("LEGACY_CUSTOMER_WISE_PRODUCT_WISE", "CUSTOMER_WISE_PRODUCT_WISE", "fallback customer wise product wise")
        
    if any(_compact_signature(kw) in compact for kw in ["MFR/CUSTOMER WISE SALES", "CUSTOMER WISE SALES SUMMARY", "MFR / CUSTOMER WISE", "GROUP VS. CUSTOMER", "GROUP VS CUSTOMER"]):
        return found("LEGACY_MFR_CUSTOMER_SUMMARY", "MFR_CUSTOMER_SUMMARY", "fallback mfr customer summary")
        
    if "PARTY WISE SALES BOOK" in head_text_clean:
        return found("LEGACY_PARTY_SALES_BOOK", "PARTY_SALES_BOOK", "fallback party sales book")
        
    if any(kw in head_text_clean for kw in ["GST INVOICE", "TAX INVOICE", "INVOICE NO"]) or ("SGST" in head_text_clean and "CGST" in head_text_clean and "HSN" in head_text_clean):
        return found("LEGACY_INVOICE_SINGLE_PARTY", "INVOICE_SINGLE_PARTY", "fallback invoice keywords")
        
    if "SALE REGISTER" in head_text_clean or "SALES REGISTER" in head_text_clean:
        return found("LEGACY_SALE_REGISTER_ITEMIZED", "SALE_REGISTER_ITEMIZED" if "QTY" in head_text_clean or "ITEM" in head_text_clean else "SUMMARY", "fallback register")
        
    if "PARTYPRODUCTWISE" in compact or "PARTY PRODUCTWISE" in head_text_clean: 
        return found("LEGACY_PARTY_PRODUCT_WISE", "PARTY_PRODUCT_WISE", "fallback party-product")
        
    if "ITEM WISE" in head_text_clean and "PARTY" not in head_text_clean and "AREA" not in head_text_clean: 
        return found("LEGACY_ITEM_WISE", "ITEM_WISE", "fallback item-wise")
        
    if "SALE SUMMARY" in head_text_clean or "AMT" in head_text_clean or any(kw in head_text_clean for kw in ["PAERIES", "SALE DATA", "RETAILER SALE", "SUMMARY", "DEALER"]):
        return found("LEGACY_SUMMARY", "SUMMARY", "fallback summary keywords")

    return found("LEGACY_UNIVERSAL_TWO_COLUMN", "UNIVERSAL_TWO_COLUMN", "absolute fallback universal two-column parser")

# ══════════════════════════════════════════════════════════════════════════════
#  THE PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

# 🛑 YOUR OLD ROUTER (Renamed) 🛑
def legacy_route_and_parse(raw_text: str, source_file: str = "") -> dict:
    # REMOVED the healer from here, we will do it earlier!
    fmt = detect_report_format(raw_text.split('\n'))
    
    if fmt["ReportType"] in ("SCHEMA_INFERRED", "UNIVERSAL_TWO_COLUMN"):
        try:
            from parsers.schema_engine import parse_by_schema_inference
            data = parse_by_schema_inference(raw_text, source_file=source_file)
            if sum(len(s.get("Items",[])) for a in data.get("Areas",[]) for s in a.get("Stores",[])) > 0:
                return data
        except Exception as e:
            logger.warning("Schema engine error: %s", e, exc_info=True)
            
    from parsers.legacy_machine import parse_text
    return parse_text(raw_text)

# 🚀 THE NEW SAFE GATEWAY 🚀
def _dynamic_parse(raw_text: str, format_config: dict, source_file: str) -> dict:
    """Instantiate a parser from YAML config and run extraction."""
    module_name = format_config.get("parser_module")
    class_name = format_config.get("parser_class")
    if not module_name or not class_name:
        logger.warning(f"Format {format_config.get('format_id')} has no parser_module/parser_class. Falling back.")
        return legacy_route_and_parse(raw_text, source_file)

    # Normalize keys for legacy parsers that expect Format/ReportType
    format_meta = {
        "Format": format_config.get("format_id", format_config.get("format_name", "UNKNOWN")),
        "ReportType": format_config.get("report_type", "UNKNOWN"),
        "Reason": "YAML Dynamic Match",
    }

    logger.info(f"Dynamic Format [{format_meta['Format']}] for {source_file}")
    try:
        module = importlib.import_module(module_name)
        ParserClass = getattr(module, class_name)
        parser_instance = ParserClass(format_meta=format_meta)
        raw_lines = raw_text.split('\n')
        return parser_instance.extract(raw_lines)
    except Exception as e:
        logger.error(f"Dynamic Parser Failed: {e}. Falling back to legacy.")
        return legacy_route_and_parse(raw_text, source_file)


def route_and_parse(raw_text: str, source_file: str) -> dict:
    """
    Four-tier routing:
      0. AI format inference (learned schemas from previous runs)
      1. Strict YAML keyword match (formats_config.yaml style)
      2. Legacy YAML fingerprint match (format_*.yaml style)
      3. Pure legacy fallback (state machine / schema engine)
    """
    # 🚀 FIX: Heal the text before ANY routing or parsing happens!
    raw_text = _heal_marg_csv_artifacts(raw_text)

    # ── Tier 0: AI Format Inference ──
    if _AI_AVAILABLE:
        ai_format, ai_conf = _ai_format_inference.infer_format(raw_text)
        if ai_format != "UNKNOWN" and ai_conf > 0.5:
            logger.info(f"AI Format Inference: {ai_format} (confidence={ai_conf:.2f})")
            from parsers.legacy_machine import parse_text
            from parsers.universal_router import detect_report_format
            fmt = detect_report_format(raw_text.split('\n'))
            fmt["Format"] = ai_format
            data = parse_text(raw_text, format_meta=fmt)
            if _erp_template_engine:
                schema = _erp_template_engine.get_schema(ai_format)
                if schema:
                    data['_erp_schema'] = schema
            data['_ai_format'] = ai_format
            data['_ai_confidence'] = ai_conf
            return data

    # ── Tier 1: formats_config.yaml keyword matching ──
    format_name = identify_yaml_format(raw_text)
    if format_name != "UNKNOWN_FORMAT":
        config = FORMATS_REGISTRY[format_name]
        return _dynamic_parse(raw_text, config, source_file)

    # ── Tier 2: format_*.yaml fingerprint matching ──
    lines = raw_text.split('\n')
    fmt = detect_report_format(lines)
    format_id = fmt.get("Format", "")
    if format_id and not format_id.startswith("LEGACY_"):
        # Find the matching YAML config by format_id
        for config in _YAML_CONFIGS:
            if config.get("format_id") == format_id:
                # Merge detect_report_format's ReportType (may override, e.g. SUMMARY → MFR_CUSTOMER_SUMMARY)
                merged_config = {**config, "report_type": fmt.get("ReportType", config.get("report_type", "UNKNOWN"))}
                return _dynamic_parse(raw_text, merged_config, source_file)

    # ── Tier 3: Pure legacy fallback ──
    logger.info(f"Delegating to Legacy Router for {source_file}...")
    return legacy_route_and_parse(raw_text, source_file)