import os
import re
import yaml
import json
import logging
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from parsers.erp_template_engine import AIFormatInferenceEngine, ERPTemplateEngine, FormatInferenceResult
    _ai_format_inference = AIFormatInferenceEngine.instance()
    _erp_template_engine = ERPTemplateEngine.instance()
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

def _has_all(compact: str, *phrases: str) -> bool:
    return all(_compact_signature(phrase) in compact for phrase in phrases)

def _line_has_all(compact_lines: list, *phrases: str) -> bool:
    needles = [_compact_signature(phrase) for phrase in phrases]
    return any(all(needle in line for needle in needles) for line in compact_lines)

def _legacy_report_type_from_header(spaced: str, compact: str) -> tuple:
    head_text_clean = spaced
    report_type = "UNIVERSAL_TWO_COLUMN"
    reason = "fallback universal two-column parser"

    if any(kw in head_text_clean for kw in ["GST INVOICE", "TAX INVOICE", "INVOICE NO"]) or ("SGST" in head_text_clean and "CGST" in head_text_clean and "HSN" in head_text_clean):
        return ("INVOICE_SINGLE_PARTY", "invoice keywords (before quarterly check)")
    if any(kw in head_text_clean for kw in ["QUARTERLY", "QTR", "Q1", "Q2", "Q3", "Q4", "AMT OCT", "AMT NOV", "AMT DEC"]) or \
       _has_quarter_or_months(head_text_clean):
        return ("QUARTERLY_SUMMARY", "quarter/month keywords")
    if "SALES ANALYSIS" in head_text_clean and "ALL PARTIES" in head_text_clean:
        return ("SALES_ANALYSIS_SUMMARY", "sales analysis all parties")
    if "COMPANY WISE CUSTOMER SALES" in head_text_clean:
        return ("SUMMARY", "company wise customer sales")
    if "CUSTOMER COMPANY AND PRODUCT SALES" in head_text_clean:
        return ("CUSTOMER_PRODUCT_SALES", "customer company product sales")
    if any(_compact_signature(kw) in compact for kw in ["CUSTOMER-WISE PRODUCT-WISE", "CUSTOMER WISE PRODUCT WISE", "CUSTOMER-PRODUCT WISE", "CUSTOMER PRODUCT WISE"]):
        return ("CUSTOMER_WISE_PRODUCT_WISE", "customer wise product wise")
    if any(_compact_signature(kw) in compact for kw in ["MFR/CUSTOMER WISE SALES", "CUSTOMER WISE SALES SUMMARY", "MFR / CUSTOMER WISE", "GROUP VS. CUSTOMER", "GROUP VS CUSTOMER"]):
        return ("MFR_CUSTOMER_SUMMARY", "manufacturer/customer summary")
    if "PARTY WISE SALES BOOK" in head_text_clean:
        return ("PARTY_SALES_BOOK", "party wise sales book")
    if any(kw in head_text_clean for kw in ["GST INVOICE", "TAX INVOICE", "INVOICE NO"]) or ("SGST" in head_text_clean and "CGST" in head_text_clean and "HSN" in head_text_clean):
        return ("INVOICE_SINGLE_PARTY", "invoice keywords")
    if "SALE REGISTER" in head_text_clean or "SALES REGISTER" in head_text_clean:
        if "QTY" in head_text_clean or "ITEM" in head_text_clean:
            return ("SALE_REGISTER_ITEMIZED", "sale register itemized")
        return ("SUMMARY", "sale register summary")
    if "PARTY WISE SALE" in head_text_clean or "PARTY WISE SALES" in head_text_clean:
        return ("SUMMARY", "party wise sales summary")
    if "AMOUNT N A M E" in head_text_clean or "AMOUNT NAME" in head_text_clean:
        return ("PARTY_SALES_BOOK", "amount-name party book")
    if "AREA ITEM" in head_text_clean or "AREAITEM" in compact:
        return ("PARTY_WISE", "area/item heading")
    if "ITEM PARTY" in head_text_clean or "ITEMPARTY" in compact:
        return ("ITEM_WISE", "item/party heading")
    if "PARTYPRODUCTWISE" in compact or "PARTY PRODUCTWISE" in head_text_clean:
        return ("PARTY_PRODUCT_WISE", "party product-wise heading")
    if "PARTY ITEM" in head_text_clean or "PARTYITEM" in compact:
        return ("PARTY_WISE", "party/item heading")
    if "ITEM WISE" in head_text_clean and "PARTY" not in head_text_clean and "AREA" not in head_text_clean:
        return ("ITEM_WISE", "item-wise heading")
    if "SALE SUMMARY" in head_text_clean or "AMT" in head_text_clean or any(kw in head_text_clean for kw in ["PAERIES", "SALE DATA", "RETAILER SALE", "SUMMARY", "DEALER"]):
        if "PARTY ITEM" not in head_text_clean and "PARTYITEM" not in compact and "AREA ITEM" not in head_text_clean:
            return ("SUMMARY", "summary keywords")

    return (report_type, reason)


def detect_report_format(lines: list) -> dict:
    spaced, compact, compact_lines = _header_views(lines, max_lines=30)

    def found(fmt: str, parser_mode: str, reason: str) -> dict:
        return {"Format": fmt, "ReportType": parser_mode, "Reason": reason}

    for config in _YAML_CONFIGS:
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
                if report_type == "SUMMARY" and any(k in spaced for k in ["MFR/CUSTOMER", "MFR CUSTOMER", "CUSTOMER WISE SALES SUMMARY"]):
                    report_type = "MFR_CUSTOMER_SUMMARY"
                return found(config.get("format_id", "UNKNOWN"), report_type, "YAML: " + config.get('format_id', 'UNKNOWN'))
        except Exception as e:
            logger.warning("Failed to parse YAML Config: %s", e)

    # ── Enterprise structural column-header matching ────────────────────────
    if _line_has_all(compact_lines, "SL", "ITEM DESCRIPTION", "PACK", "EXPIRY", "BATCH", "C/B", "QTY+FR", "MRP", "RATE", "GST", "AMOUNT", "HSNCODE") and \
       "INVOICENO" in compact and ("MS" in compact or "M/S" in spaced):
        return found("FORMAT_26_PAGED_STORE_INVOICE", "PAGED_STORE_INVOICE", "page-wise store invoice with item table")

    if _line_has_all(compact_lines, "DATE", "BILLNO", "ITEM", "BATCH NO", "QTY", "FREE", "SALE PR", "AMOUNT", "EXP.DATE"):
        return found("FORMAT_14", "SALE_REGISTER_ITEMIZED", "date/bill/item/batch sale-register columns")

    if _line_has_all(compact_lines, "BILL NO", "STORE NAME", "AMOUNT", "DISCOUNT", "NET AMT", "TAX PAYABLE", "DR/CR"):
        return found("FORMAT_08", "UNIVERSAL_TWO_COLUMN", "bill-wise single-store summary columns")

    if _line_has_all(compact_lines, "INVOICE NO", "INVOICE DATE", "STORENAME", "PLACE", "VALUE", "TOTAL"):
        return found("FORMAT_10", "SUMMARY", "invoice/store/place/value/total columns")

    if _line_has_all(compact_lines, "QTY", "UNITS", "NETT SALE AMT", "AVG PRICE", "NO OF VCH"):
        return found("FORMAT_09", "SUMMARY", "quantity units net-sale average-price voucher columns")

    if _line_has_all(compact_lines, "SL NO", "STORE NAME", "AREA") and _has_quarter_or_months(spaced):
        return found("FORMAT_17", "QUARTERLY_SUMMARY", "store/area quarterly heading")

    if _line_has_all(compact_lines, "CUSTOMER NAME", "NET AMOUNT", "GOODS VALUE", "GST AMOUNT", "PRODDISC"):
        return found("FORMAT_18", "SUMMARY", "customer net/goods/gst/discount columns")

    if _line_has_all(compact_lines, "GROUP/CUSTOMER", "QTY", "FREE", "GROSS VAL", "NET VALUE"):
        return found("FORMAT_15_OR_16", "MFR_CUSTOMER_SUMMARY", "group/customer gross and net value columns")

    if _line_has_all(compact_lines, "CUSTOMER", "STATION", "MOBILENO", "SALES VALUE"):
        return found("FORMAT_25", "SUMMARY", "customer/station/mobile/sales-value columns")

    if _line_has_all(compact_lines, "PRODUCT CODE", "PRODUCT NAME", "PACKING", "QTY", "VALUE"):
        return found("FORMAT_23_OR_24", "ITEM_WISE", "product code/name/packing/qty/value columns")

    if _line_has_all(compact_lines, "SL NO", "STORE NAME", "STORE LOCATION", "SALE-NET", "SAL-AMT"):
        return found("FORMAT_22", "SUMMARY", "store location sale-net sale-amount columns")

    if _line_has_all(compact_lines, "STORENAME", "STORE LOCATION", "SALES", "RETURN", "AMOUNT"):
        return found("FORMAT_12_OR_13", "SUMMARY", "store sales return amount columns")

    if _line_has_all(compact_lines, "STORE NAME", "STORE LOCATION", "QTY", "FREE", "AMOUNT", "FREE AMOUNT", "PACKS") and "BRANDDETAILS" in compact:
        return found("FORMAT_19_OR_20", "PARTY_PRODUCT_WISE", "store qty/free/amount/packs with brand details")

    if _line_store_month_total_order(compact_lines, total_before_month=True):
        return found("FORMAT_21", "QUARTERLY_SUMMARY", "store total before monthly amount columns")

    if _line_store_month_total_order(compact_lines, total_before_month=False):
        return found("FORMAT_07", "QUARTERLY_SUMMARY", "store monthly amount columns ending in total")

    if _line_has_all(compact_lines, "SERIAL NO", "PRODUCT CODE", "BRAND NAME", "QTY", "PRICE", "FREE UNITS", "AMOUNT", "DISCOUNT"):
        return found("FORMAT_03", "PARTY_PRODUCT_WISE", "serial/product/brand qty-price-free-amount-discount columns")

    if _line_has_all(compact_lines, "STORE NAME", "STORE LOCATION", "BRAND NAME", "QTY", "PRICE", "FREE UNITS", "AMOUNT", "DISCOUNT"):
        no_agency_hint = "NOTBETHERE" in compact or "WITHOUTAGENCY" in compact
        has_agency_hint = any(
            re.search(r'\b(AGENCY|AGENCIES|DISTRIBUTORS?|TRADERS?|PHARMA|MEDICALS?|GSTIN|DL\s*NO)\b', line, re.IGNORECASE)
            for line in lines[:10]
        ) and not no_agency_hint
        return found("FORMAT_01" if has_agency_hint else "FORMAT_02", "PARTY_PRODUCT_WISE", "store/location/brand qty-price-free-amount-discount columns")

    if _line_has_all(compact_lines, "STORE NAME", "QTY", "PRICE", "FREE UNITS", "AMOUNT", "DISCOUNT") and "BRANDNAME" in compact:
        return found("FORMAT_04", "PARTY_PRODUCT_WISE", "store/address plus qty-price-free-amount-discount with brand line")

    if _line_order(compact_lines, "AMOUNT", "STORE NAME"):
        return found("FORMAT_06", "PARTY_SALES_BOOK", "amount first then store/address")

    if _line_has_all(compact_lines, "STORE NAME", "AMOUNT") and not _has_all(compact, "QTY", "PRICE", "FREE UNITS"):
        return found("FORMAT_05", "SUMMARY", "store/address amount summary")

    if _line_has_all(compact_lines, "CODE", "STORENAME", "PLACE", "TOTAL"):
        return found("FORMAT_11", "SUMMARY", "code/store/place/total columns")

    if _line_has_all(compact_lines, "DATE", "BILL NO", "PARTY NAME", "BILL AMT"):
        return found("FORMAT_27_SALES_BOOK", "SALES_BOOK_REGISTER", "date/billno/partyname/billamt sales-book columns")
    if "SALESBOOK" in compact or "SALES BOOK" in spaced:
        if _has_all(compact, "BILLNO", "PARTYNAME") or _has_all(compact, "BILLAMT", "TAXABLE"):
            return found("FORMAT_27_SALES_BOOK", "SALES_BOOK_REGISTER", "sales-book with billamt/taxable columns")

    # ── MARG Party & Product Wise flat-table fingerprint ─────────────────────
    _marg_flat_drug_re = re.compile(
        r'\b(?:HEPP\s*FORT|ONECLAV|DEFENAC|REVEAL|MEGARICH|CIPROVA|'
        r'XIMECEF|SOLUBET|MULTIRICH|CANAZOLE|PILES\s*CURE|CEFFOREN)\w*', re.I
    )
    _marg_flat_hits = 0
    for _fl in lines[:20]:
        _fp = _fl.split('\t')
        if len(_fp) >= 8:
            _has_drug = bool(_marg_flat_drug_re.search(_fp[3] if len(_fp) > 3 else ''))
            _has_amt   = bool(re.search(r'\d{2,}\.\d{2}', _fl))
            if _has_drug and _has_amt:
                _marg_flat_hits += 1
    if _marg_flat_hits >= 3:
        return found("FORMAT_MARG_PPW_FLAT", "MARG_PARTY_PRODUCT_FLAT",
                     f"MARG flat-table data-pattern ({_marg_flat_hits}/20 rows matched)")

    # ── AI Format Inference Engine ───────────────────────────────────────────
    if _AI_AVAILABLE:
        _ai_lines = lines[:40]
        _ai_result = _ai_format_inference.infer(_ai_lines, "\n".join(lines))
        if _ai_result.confidence >= 0.65 and _ai_result.format_id not in ("UNKNOWN", "GENERIC_TABULAR") and not _ai_result.format_id.startswith("GENERIC_AUTO"):
            logger.info("AI Format Inference: %s / %s  conf=%.2f  tier=%d  source=%s",
                        _ai_result.erp_name, _ai_result.format_id,
                        _ai_result.confidence, _ai_result.tier, _ai_result.source)
            return found(
                f"AI_{_ai_result.format_id}",
                _ai_result.format_id,
                f"AI-inferred (tier={_ai_result.tier}, conf={_ai_result.confidence:.2f})",
            )

    legacy_mode, legacy_reason = _legacy_report_type_from_header(spaced, compact)
    return found("LEGACY_" + legacy_mode, legacy_mode, legacy_reason)

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
    raw_text = _heal_marg_csv_artifacts(raw_text)

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