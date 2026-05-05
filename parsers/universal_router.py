import os
import re
import yaml
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ── Absolute Import for Pipeline Transformers ──
from transformers.cleaners import preprocess_line

# ══════════════════════════════════════════════════════════════════════════════
#  SIGNATURE DETECTION UTILITIES 
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
#  THE YAML ROUTER ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def detect_report_format(lines: list) -> dict:
    spaced, compact, compact_lines = _header_views(lines, max_lines=30)

    def found(fmt: str, parser_mode: str, reason: str) -> dict:
        return {"Format": fmt, "ReportType": parser_mode, "Reason": reason}

    # ── 1. Read YAML Configurations ──
    formats_dir = Path(__file__).parent.parent / "formats"
    if formats_dir.exists():
        for yaml_file in formats_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
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
                    return found(fmt_id, config["report_type"], "YAML: " + yaml_file.name)
                    
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
                        return found(config["format_id"], config["report_type"], "YAML alt-layout: " + yaml_file.name)

                if has_all and has_any and condition_met:
                    return found(config["format_id"], config["report_type"], "YAML: " + yaml_file.name)
            except Exception as e:
                log.warning("Failed to parse YAML %s: %s", yaml_file.name, e)

    # ── 2. Fallback to Legacy Headers (VIP PRIORITY RESTORED) ──
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

def route_and_parse(raw_text: str, source_file: str = "") -> dict:
    """
    Priority 1: The Battle-Tested Legacy Machine & YAML Rules
    Priority 2: The Generic Schema Engine (ONLY if Priority 1 fails)
    """
    from parsers.legacy_machine import parse_text
    data = parse_text(raw_text)
    
    # ── Check if your legacy rules successfully found items ──
    item_count = sum(len(s.get("Items", [])) for a in data.get("Areas", []) for s in a.get("Stores", []))
    
    if item_count > 0:
        return data
        
    # ── If legacy completely failed (0 items), ONLY THEN try the Schema Engine ──
    log.warning("Legacy rules found 0 items. Falling back to Generic Schema Engine...")
    try:
        from parsers.schema_engine import parse_by_schema_inference
        schema_data = parse_by_schema_inference(raw_text, source_file=source_file)
        
        schema_item_count = sum(len(s.get("Items", [])) for a in schema_data.get("Areas", []) for s in a.get("Stores", []))
        if schema_item_count > 0:
            return schema_data
            
    except Exception as e:
        log.warning("Schema engine encountered an error (%s).", e)

    return data