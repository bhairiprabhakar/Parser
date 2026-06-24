import re
from config import _KW_PATTERN_REGEX, _AGENCY_PATTERN_REGEX
from transformers.entity_scrubber import parse_store_and_location
from transformers.item_parser import parse_item_description

try:
    from image_processing.confidence_scorer import ConfidenceScorer
    _confidence_scorer = ConfidenceScorer()
except ImportError:
    _confidence_scorer = None

_OCR_NOISE_RULES = [
    (re.compile(r'(?<=\d)O(?=\d)'),           '0',   ['Amount', 'Rate', 'Qty']),
    (re.compile(r'(?<=[A-Z])0(?=[A-Z])'),     'O',   ['Description']),
    (re.compile(r'\bI(?=\d{3,})'),            '1',   ['Amount', 'Rate']),
    (re.compile(r'(?<=\d)l(?=\d)'),           '1',   ['Amount', 'Rate']),
    (re.compile(r'\bS(?=\d{3,})'),            '5',   ['Amount']),
    (re.compile(r'\bLUPISULIDE\s*-\s*P\b', re.I),  'LUPISULIDE-P',  ['Description']),
    (re.compile(r'\bHEPP\s+FORT\b',         re.I),  'HEPP FORT',     ['Description']),
    (re.compile(r'\bONECLAV\s+625\b',       re.I),  'ONECLAV 625',   ['Description']),
    (re.compile(r'\bPANTOLUP\s+DSR\b',      re.I),  'PANTOLUP DSR',  ['Description']),
    (re.compile(r'\bLUPICEF[\s\-]*O\b',     re.I),  'LUPICEF-O',     ['Description']),
    (re.compile(r'\bXIMECEF+\b',            re.I),  'XIMECEFF',      ['Description']),
    (re.compile(r'\bMEDICAL\s+L\b',         re.I),  'MEDICALS',      ['StoreName']),
    (re.compile(r'\bMEDICLS\b',             re.I),  'MEDICALS',      ['StoreName']),
    (re.compile(r'\bDRUGSTORE\b',           re.I),  'DRUG STORE',    ['StoreName']),
    (re.compile(r'\bMEDlCAL\b',             re.I),  'MEDICAL',       ['StoreName']),
    (re.compile(r'(\d+\.\d{2})\s*[A-Z]{2,}$'),  r'\1', ['Amount']),
]

_KNOWN_DRUGS = {
    'LUPISULIDE P TAB':   'LUPISULIDE-P TAB(AMB)',
    'LUPISULIDE P GE':    'LUPISULIDE-P(GE)',
    'LUPISULIDE P GF':    'LUPISULIDE-P(GF)',
    'HEPP FORT SYP':      'HEPP FORT SYP',
    'LUPICEF 0 200MG TAB':'LUPICEF-O 200MG TAB',
    'XIMECEFF O 200 TAB': 'XIMECEFF O 200 TAB.',
    'ONECLAV 625 TAB':    'ONECLAV 625 TAB.',
    'PANTOLUP DSR CAP':   'PANTOLUP DSR CAP',
    'AZILUP 250 TAB':     'AZILUP 250 TAB',
    'AZILUP 500 TAB':     'AZILUP 500 TAB.',
    'REVEAL KIT':         'REVEAL KIT',
    'MEGARICH CAP':       'MEGARICH CAP',
    'BILALUP M TAB':      'BILALUP M TAB.',
    'CEFPOLUP 100 DRY SYP':'CEFPOLUP 100 DRY SYP',
    'CEFPOLUP 200 TAB':   'CEFPOLUP 200 TAB.',
}

# ══════════════════════════════════════════════════════════════════════════════
#  STRICT ENTITY CLEANING WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_strict_entity_cleaning(text: str, is_location: bool = False) -> str:
    if not text:
        return ""
    
    cleaned = str(text).upper()
    cleaned = re.sub(r'\b\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}\b', ' ', cleaned)
    
    if is_location:
        # 🚀 RC3 FIX: Keep digits and hyphens — needed for F-3, SEC 45
        cleaned = re.sub(r'[^A-Z0-9\-\s]', ' ', cleaned)   
        cleaned = re.sub(r'-{2,}', '-', cleaned)             
    else:
        cleaned = re.sub(r'[^A-Z0-9&\-\s]', ' ', cleaned)
        cleaned = re.sub(r'-{2,}', '-', cleaned)
        cleaned = cleaned.strip('- ')

    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned

def safe_clean_store_entities(data: dict) -> dict:
    if not data or "Areas" not in data:
        return data
        
    for area in data.get("Areas", []):
        for store in area.get("Stores", []):
            if "StoreName" in store:
                store["StoreName"] = _apply_strict_entity_cleaning(store["StoreName"], is_location=False)
            if "StoreLocation" in store:
                store["StoreLocation"] = _apply_strict_entity_cleaning(store["StoreLocation"], is_location=True)
                
    return data

# ══════════════════════════════════════════════════════════════════════════════
#  QA FLAGGING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_qa_flags(item: dict, store: dict = None) -> list:
    flags = []
    desc = item.get('Description', '')
    amount = item.get('Amount', 0)
    if not desc or not desc.strip():
        flags.append('MISSING_DESCRIPTION')
    if _confidence_scorer:
        score = _confidence_scorer.score_text(desc)
        if score < 0.3:
            flags.append('OCR_VERY_LOW_CONFIDENCE')
        elif score < 0.5:
            flags.append('OCR_LOW_CONFIDENCE')
    try:
        amt = float(str(amount).replace(',', ''))
        if amt < 0:
            flags.append('NEGATIVE_AMOUNT')
        elif amt == 0:
            flags.append('ZERO_AMOUNT')
    except (ValueError, TypeError):
        flags.append('INVALID_AMOUNT')
    qty = item.get('Qty', 0)
    try:
        q = int(float(str(qty).replace(',', '')))
        if q <= 0:
            flags.append('ZERO_QTY')
    except (ValueError, TypeError):
        pass
    rate = item.get('Rate', 0)
    try:
        r = float(str(rate).replace(',', ''))
        if r > 99999:
            flags.append('UNUSUALLY_HIGH_RATE')
    except (ValueError, TypeError):
        pass
    if store:
        s_name = store.get('StoreName', '')
        if s_name and not re.search(r'[A-Za-z]{3,}', s_name):
            flags.append('STORE_NAME_GARBLED')
    return flags

# ══════════════════════════════════════════════════════════════════════════════
#  OCR NOISE CORRECTION (4th pass)
# ══════════════════════════════════════════════════════════════════════════════

def apply_ocr_noise_correction(value: str, field: str) -> str:
    if not value or not isinstance(value, str):
        return value
    for pattern, replacement, fields in _OCR_NOISE_RULES:
        if field in fields:
            try:
                value = pattern.sub(replacement, value)
            except Exception:
                pass
    if field == 'Description':
        normalised = re.sub(r'[^A-Z0-9 ]', ' ', value.upper())
        normalised = re.sub(r'\s+', ' ', normalised).strip()
        if normalised in _KNOWN_DRUGS:
            value = _KNOWN_DRUGS[normalised]
    return value.strip()

# ══════════════════════════════════════════════════════════════════════════════
#  FLAG SUSPICIOUS ITEMS (4th pass)
# ══════════════════════════════════════════════════════════════════════════════

def flag_suspicious_items(items: list) -> list:
    flagged = []
    for item in items:
        flags = []
        amt  = float(item.get('Amount', 0) or 0)
        rate = float(item.get('Rate',   0) or 0)
        qty  = int(item.get('Qty',      0) or 0)
        free = int(item.get('Free',     0) or 0)
        desc = str(item.get('Description', '') or '')

        if amt <= 0:
            flags.append('ZERO_AMOUNT')
        if rate > 0 and amt > 0 and rate > amt:
            flags.append('RATE_EXCEEDS_AMOUNT')
        if len(desc.strip()) < 4:
            flags.append('SHORT_DESC')
        if amt > 500000:
            flags.append('UNUSUALLY_LARGE_AMOUNT')
        if qty == 0 and free == 0 and amt > 0:
            flags.append('QTY_MISSING')

        item['_qa_flags'] = flags
        flagged.append(item)
    return flagged

# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING & REPARENTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def post_process_extracted_data(data: dict, run_qa: bool = True) -> dict:
    for area in data.get("Areas", []):
        reparented_stores = []
        for store in area.get("Stores", []):
            current_active_store = {
                "StoreName": store.get("StoreName", ""),
                "StoreLocation": store.get("StoreLocation", ""),
                "Items": []
            }
            
            for item in store.get("Items", []):
                desc = item.get("Description", "")
                if desc:
                    desc = re.sub(r'(?i)DATE\s+BILLNO\s+ITEM.*?EXP\.?DATE\s*(D\s+)?', '', desc).strip()
                    store_split_match = re.search(r'^(.+?)\s+D\s+([A-Za-z0-9\-].+)$', desc)
                    
                    if store_split_match:
                        potential_store = store_split_match.group(1).strip()
                        potential_item = store_split_match.group(2).strip()
                        
                        is_valid_store = False
                        if len(potential_store) > 8:
                            if ',' in potential_store:
                                is_valid_store = True
                            else:
                                if bool(re.search(_KW_PATTERN_REGEX, potential_store.upper())):
                                    is_valid_store = True
                                    
                        if is_valid_store and not re.search(r'\b\d{1,4}\s*(ML|TAB|CAP|GM|INJ|SYP)\b', potential_store, re.IGNORECASE):
                            if current_active_store["Items"]:
                                reparented_stores.append(current_active_store)
                                
                            s_name, s_loc = parse_store_and_location(potential_store)
                            if not s_name or s_name == potential_store:
                                if ',' in potential_store:
                                    parts = potential_store.split(',', 1)
                                    s_name = parts[0].strip()
                                    s_loc = parts[1].strip()
                                else:
                                    s_name = potential_store
                                    s_loc = ""
                            
                            current_active_store = {
                                "StoreName": s_name,
                                "StoreLocation": s_loc,
                                "Items": []
                            }
                            desc = potential_item 
                    
                    desc = re.sub(r'^D\s+', '', desc)
                    item["Description"] = desc.strip()
                    
                current_active_store["Items"].append(item)
            
            if current_active_store["Items"] or not reparented_stores:
                reparented_stores.append(current_active_store)
                
        area["Stores"] = reparented_stores

    for area in data.get("Areas", []):
        for store in area.get("Stores", []):
            
            # 🚀 RC4 FIX: Prevent double location splitting if it already has one!
            s_name, s_loc = store.get("StoreName", ""), store.get("StoreLocation", "")
            if s_loc:
                pass # Already split by schema engine
            else:
                s_name, s_loc = parse_store_and_location(s_name)
                store["StoreName"] = s_name
                store["StoreLocation"] = s_loc

            if s_name:
                s_name = re.sub(r'^[A-Z0-9]+\s*-\s*', '', s_name)
                s_name = re.sub(r'[\(\[\{<].*?[\)\]\}>]', ' ', s_name)
                chop_triggers = [r'\bDL\.?\s*NO', r'\bGST', r'\bTIN', r'\bMOB', r'\bPH:', r'\bINV\b']
                for trigger in chop_triggers:
                    s_name = re.split(trigger, s_name, flags=re.IGNORECASE)[0]
                s_name = re.sub(r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b', ' ', s_name)
                s_name = re.sub(r'(?i)\bDATE\b', ' ', s_name)
                store["StoreName"] = re.sub(r'\s{2,}', ' ', s_name).strip()

            s_loc = store.get("StoreLocation", "")
            if s_loc:
                s_loc = re.sub(r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b', ' ', s_loc)
                s_loc = re.sub(_KW_PATTERN_REGEX, '', s_loc, flags=re.IGNORECASE)
                s_loc = re.sub(_AGENCY_PATTERN_REGEX, '', s_loc, flags=re.IGNORECASE)
                store["StoreLocation"] = re.sub(r'\s{2,}', ' ', s_loc).strip()

            for item in store.get("Items", []):
                desc = item.get("Description", "")
                if desc:
                    desc = re.sub(r'^\s*\d{1,4}[\.\)\-]+\s+', '', desc)
                    if re.match(r'^\s*\d{1,3}\s+[A-Za-z]', desc) and not re.match(r'^\s*(24|3D|5D|A2Z|B12|A\s*2\s*Z)\b', desc, re.IGNORECASE):
                        desc = re.sub(r'^\s*\d{1,3}\s+', '', desc)

                    desc = re.sub(r'[*.\-=,]{2,}', ' ', desc)
                    desc = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', ' ', desc)
                    
                    desc = re.sub(r'\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*\'?-?\s*\d{2,4}\b', ' ', desc, flags=re.IGNORECASE)
                    desc = re.sub(r'\b\d{1,2}\s*%\b', ' ', desc)

                    desc = re.sub(r'\b(?!\d+\s*[xX\*]\s*\d+[A-Z]*\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d{3})[A-Z0-9]{6,15}\b', ' ', desc)
                    desc = re.sub(r'\b\d{4,6}[A-Z]{1,3}(?:\s+[A-Z]{1,3})?\b', ' ', desc)
                    desc = re.sub(r'\b[A-Za-z]{0,4}\d{4,8}\b', ' ', desc)
                    
                    desc = re.sub(r'^\s*[A-Z]\s+(?=[A-Za-z]{3,})', '', desc)
                    desc = re.sub(r'^\s*[A-Za-z]\s*\d{3,6}\b', ' ', desc)
                    desc = re.sub(r'\b[A-Z]\s*\d{3,6}\s*$', ' ', desc)
                    desc = re.sub(_AGENCY_PATTERN_REGEX, '', desc, flags=re.IGNORECASE)
                    desc = re.sub(r'\s{2,}', ' ', desc).strip()
                    
                    parts = desc.split()
                    if len(parts) >= 2:
                        t1 = parts[0]
                        if 3 <= len(t1) <= 7 and any(c.isdigit() for c in t1) and any(c.isalpha() for c in t1):
                            if t1 not in ['A2Z', 'B12', 'Q10', 'Z21']:
                                desc = desc[len(t1):].strip()
                        elif t1.isalpha() and 2 <= len(t1) <= 5 and len(parts) > 1:
                            if parts[1].startswith(t1):
                                desc = desc[len(t1):].strip()
                    
                    desc = re.sub(r'([A-Za-z]{3,})(\d{1,2}[/\-\.\|]\d{2,4})\b', r'\1 \2', desc)
                    desc = re.sub(r'\b(\d{1,2}[/\-\.\|]\d{2,4})([A-Za-z]{3,})', r'\1 \2', desc)
                    desc = re.sub(r'([A-Za-z]{3,})(\d+\s*[xX\*]\s*\d+)\b', r'\1 \2', desc)
                    desc = re.sub(r'[\(\[\{<].*?[\)\]\}>]', ' ', desc)
                    desc = re.sub(r'\s{2,}', ' ', desc).strip()
                    
                    parsed = parse_item_description(desc)
                    item["Brand_Name"] = parsed["Brand_Name"]
                    item["Dosage"] = parsed["Dosage"]
                    item["Packaging"] = parsed["Packaging"]
                    item["Description"] = parsed["Original_OCR"] 

    # ── Enterprise 4-pass pipeline ──
    # Pass 1: clean_store_entities (already done above)
    # Pass 2: parse_items (already done above via parse_item_description)
    for area in data.get("Areas", []):
        for store in area.get("Stores", []):
            for item in store.get("Items", []):
                item["Description"] = apply_ocr_noise_correction(
                    item.get("Description", ""), "Description")
            store["StoreName"] = apply_ocr_noise_correction(
                store.get("StoreName", ""), "StoreName")
            store["Items"] = flag_suspicious_items(store.get("Items", []))

    # Compute per-document extraction quality metrics
    all_items = [i for area in data.get("Areas", [])
                 for store in area.get("Stores", [])
                 for i in store.get("Items", [])]

    total_rows    = len(all_items)
    flagged_rows  = sum(1 for i in all_items if i.get("_qa_flags"))
    total_amt     = sum(float(i.get("Amount", 0) or 0) for i in all_items)
    grand_total   = float(data.get("_grand_total", 0) or 0)
    coverage_pct  = (total_amt / grand_total * 100) if grand_total > 0 else 100.0

    data["_extraction_quality"] = {
        "total_rows":        total_rows,
        "flagged_rows":      flagged_rows,
        "clean_rows":        total_rows - flagged_rows,
        "total_amount":      round(total_amt, 2),
        "grand_total":       round(grand_total, 2),
        "coverage_pct":      round(coverage_pct, 2),
        "qa_pass":           flagged_rows == 0 and coverage_pct >= 85.0,
        "qa_summary":        (
            f"{total_rows} rows | {flagged_rows} flagged | "
            f"coverage {coverage_pct:.1f}%"
        ),
    }
    return data