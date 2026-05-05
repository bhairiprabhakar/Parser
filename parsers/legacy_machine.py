import re
import logging
from config import _SKIP_HEADERS, _KW_PATTERN_REGEX, _ROOTS

log = logging.getLogger(__name__)

from transformers.cleaners import preprocess_line, clean_number, is_numeric_token
from transformers.entity_scrubber import parse_store_and_location

# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_empty_data() -> dict:
    return {
        "AgencyDetails": {"Name": "", "Address": "", "Phone": "", "Email": "", "GSTIN": "", "TIN": ""},
        "ReportDetails": {"FromDate": "", "ToDate": "", "Company": "", "DetectedFormat": "", "ParserMode": ""},
        "Areas": [],
    }

def _new_area(name: str) -> dict: 
    return {"AreaName": name, "Stores": []}

def _new_store(name: str, location: str = "") -> dict: 
    return {"StoreName": name, "StoreLocation": location, "Items": []}

def _new_item(desc: str, nums: list, has_date: bool = False) -> dict:
    qty, free, rate, amt, perc = '0', '0', '0', '0', '0'
    if len(nums) == 1: amt = nums[0]
    elif len(nums) == 2: qty, amt = nums[0], nums[1]
    elif len(nums) == 3: qty, rate, amt = nums[0], nums[1], nums[2]
    elif len(nums) >= 4:
        if has_date:
            if len(nums) == 4: qty, free, rate, amt = nums[0], nums[1], nums[2], nums[3]
            else: qty, free, rate, amt, perc = nums[0], nums[1], nums[2], nums[3], nums[4]
        else:
            qty = nums[0]
            if len(nums) == 4:
                n_qty, n_2, n_3 = clean_number(nums[0]), clean_number(nums[2]), clean_number(nums[3])
                is_excel_format = False
                if n_qty > 0 and n_2 > 0 and abs((n_qty * n_2) - n_3) <= (n_qty * n_2 * 0.3): is_excel_format = True
                elif nums[1] in ('-', '0', '0.0', '0.00') and n_3 > n_2: is_excel_format = True
                if is_excel_format: free, rate, amt = nums[1], nums[2], nums[3]
                else: rate, amt, perc = nums[1], nums[2], nums[3]
            else:
                free, rate, amt, perc = nums[1], nums[2], nums[3], nums[4]
    return {
        "Description": desc, "Qty": clean_number(qty, is_int=True), "Free": clean_number(free, is_int=True),
        "Rate": clean_number(rate), "Amount": clean_number(amt), "Percent": clean_number(perc),
    }

def _extract_float_after(label: str, text: str):
    m = re.search(label + r'\s*:?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text, re.IGNORECASE)
    if not m: return None
    return clean_number(m.group(1))

# ══════════════════════════════════════════════════════════════════════════════
#  DEDICATED FORMAT PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_paged_invoice_item(line_str: str):
    row = re.match(
        r'^\s*(?P<sl>\d+)\s+(?P<body>.*?)\s+(?P<case_qty>\d+)\s+(?P<case_unit>[A-Z/]+)\s+'
        r'(?P<qty>\d+(?:\.\d+)?)\s+(?P<mrp>\d+(?:\.\d+)?)\s+(?P<rate>\d+(?:\.\d+)?)\s+'
        r'(?:(?P<disc>\d+(?:\.\d+)?)\s+)?(?P<gst>\d+(?:\.\d+)?)%\s+(?P<amount>\d+(?:\.\d+)?)\s+(?P<hsn>\d+)\s*$',
        line_str, re.IGNORECASE
    )
    if not row: return None
    body = re.sub(r'\s{2,}', ' ', row.group("body")).strip()
    detail = re.match(r'(?P<desc>.+?)\s+(?P<pack>\d+\s*[xX]\s*[A-Z0-9]+|\d+\s*(?:ML|GM|G|MG|TAB|CAP|PCS|M))\s+(?P<expiry>[A-Za-z]{3}\'?\d{2,4})\s+(?P<batch>\S+)\s*$', body, re.IGNORECASE)
    if detail:
        desc, pack, expiry, batch = detail.group("desc").strip(), detail.group("pack").upper().replace(" ", ""), detail.group("expiry"), detail.group("batch")
    else:
        desc, pack, expiry, batch = body, "", "", ""
    desc = re.sub(r'\([^)]*\)', ' ', desc)
    desc = re.sub(r'\([^)]*$', ' ', desc)
    desc = re.sub(r'\s{2,}', ' ', desc).strip()
    if pack: desc = f"{desc} {pack}".strip()
    return {
        "Description": desc, "Qty": clean_number(row.group("qty"), is_int=True), "Free": 0, "Rate": clean_number(row.group("rate")),
        "Amount": clean_number(row.group("amount")), "Percent": clean_number(row.group("disc") or "0"), "MRP": clean_number(row.group("mrp")),
        "GSTPercent": clean_number(row.group("gst")), "HSNCode": row.group("hsn"), "Batch": batch, "Expiry": expiry,
        "CaseQty": clean_number(row.group("case_qty"), is_int=True), "CaseUnit": row.group("case_unit").upper()
    }

def parse_paged_store_invoice(raw_text: str, format_meta: dict = None) -> dict:
    data = _make_empty_data()
    area = _new_area("UNKNOWN AREA")
    data["Areas"].append(area)
    rd = data["ReportDetails"]
    rd["DetectedFormat"] = format_meta.get("Format", "FORMAT_26_PAGED_STORE_INVOICE") if format_meta else "FORMAT_26_PAGED_STORE_INVOICE"
    rd["ParserMode"] = format_meta.get("ReportType", "PAGED_STORE_INVOICE") if format_meta else "PAGED_STORE_INVOICE"
    raw_text = re.sub(r'\(cid:\s*12\)', '\n', raw_text)
    pages = [p for p in re.split(r'\x0c+|(?=\n?GST\s+NO\s*:)', raw_text) if p.strip()]
    all_dates, total_balance_sum, subtotal_sum, saw_company_marker = [], 0.0, 0.0, False

    for page in pages:
        lines = [preprocess_line(l) for l in page.split('\n') if l and not set(l.strip()) <= {"-"}]
        if not lines: continue
        page_text = "\n".join(lines)
        if re.search(r'(?m)^\s*RELIABO\s*$', page_text, re.IGNORECASE): saw_company_marker = True
        ag = data["AgencyDetails"]
        if not ag["GSTIN"]:
            gm = re.search(r'GST\s+NO\s*:\s*([A-Z0-9]+)', page_text, re.IGNORECASE)
            if gm: ag["GSTIN"] = gm.group(1).strip()
        if not ag["Phone"]:
            pm = re.search(r'MOBILE\s*:\s*([0-9,\s]+)', page_text, re.IGNORECASE)
            if pm: ag["Phone"] = pm.group(1).strip()
        if not ag["TIN"]:
            dm = re.search(r'D\.?L\.?\s*NO\s*:\s*([A-Z0-9/\-]+)', page_text, re.IGNORECASE)
            if dm: ag["TIN"] = dm.group(1).strip()
        if not ag["Address"]:
            address_parts = [l.strip() for l in lines[:6] if not any(k in l.upper() for k in ["GST NO", "D.L. NO", "FOODLIC", "M/S", "INVOICE", "DATE", "NO OF CASES", "TERMS", "SALES MAN"]) and re.search(r'[A-Za-z]{3,}', l)]
            ag["Address"] = ", ".join(address_parts[:2])

        store_name, store_loc = "", ""
        m_store = re.search(r'M/S\s+(.+?)\s+INVOICE\s+NO\.?\s*:', page_text, re.IGNORECASE)
        if m_store: store_name = m_store.group(1).strip()
        m_date = re.search(r'DATE\s*:\s*(\d{1,2}/\d{1,2}/\d{2,4})', page_text, re.IGNORECASE)
        if m_date: all_dates.append(m_date.group(1))

        for idx, l in enumerate(lines):
            if "DATE" in l.upper() and "NO OF CASES" in l.upper():
                lm = re.match(r'\s*([A-Za-z][A-Za-z\s.\-]{2,}?)\s+DATE\s*:', l, re.IGNORECASE)
                if lm: store_loc = lm.group(1).strip()
                break
            if m_store and idx > 0 and "DATE" in l.upper():
                before_date = re.split(r'\bDATE\s*:', l, flags=re.IGNORECASE)[0].strip()
                if before_date: store_loc = before_date
                break

        if not store_name: continue
        s_name, parsed_loc = parse_store_and_location(store_name)
        store = _new_store(s_name, store_loc or parsed_loc)
        area["Stores"].append(store)

        for l in lines:
            if re.match(r'^\s*RELIABO\s*$', l, re.IGNORECASE): continue
            item = _parse_paged_invoice_item(l)
            if item: store["Items"].append(item)

        add_gst = _extract_float_after(r'ADD\s+GST', page_text)
        if add_gst: store["Items"].append({"Description": "SGST CGST ADD GST", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": add_gst, "Percent": 0.0})
        round_off = _extract_float_after(r'Round\s+off', page_text)
        if round_off: store["Items"].append({"Description": "Round off", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": round_off, "Percent": 0.0})
        total_balance = _extract_float_after(r'Total\s+Balance', page_text)
        if total_balance: total_balance_sum += total_balance
        subtotal = _extract_float_after(r'Sub\s+total', page_text)
        if subtotal: subtotal_sum += subtotal

    if saw_company_marker:
        data["AgencyDetails"]["Name"] = data["AgencyDetails"]["Name"] or "RELIABO"
        rd["Company"] = rd["Company"] or "RELIABO"

    if all_dates:
        def date_key(d_str):
            parts = re.split(r'[-/]', d_str)
            if len(parts[-1]) == 2: parts[-1] = "20" + parts[-1]
            return int(parts[-1]) * 10000 + int(parts[1]) * 100 + int(parts[0])
        try:
            ordered = sorted(all_dates, key=date_key)
            rd["FromDate"], rd["ToDate"] = ordered[0], ordered[-1]
        except Exception:
            rd["FromDate"], rd["ToDate"] = all_dates[0], all_dates[-1]

    if subtotal_sum > 0: data["_grand_total"] = round(subtotal_sum, 2)
    elif total_balance_sum > 0: data["_grand_total"] = round(total_balance_sum, 2)
    return data

def parse_sales_book_register(raw_text: str, format_meta: dict = None) -> dict:
    data = _make_empty_data()
    rd, ag = data["ReportDetails"], data["AgencyDetails"]
    rd["DetectedFormat"] = format_meta.get("Format", "FORMAT_27_SALES_BOOK") if format_meta else "FORMAT_27_SALES_BOOK"
    rd["ParserMode"] = "SALES_BOOK_REGISTER"
    raw_text = raw_text.replace('\x0c', '\n')
    raw_text = re.sub(r'\(cid:\s*12\)', '\n', raw_text)
    lines = raw_text.split('\n')

    header_done = False
    for raw_line in lines:
        line = preprocess_line(raw_line)
        if not line or line.startswith('---'): continue
        lower = line.lower()
        cleaned = re.sub(r'\s+', '', line).upper()
        if ('PARTYNAME' in cleaned and 'BILLNO' in cleaned) or ('DATEbillno' in cleaned or ('DATE' in cleaned and 'BILLAMT' in cleaned)):
            header_done = True
            break
        _parse_header_line(line, data)

        if not ag["Name"]:
            valid_suffixes = ['AGENCY', 'AGENCIES', 'COMPANY', 'TRADERS', 'TRADING', 'DISTRIBUTORS', 'DISTRIBUTOR', 'ENTERPRISES', 'ENTERPRISE', 'CORPORATION', 'CORP', 'PHARMA', 'MEDICAL', 'MEDICALS', 'MEDICOS', 'MEDICOSE', 'DRUG', 'CHEMIST', 'SUPPLIERS', 'SUPPLY', 'M/S']
            if any(kw in line.upper() for kw in valid_suffixes):
                ag_name = re.sub(r'(?i)^M/S\.?\s*', '', line).strip()
                ag_name = re.sub(r'\*+', '', ag_name)
                ag_name = re.sub(r'\b20\d{2}[-/]\d{2,4}\b', '', ag_name).strip(" -:,")
                if re.search(r'[A-Za-z]{3,}', ag_name): ag["Name"] = ag_name

        if not ag["GSTIN"] and "gstin" in lower:
            gm = re.search(r'GSTIN\s*:\s*([A-Z0-9]+)', line, re.IGNORECASE)
            if gm: ag["GSTIN"] = gm.group(1).strip()
        if "phone" in lower or "e-mail" in lower:
            pm = re.search(r'Phone\s*:\s*([0-9,]+)', line, re.IGNORECASE)
            em = re.search(r'E-Mail\s*:\s*([\w.\-]+@[\w.\-]+)', line, re.IGNORECASE)
            if pm: ag["Phone"] = pm.group(1).strip()
            if em: ag["Email"] = em.group(1).strip()

    for raw_line in lines[:30]:
        line = preprocess_line(raw_line)
        dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', line)
        if len(dates) >= 2:
            rd["FromDate"], rd["ToDate"] = dates[0], dates[1]
            break
        elif len(dates) == 1 and rd["FromDate"] == "":
            rd["FromDate"] = dates[0]

    area = _new_area("UNKNOWN AREA")
    data["Areas"].append(area)

    DATA_ROW = re.compile(
        r'^(?P<date>\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+(?P<bill_no>[A-Z]{2}\d{5,})\s+(?P<party>.+?)\s+'
        r'(?P<bill_amt>-?\d[\d,]*\.\d{2})(?:\s+(?P<taxable>-?\d[\d,]*\.\d{2}))?(?:\s+(?P<tax>-?\d[\d,]*\.\d{2}))?'
        r'(?:\s+(?P<sur_tax>-?\d[\d,]*\.\d{2}))?(?:\s+(?P<free_amt>-?\d[\d,]*\.\d{2}))?(?:\s+(?P<exempted>-?\d[\d,]*\.\d{2}))?(?:\s+(?P<round_off>-?\d[\d,]*\.\d{2}))?\s*$',
        re.IGNORECASE
    )

    in_table, all_dates_seen = False, []
    for raw_line in lines:
        line = preprocess_line(raw_line)
        if not line or line.startswith('---'): continue

        if not in_table:
            cleaned = re.sub(r'\s+', '', line).upper()
            if ('PARTYNAME' in cleaned and 'BILLNO' in cleaned) or ('BILLAMT' in cleaned and 'TAXABLE' in cleaned):
                in_table = True
            continue

        lower = line.lower()
        if any(kw in lower for kw in ['date', 'bill no', 'party name', 'total :', 'grand total', 'end of report', 'taxable', 'sur. tax', 'r.off']):
            if not re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', line): continue

        m = DATA_ROW.match(line.strip())
        if not m:
            if re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', line):
                tokens = line.split()
                decimals = [t for t in tokens if re.match(r'^-?\d[\d,]*\.\d{2}$', t)]
                if decimals:
                    bill_amt_val = clean_number(decimals[0])
                    try:
                        first_dec_idx = next(i for i, t in enumerate(tokens) if re.match(r'^-?\d[\d,]*\.\d{2}$', t))
                        party_tokens = tokens[2:first_dec_idx]
                        party_name = " ".join(party_tokens).strip()
                    except StopIteration: continue
                    if not party_name: continue
                    party_name_clean = re.sub(r'\s{2,}', ' ', party_name).strip().upper()
                    store = _new_store(party_name_clean, "")
                    area["Stores"].append(store)
                    store["Items"].append({
                        "Description": "BILL AMOUNT", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": bill_amt_val, "Percent": 0.0,
                        "Bill_Date": tokens[0], "Bill_No": tokens[1] if len(tokens) > 1 else "",
                        "Taxable": clean_number(decimals[1]) if len(decimals) > 1 else 0.0,
                        "Tax": clean_number(decimals[2]) if len(decimals) > 2 else 0.0,
                        "Sur_Tax": clean_number(decimals[3]) if len(decimals) > 3 else 0.0,
                        "Free_Amt": clean_number(decimals[4]) if len(decimals) > 4 else 0.0,
                        "Exempted": clean_number(decimals[5]) if len(decimals) > 5 else 0.0,
                        "Round_Off": clean_number(decimals[6]) if len(decimals) > 6 else 0.0,
                    })
                    all_dates_seen.append(tokens[0])
            continue

        date_val, party_name, bill_amt = m.group("date"), m.group("party").strip(), clean_number(m.group("bill_amt"))
        all_dates_seen.append(date_val)
        party_name = re.sub(r'\s{2,}', ' ', re.sub(r'[^A-Z0-9&\-\s\.\(\)]', ' ', party_name.upper())).strip()
        if not party_name or len(party_name) < 2: continue

        store = _new_store(party_name, "")
        area["Stores"].append(store)
        store["Items"].append({
            "Description": "BILL AMOUNT", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": bill_amt, "Percent": 0.0,
            "Bill_Date": date_val, "Bill_No": m.group("bill_no").strip(),
            "Taxable": clean_number(m.group("taxable")) if m.group("taxable") else 0.0,
            "Tax": clean_number(m.group("tax")) if m.group("tax") else 0.0,
            "Sur_Tax": clean_number(m.group("sur_tax")) if m.group("sur_tax") else 0.0,
            "Free_Amt": clean_number(m.group("free_amt")) if m.group("free_amt") else 0.0,
            "Exempted": clean_number(m.group("exempted")) if m.group("exempted") else 0.0,
            "Round_Off": clean_number(m.group("round_off")) if m.group("round_off") else 0.0,
        })

    if all_dates_seen and not rd["FromDate"]:
        def _dk(d):
            p = re.split(r'[-/]', d)
            if len(p[-1]) == 2: p[-1] = "20" + p[-1]
            try: return int(p[-1])*10000 + int(p[1])*100 + int(p[0])
            except: return 0
        try:
            ordered = sorted(all_dates_seen, key=_dk)
            rd["FromDate"], rd["ToDate"] = ordered[0], ordered[-1]
        except Exception: pass

    return data


def _parse_header_line(line_str: str, data: dict) -> None:
    ag, rd = data["AgencyDetails"], data["ReportDetails"]
    upper_line = line_str.upper()
    if "PAN " in upper_line and "GST" in upper_line: return 
    if "DL NO" in upper_line and not ag["Name"]: return 
    
    extracted_dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', line_str)
    if extracted_dates and not rd["FromDate"]:
        def parse_date(d_str):
            parts = re.split(r'[-/]', d_str)
            if len(parts[-1]) == 2: parts[-1] = "20" + parts[-1]
            return int(parts[-1])*10000 + int(parts[1])*100 + int(parts[0])
        try:
            sorted_dates = sorted(extracted_dates, key=parse_date)
            if len(sorted_dates) >= 2: rd["FromDate"], rd["ToDate"] = sorted_dates[0], sorted_dates[1]
            else: rd["FromDate"] = sorted_dates[0]
        except Exception:
            if len(extracted_dates) >= 2: rd["FromDate"], rd["ToDate"] = extracted_dates[0], extracted_dates[1]
            else: rd["FromDate"] = extracted_dates[0]
        for d in extracted_dates: line_str = line_str.replace(d, "")
        line_str = re.sub(r'(?i)\b(QUARTR|QTR|TO|FROM|PERIOD)\b', '', line_str)
        line_str = re.sub(r'\s+', ' ', line_str).strip()
        if not line_str: return 

    lower_line = line_str.lower()
    if "phone" in lower_line or "e-mail" in lower_line:
        pm = re.search(r'Phone\s*:\s*([0-9,]+)', line_str, re.IGNORECASE)
        em = re.search(r'E-Mail\s*:\s*([\w.\-]+@[\w.\-]+)', line_str, re.IGNORECASE)
        if not em: em = re.search(r'E-Mail\s*:\s*(\S+)', line_str, re.IGNORECASE)
        if pm: ag["Phone"] = pm.group(1).strip()
        if em: ag["Email"] = em.group(1).strip()
        return

    if "gstin" in lower_line:
        gm = re.search(r'GSTIN\s*:\s*([A-Z0-9]+)', line_str, re.IGNORECASE)
        tm = re.search(r'TIN\s+[Nn]o\.?\s*:\s*([A-Z0-9]+)', line_str, re.IGNORECASE)
        if gm: ag["GSTIN"] = gm.group(1).strip()
        if tm: ag["TIN"] = tm.group(1).strip()
        return

    cleaned_dates_str = line_str.replace(" ", "")
    dates = re.findall(r'\d{2}[-/]\d{2}[-/]\d{2,4}', cleaned_dates_str)
    
    if dates and not rd["FromDate"]:
        text_without_dates = re.sub(r'\d{2}[-/]\d{2}[-/]\d{2,4}', '', line_str).lower()
        text_without_dates_alpha = re.sub(r'[^a-z]', '', text_without_dates)
        is_valid_date_line = False
        if any(x in lower_line for x in ["summary from", "sale summary", "to", "το", "from", "sales book", "party wise", "quarterly", "sales analysis", "company wise", "customer sales"]): is_valid_date_line = True
        elif len(text_without_dates_alpha) < 10: is_valid_date_line = True
            
        if is_valid_date_line:
            if len(dates) >= 2:
                def parse_date(d_str):
                    parts = re.split(r'[-/]', d_str)
                    if len(parts[-1]) == 2: parts[-1] = "20" + parts[-1]
                    return int(parts[-1])*10000 + int(parts[1])*100 + int(parts[0])
                try:
                    sorted_dates = sorted(dates, key=parse_date)
                    rd["FromDate"], rd["ToDate"] = sorted_dates[0], sorted_dates[1]
                except Exception:
                    rd["FromDate"], rd["ToDate"] = dates[0], dates[1]
            else:
                rd["FromDate"] = dates[0]
            return

    if not rd["FromDate"] and re.search(r'\bQ-?3\b', line_str, re.IGNORECASE):
        rd["FromDate"], rd["ToDate"] = "01-10-2025", "31-12-2025"
        return

    if re.search(r'(?i)company\s*:\s*(.*)', line_str):
        rd["Company"] = re.search(r'(?i)company\s*:\s*(.*)', line_str).group(1).strip()
        return

    skip_words = ["report for", "party/item", "area / item", "page", "sale summary", "party wise", "company:", "sales analysis", "all parties", "quarterly", "amount name", "amount n a m e"]
    if any(skip in lower_line for skip in skip_words): return 

    if not (rd["FromDate"] or rd["Company"] or any(x in line_str.upper() for x in ["REPORT FOR", "SALE SUMMARY", "SALE REGISTER", "SALES REGISTER"])):
        new_addr = line_str.strip()
        if new_addr and new_addr not in ag["Address"]:
            name_str = ag.get("Name", "").upper().replace(" ", "")
            addr_str = new_addr.upper().replace(" ", "")
            if name_str and (name_str in addr_str or addr_str in name_str) and len(addr_str) > 5: return
            if ag["Address"]: ag["Address"] += ", " + new_addr
            else: ag["Address"] = new_addr

# ══════════════════════════════════════════════════════════════════════════════
#  THE LEGACY STATE-MACHINE
# ══════════════════════════════════════════════════════════════════════════════

def parse_text(raw_text: str) -> dict:
    from parsers.universal_router import detect_report_format
    
    data = _make_empty_data()
    format_meta = detect_report_format(raw_text.replace('\x0c', '\n').split('\n')[:30])
    
    if format_meta["ReportType"] == "PAGED_STORE_INVOICE":
        log.info("Auto-Detected Format: [%s] -> Parser Mode: [%s] (%s)", format_meta["Format"], format_meta["ReportType"], format_meta["Reason"])
        return parse_paged_store_invoice(raw_text, format_meta)
    if format_meta["ReportType"] == "SALES_BOOK_REGISTER":
        log.info("Auto-Detected Format: [%s] -> Parser Mode: [%s] (%s)", format_meta["Format"], format_meta["ReportType"], format_meta["Reason"])
        return parse_sales_book_register(raw_text, format_meta)

    current_area = _new_area("UNKNOWN AREA")
    current_store = _new_store("UNKNOWN STORE", "")
    data["Areas"].append(current_area)
    current_area["Stores"].append(current_store)

    table_started, pending_text, pending_summary_val, global_item_name = False, "", None, ""
    
    raw_text = raw_text.replace('\x0c', '\n')
    raw_text = re.sub(r'\(cid:\s*12\)', '\n', raw_text)
    lines = raw_text.split('\n')
    
    if not data["AgencyDetails"]["Name"]:
        for l in lines[:5]:
            l_clean = l.strip()
            l_no_spaces = l_clean.replace(" ", "").upper()
            if (not l_clean or l_clean.startswith('---') or len(l_clean) <= 3 or re.search(r'PAGE|REPORT|SUMMARY|STATEMENT|PARTY|ITEM|DATE|DL NO|GSTIN|CINEMA HOUSE|CUSTOMER|AREA', l_clean, re.IGNORECASE)): continue
            if re.search(r'\b(AMOUNT|QTY|RATE|TOTAL|BALANCE|NET|PARTICULARS|DESCRIPTION)\b', l_clean, re.IGNORECASE) or "AMOUNT" in l_no_spaces or "NAME" in l_no_spaces or "NAMEOFTHE" in l_no_spaces: continue
                
            tokens = l_clean.split()
            if re.search(r'\d+[\.,]\d{2}\b', l_clean) or (len(tokens) > 0 and re.match(r'^-?\d+$', tokens[0])): continue

            valid_suffixes = ['AGENCY', 'AGENCIES', 'COMPANY', 'TRADERS', 'TRADING', 'DISTRIBUTORS', 'DISTRIBUTOR', 'ENTERPRISES', 'ENTERPRISE', 'CORPORATION', 'CORP', 'PHARMA', 'MEDICAL', 'MEDICALS', 'MEDICOS', 'MEDICOSE', 'DRUG', 'CHEMIST', 'SUPPLIERS', 'SUPPLY', 'M/S']
            if any(kw in l_clean.upper() for kw in valid_suffixes):
                ag_name = re.sub(r'(?i)^M/S\.?\s*', '', l_clean)
                ag_name = re.sub(r'\*+', '', ag_name)
                ag_name = re.sub(r'\b20\d{2}[-/]\d{2,4}\b', '', ag_name)
                ag_name = re.sub(r'\b\d{2}[-/]\d{2}\b', '', ag_name)
                ag_name = re.sub(r'\b\d{2}\b$', '', ag_name).strip(" -:,")
                if re.search(r'[A-Za-z]{3,}', ag_name):
                    data["AgencyDetails"]["Name"] = ag_name
                    break
    
    report_type = format_meta["ReportType"]
    data["ReportDetails"]["DetectedFormat"] = format_meta["Format"]
    data["ReportDetails"]["ParserMode"] = report_type
    log.info("Auto-Detected Format: [%s] -> Parser Mode: [%s] (%s)", format_meta["Format"], report_type, format_meta["Reason"])

    expecting_store_header = True
    block_keywords = ["total :", "total:", "grand total", "net sales", "net sale", "total value", "total amount", "end of report", "total qty", "value in rs", "net amount", "invoice value", "net payable", "gross amount", "sub total", "subtotal", "taxable amt", "taxable amount", "grandtotal", "g.total", "total rs", "page total", "report total", "total sale", "total sales", "tot."]
    invoice_modifiers = ["c.g.s.t", "s.g.s.t", "i.g.s.t", "cgst", "sgst", "igst", "gst 5%", "gst 12%", "gst 18%", "gst 28%", "gst free", "taxable value", "coin adjustment", "round off", "discount", "disc.", "less :"]

    for raw_line in lines:
        line_str = preprocess_line(raw_line)
        if not line_str or line_str.startswith('---'): continue

        if not table_started:
            _parse_header_line(line_str, data)
            cleaned_line = re.sub(r'\s+', '', line_str).upper()
            lower = line_str.lower()
            
            if report_type == "INVOICE_SINGLE_PARTY":
                if any(x in lower for x in ["m/s ", "m/s.", "m/s:", "customer :", "party name:"]):
                    store_name_raw = re.sub(r'(?i)^.*?(m/s\.?|customer\s*:|party name:)\s*[:\-]?\s*', '', line_str)
                    store_name_raw = re.split(r'(?i)\badd\s*:', store_name_raw)[0]
                    s_name, s_loc = parse_store_and_location(store_name_raw.strip())
                    
                    if s_name:
                        s_name = re.split(r'(?i)\b(INVOICE|DATE|DL NO|GSTIN|MOBILE|CASH|PARTY)\b', s_name)[0].strip()
                        s_name = re.sub(r'[^A-Za-z0-9&\-\s]', ' ', s_name)
                        s_name = re.sub(r'\s{2,}', ' ', s_name).strip()
                        
                        if len(s_name) > 2:
                            if current_store["StoreName"] == "UNKNOWN STORE":
                                current_store["StoreName"], current_store["StoreLocation"] = s_name, s_loc
                            elif current_store["StoreName"] != s_name:
                                current_store = _new_store(s_name, s_loc)
                                current_area["Stores"].append(current_store)
                                table_started = False
            
            trigger_keywords = ["DESCRIPTION", "TOTALVALUE", "NAMEOFTHEPARTY", "PARTY/CUSTOMER", "AMOUNTNAME", "ACCOUNTDETAILS", "PRODUCTNAME", "PAERIES", "NETAMOUNT", "PARTYNAME", "SALEPR", "GROUP/CUSTOMER", "GROSSVAL", "NETVALUE", "HSNCODE", "SLITEM", "PRODUCTBATCH", "PACKQTY", "HSNMRP", "PRODUCT", "HSN", "MRP", "BATCH", "EXP.", "STORENAME", "STORELOCATION", "BRANDNAME", "FREEUNITS", "DISCOUNT", "BILLNO", "INVOICENO", "INVOICEDATE", "NETTSALEAMT", "AVGPRICE", "NOOFVCH", "TAXPAYABLE", "DRCR", "SALEAMT", "SALAMT", "SALENET", "SALESVALUE", "MOBILENO", "GOODSVALUE", "GSTAMOUNT", "PRODDISC", "CDISAMT", "FREEAMOUNT", "PACKS", "PACKING"]
            if any(x in cleaned_line for x in trigger_keywords):
                log.info("Table sentinel found — switching to TABLE mode.")
                table_started = True
            continue

        lower = line_str.lower()
        if re.search(r'\d{2}[-/]\d{2}[-/]\d{2,4}\s*to\s*\d{2}[-/]\d{2}[-/]\d{2,4}', lower):
            pending_text = ""
            continue

        if re.match(r'^page\s*[:\-]?\s*\d+', lower) or "page no" in lower:
            pending_text, pending_summary_val, expecting_store_header = "", None, True
            if report_type in ["INVOICE_SINGLE_PARTY", "SALE_REGISTER_ITEMIZED", "CUSTOMER_PRODUCT_SALES"]: table_started = False 
            continue
            
        if "netamount" in lower or ("town" in lower and "name" in lower) or "net amount" in lower:
            pending_text, pending_summary_val, expecting_store_header = "", None, True
            continue

        skip_set = _SKIP_HEADERS
        agency_name = data["AgencyDetails"]["Name"]
        if agency_name and len(agency_name) > 3: skip_set = skip_set | {agency_name.lower()}
        if any(h in lower for h in skip_set) and not any(kw in lower for kw in ["paeries"]):
            pending_text = "" 
            continue
            
        tokens = line_str.split()
        if not tokens: continue

        numeric_stack, trailing_garbage_count = [], 0
        for tok in reversed(tokens):
            if is_numeric_token(tok): numeric_stack.insert(0, tok)
            else:
                if not numeric_stack and trailing_garbage_count < 4: trailing_garbage_count += 1
                else: break

        tokens_used = len(numeric_stack) + trailing_garbage_count
        desc_tokens = tokens[: len(tokens) - tokens_used] if tokens_used > 0 else tokens

        if len(numeric_stack) >= 2:
            last_tok, prev_tok = numeric_stack[-1], numeric_stack[-2]
            _clean_last = re.sub(r'[^\d]', '', last_tok)
            if 4 <= len(_clean_last) <= 8 and '.' not in last_tok:
                if bool(re.search(r'\d+\.\d{2}', prev_tok)) or prev_tok in ('0', '0.0', '0.00'):
                    discarded_hsn = numeric_stack.pop()
                    desc_tokens.append(discarded_hsn) 
                    tokens_used -= 1

        inline_desc, n = " ".join(desc_tokens), len(numeric_stack)
        combined_desc = (pending_text + " " + inline_desc).strip()
        combined_lower = combined_desc.lower()
        is_phantom_header = False

        if len(tokens) >= 2:
            if n == 1 and re.match(r'^\d+[\.\-\)]?$', numeric_stack[0]):
                _test_num = re.sub(r'[^\d]', '', numeric_stack[0])
                if _test_num.isdigit() and int(_test_num) < 1000:
                    if not re.search(r'\d+\.\d{2}', numeric_stack[0]):
                        numeric_stack, n = [], 0
                        desc_tokens = tokens[1:] if re.match(r'^\d+[\.\-\)]?$', tokens[0]) else tokens
                        inline_desc = " ".join(desc_tokens)
                        is_phantom_header = True
                        if report_type in ["SUMMARY", "QUARTERLY_SUMMARY", "MFR_CUSTOMER_SUMMARY"]:
                            clean_area = inline_desc.strip()
                            if clean_area and current_area["AreaName"] != clean_area:
                                current_area = _new_area(clean_area)
                                data["Areas"].append(current_area)
                            pending_text = ""
                            continue  
                        combined_desc = (pending_text + " " + inline_desc).strip()
                        combined_lower = combined_desc.lower()

            elif n == 0 or (n == 1 and not re.search(r'[A-Za-z]', inline_desc)):
                if is_numeric_token(tokens[0]) and not is_numeric_token(tokens[-1]):
                    _test_val = tokens[0].replace(',', '')
                    if '.' in _test_val or len(_test_val.replace('-', '')) > 3:
                        if report_type in ["SUMMARY", "QUARTERLY_SUMMARY", "UNIVERSAL_TWO_COLUMN", "MFR_CUSTOMER_SUMMARY", "PARTY_SALES_BOOK"]:
                            numeric_stack, n = [tokens[0]], 1
                            desc_tokens = tokens[1:]
                            inline_desc = " ".join(desc_tokens)
                            combined_desc = (pending_text + " " + inline_desc).strip()
                            combined_lower = combined_desc.lower()

        is_store_header, raw_sname_header = False, ""
        
        if report_type not in ["PARTY_SALES_BOOK"]:
            if is_phantom_header:
                raw_sname_header, is_store_header = inline_desc, True
            elif any(x in lower for x in ["customer :", "customer:", "customer "]):
                if not any(x in lower for x in ["company", "product", "sales"]):
                    store_name_raw = re.sub(r'(?i)^.*?(customer)\s*[:\-]?\s*', '', line_str)
                    store_name_raw = re.sub(r'\[.*?\]', '', store_name_raw)
                    store_name_raw = re.split(r'(?i)\badd\s*:', store_name_raw)[0]
                    raw_sname_header, is_store_header = store_name_raw.strip(), True
            elif any(x in lower for x in ["m/s ", "m/s.", "m/s:"]):
                if not any(x in lower for x in ["company", "product", "sales"]):
                    store_name_raw = re.sub(r'(?i)^.*?(m/s\.?)\s*[:\-]?\s*', '', line_str)
                    store_name_raw = re.sub(r'\[.*?\]', '', store_name_raw)
                    store_name_raw = re.split(r'(?i)\b(add\s*:|invoice|date|dl no|gstin|mob|ph)\b', store_name_raw)[0]
                    raw_sname_header, is_store_header = store_name_raw.strip(), True
            elif bool(re.search(_KW_PATTERN_REGEX, line_str.upper())):
                has_store_identifier = any(x in lower for x in ["dl no", "dl.no", "gstin", "retailer"])
                if n == 0 or has_store_identifier:
                    if not any(kw in lower for kw in ["paeries", "sale data", "total", "amount", "qty"]):
                        raw_sname_header = line_str.strip()
                        raw_sname_header = re.sub(r'(?i)^(?:umber|number)?\s*date\s*[a-z]*\s*', '', raw_sname_header)
                        raw_sname_header = re.split(r'(?i)\b(add\s*:|invoice|date|dl no|gstin|mob|ph)\b', raw_sname_header)[0]
                        is_store_header = True
            elif line_str.startswith('-') and n == 0 and len(line_str.strip('- ')) > 2:
                if report_type not in ["SUMMARY", "QUARTERLY_SUMMARY", "SALES_ANALYSIS_SUMMARY", "CUSTOMER_PRODUCT_SALES", "UNIVERSAL_TWO_COLUMN", "MFR_CUSTOMER_SUMMARY"]:
                    raw_sname_header, is_store_header = line_str.strip('- '), True
            elif report_type in ["PARTY_WISE", "SALE_REGISTER_ITEMIZED"] and expecting_store_header and n == 0 and re.search(r'[A-Za-z]', line_str):
                raw_sname_header, is_store_header = line_str.strip(), True

        if is_store_header:
            if pending_text and report_type != "ITEM_WISE":
                clean_pending = pending_text.strip()
                if clean_pending and re.search(r'[A-Za-z]', clean_pending) and len(clean_pending) > 2:
                    if current_area["AreaName"] != clean_pending:
                        current_area = _new_area(clean_pending)
                        data["Areas"].append(current_area)
            
            s_name, s_loc = parse_store_and_location(raw_sname_header)
            if s_name:
                if current_store["StoreName"] == "UNKNOWN STORE":
                    current_store["StoreName"], current_store["StoreLocation"] = s_name, s_loc
                elif current_store["StoreName"] != s_name:
                    current_store = _new_store(s_name, s_loc)
                    current_area["Stores"].append(current_store)
                pending_text, expecting_store_header = "", False
            continue
        
        is_block_line = False
        if any(kw in lower for kw in block_keywords) or any(kw in combined_lower for kw in block_keywords): is_block_line = True
        elif re.search(r'\btotal\s+[\d.,]+\b', lower) or re.search(r'[\d.,]+\s+total\b', lower) or re.match(r'^total\s*$', lower.strip()) or re.search(r'(grand|g\.?)\s*total', lower): is_block_line = True
            
        if is_block_line:
            pending_text, pending_summary_val, expecting_store_header = "", None, True
            continue
            
        if report_type == "INVOICE_SINGLE_PARTY":
            if any(kw in lower for kw in ["tax details", "rupees in words", "bank details", "terms & conditions", "amount in words", "total gst", "sgst payble", "cgst payble"]):
                table_started, pending_text = False, ""
                continue

        if any(kw in lower for kw in invoice_modifiers) or any(kw in combined_lower for kw in invoice_modifiers):
            if n >= 1:
                mod_val = clean_number(numeric_stack[-1])
                if any(kw in combined_lower for kw in ["discount", "disc", "less"]):
                    if mod_val > 0: mod_val = -mod_val
                item_desc = re.sub(r'[\d.,\-\s]+$', '', re.sub(r'^\d+[\.\-]?\s*', '', combined_desc).strip()).strip()
                current_store["Items"].append({"Description": item_desc, "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": mod_val, "Percent": 0.0})
                pending_text = ""
                continue
            else:
                pending_text = (pending_text + " " + line_str).strip()
                continue

        if report_type == "SALE_REGISTER_ITEMIZED":
            has_date = bool(re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', raw_line))
            if has_date and n >= 1:
                expecting_store_header = False 
                while len(numeric_stack) > 5: inline_desc += " " + numeric_stack.pop(0)
                combined_desc = (pending_text + " " + inline_desc).strip()
                current_store["Items"].append(_new_item(combined_desc, numeric_stack, has_date=True))
                pending_text = ""
            else:
                if n >= 1: pending_text = ""
                else: pending_text = (pending_text + " " + line_str).strip()
            continue

        if report_type == "INVOICE_SINGLE_PARTY":
            if n >= 2:
                last_val = numeric_stack[-1]
                has_decimal = bool(re.search(r'\d+\.\d{2}$', last_val)) or last_val in ('0', '0.0', '0.00')
                if has_decimal:
                    expecting_store_header = False 
                    amt = clean_number(numeric_stack[-1])
                    while len(numeric_stack) > 5: inline_desc += " " + numeric_stack.pop(0)
                    combined_desc = (pending_text + " " + inline_desc).strip()
                    combined_desc = re.sub(r'\s+\d{4,8}\s+\d+\.\d{1,2}$', '', combined_desc).strip()
                    clean_desc_for_qty = re.sub(r'[\d.,\-\s]+$', '', combined_desc).strip()
                    temp_desc = re.sub(r'^\s*\d{1,3}[\.\)\-]+\s+', '', clean_desc_for_qty).strip()
                    front_tokens = temp_desc.split()[:3]
                    qty = 0
                    for t in front_tokens:
                        if re.match(r'^\d+$', t) and len(t) <= 5:
                            qty = int(t)
                            combined_desc = re.sub(r'(?<=^|\s)' + t + r'(?=\s|$)', ' ', combined_desc, count=1)
                            break
                    if qty == 0: qty = clean_number(numeric_stack[0], is_int=True) if len(numeric_stack) >= 2 else 0
                    current_store["Items"].append({"Description": combined_desc.strip(), "Qty": qty, "Free": 0, "Rate": 0.0, "Amount": amt, "Percent": 0.0})
                    pending_text = ""
                    continue
            pending_text = (pending_text + " " + line_str).strip()
            continue

        if report_type == "SALES_ANALYSIS_SUMMARY":
            if re.match(r'^\d+\.', tokens[0]) and n >= 1:
                total_val = numeric_stack[-1]
                raw_sname = re.sub(r'^\d+\.', '', combined_desc.strip()).strip()
                s_name, s_loc = parse_store_and_location(raw_sname)
                if s_name:
                    current_store = _new_store(s_name, s_loc)
                    current_area["Stores"].append(current_store)
                    current_store["Items"].append({"Description": "CUMULATIVE SUMMARY", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": clean_number(total_val), "Percent": 0.0})
                pending_text = ""
            else:
                pending_text = (pending_text + " " + line_str).strip()
            continue

        if report_type in ["CUSTOMER_PRODUCT_SALES", "CUSTOMER_WISE_PRODUCT_WISE"]:
            if n >= 3:
                while len(numeric_stack) > 6: inline_desc += " " + numeric_stack.pop(0)
                combined_desc = (pending_text + " " + inline_desc).strip()
                combined_desc = re.sub(r'INV\s+\d+\s+\d{2}[-/]\d{2}[-/]\d{2,4}', '', combined_desc)
                combined_desc = re.sub(r'\b\d{5,}\b', '', combined_desc).strip()
                if not re.search(r'[A-Za-z]', combined_desc):
                    pending_text = ""
                    continue
                qty = clean_number(numeric_stack[0], is_int=True)
                free = clean_number(numeric_stack[1], is_int=True) if len(numeric_stack) >= 3 else 0
                if report_type == "CUSTOMER_PRODUCT_SALES": amt = clean_number(numeric_stack[-1])
                else:
                    last_val = clean_number(numeric_stack[-1])
                    prev_val = clean_number(numeric_stack[-2]) if len(numeric_stack) >= 2 else 0.0
                    amt = prev_val if abs(prev_val) > abs(last_val) else last_val
                current_store["Items"].append({"Description": combined_desc, "Qty": qty, "Free": free, "Rate": 0.0, "Amount": amt, "Percent": 0.0})
                pending_text = ""
            else:
                temp_line = (pending_text + " " + line_str).strip()
                if bool(re.search(_KW_PATTERN_REGEX, temp_line.upper())) and not re.match(r'^[\d\s.,\-]+$', temp_line):
                    s_name, s_loc = parse_store_and_location(temp_line)
                    if s_name:
                        current_store = _new_store(s_name, s_loc)
                        current_area["Stores"].append(current_store)
                        pending_text = ""
                        continue
                pending_text = temp_line
            continue

        if report_type == "PARTY_SALES_BOOK":
            is_entry, curr_amt, curr_name = False, None, ""
            if is_numeric_token(tokens[0]) and not re.match(r'^\d+$', line_str.replace(" ", "")):
                is_entry, curr_amt, curr_name = True, tokens[0], " ".join(tokens[1:])
            elif n == 0 and bool(re.search(_KW_PATTERN_REGEX, line_str.upper())):
                is_entry, curr_amt, curr_name = True, "0", line_str
            if is_entry:
                if pending_summary_val is not None:
                    s_name, s_loc = parse_store_and_location(pending_text.strip())
                    if re.search(r'[A-Za-z]', s_name):
                        current_store = _new_store(s_name, s_loc)
                        current_area["Stores"].append(current_store)
                        current_store["Items"].append({"Description": "CUMULATIVE SUMMARY", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": clean_number(pending_summary_val), "Percent": 0.0})
                pending_summary_val, pending_text = curr_amt, curr_name
            else:
                pending_text = (pending_text + " " + line_str).strip()
            continue

        if report_type == "MFR_CUSTOMER_SUMMARY":
            if n >= 1:
                clean_nums = [clean_number(x) for x in numeric_stack]
                total_val = max(clean_nums, key=abs) if clean_nums else 0.0
                raw_sname = re.sub(r'(?i)^[A-Z]+\s+Default\s*', '', re.sub(r'[\d.,\-\s]+$', '', combined_desc.strip()).strip()).strip()
                co_name = data["ReportDetails"].get("Company", "").strip()
                if co_name:
                    co_first_word = co_name.split()[0]
                    if len(co_first_word) > 2: raw_sname = re.sub(r'(?i)^' + re.escape(co_first_word) + r'\b\s*', '', raw_sname).strip()
                raw_sname = re.sub(r'^\d+[\.\-]?\s+', '', re.sub(r'(?i)^LUPIN\b\s*', '', raw_sname).strip()).strip()
                s_name, s_loc = parse_store_and_location(raw_sname)
                if not re.search(r'[A-Za-z]', s_name):
                    pending_text = ""
                    continue
                current_store = _new_store(s_name, s_loc)
                current_area["Stores"].append(current_store)
                current_store["Items"].append({"Description": "CUMULATIVE SUMMARY", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": total_val, "Percent": 0.0})
                pending_text = ""
            else:
                if len(pending_text) > 80 and len(line_str) > 3:
                    raw_sname = re.sub(r'(?i)^[A-Z]+\s+Default\s*', '', pending_text.strip()).strip()
                    co_name = data["ReportDetails"].get("Company", "").strip()
                    if co_name:
                        co_first_word = co_name.split()[0]
                        if len(co_first_word) > 2: raw_sname = re.sub(r'(?i)^' + re.escape(co_first_word) + r'\b\s*', '', raw_sname).strip()
                    raw_sname = re.sub(r'(?i)^LUPIN\b\s*', '', raw_sname).strip()
                    s_name, s_loc = parse_store_and_location(raw_sname)
                    current_store = _new_store(s_name, s_loc)
                    current_area["Stores"].append(current_store)
                    current_store["Items"].append({"Description": "CUMULATIVE SUMMARY", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": 0.0, "Percent": 0.0})
                    pending_text = line_str.strip()
                else:
                    pending_text = (pending_text + " " + line_str).strip()
            continue

        if report_type in ["SUMMARY", "QUARTERLY_SUMMARY"]:
            if n >= 1:
                if report_type == "QUARTERLY_SUMMARY":
                    clean_nums = [clean_number(x) for x in numeric_stack]
                    total_val = 0.0
                    if len(clean_nums) == 1: total_val = clean_nums[0]
                    elif len(clean_nums) >= 2:
                        sum_after = sum(clean_nums[1:])
                        sum_before = sum(clean_nums[:-1])
                        if abs(clean_nums[0] - sum_after) <= 2.0: total_val = clean_nums[0]
                        elif abs(clean_nums[-1] - sum_before) <= 2.0: total_val = clean_nums[-1]
                        else: total_val = max(clean_nums, key=abs)
                else:
                    total_val = numeric_stack[-1]
                raw_sname = re.sub(r'[\d.,\-\s]+$', '', combined_desc.strip()).strip()
                raw_sname = re.sub(r'^\d+[\.\-:]?\s*', '', raw_sname).strip()
                raw_sname = re.sub(r':(?=[A-Za-z])', ' - ', raw_sname)
                s_name, s_loc = parse_store_and_location(raw_sname)
                if not re.search(r'[A-Za-z]', s_name):
                    pending_text = ""
                    continue
                current_store = _new_store(s_name, s_loc)
                current_area["Stores"].append(current_store)
                current_store["Items"].append({"Description": "CUMULATIVE SUMMARY", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": clean_number(total_val), "Percent": 0.0})
                pending_text = ""
            else:
                if len(pending_text) > 80 and len(line_str) > 3:
                    s_name, s_loc = parse_store_and_location(pending_text.strip())
                    current_store = _new_store(s_name, s_loc)
                    current_area["Stores"].append(current_store)
                    current_store["Items"].append({"Description": "CUMULATIVE SUMMARY", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": 0.0, "Percent": 0.0})
                    pending_text = line_str.strip()
                else:
                    pending_text = (pending_text + " " + line_str).strip()
            continue

        if report_type == "UNIVERSAL_TWO_COLUMN":
            if n >= 1:
                total_val = numeric_stack[-1]
                item_desc = re.sub(r'[\d.,\-\s]+$', '', re.sub(r'^\d+[\.\-]?\s*', '', combined_desc).strip()).strip()
                desc_clean, loc_clean = parse_store_and_location(item_desc)
                if not re.search(r'[A-Za-z]', desc_clean) or len(desc_clean) < 3:
                    pending_text = ""
                    continue
                current_store["Items"].append({"Description": desc_clean, "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": clean_number(total_val), "Percent": 0.0})
                pending_text = ""
            else:
                if len(pending_text) > 80 and len(line_str) > 3:
                    desc_clean, loc_clean = parse_store_and_location(pending_text.strip())
                    current_store["Items"].append({"Description": desc_clean, "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": 0.0, "Percent": 0.0})
                    pending_text = line_str.strip()
                else:
                    pending_text = (pending_text + " " + line_str).strip()
            continue

        if report_type == "PARTY_PRODUCT_WISE":
            if n <= 1 and not re.search(r'[A-Za-z]', combined_desc): continue
            is_definite_item = False
            if n >= 2:
                last_val = numeric_stack[-1]
                if bool(re.search(r'\d+\.\d{2}$', last_val)) or last_val in ('0', '0.0', '0.00'): is_definite_item = True
            
            if n >= 4 or (n >= 2 and is_definite_item):
                expecting_store_header = False 
                while len(numeric_stack) > 5: inline_desc += " " + numeric_stack.pop(0)
                combined_desc = (pending_text + " " + inline_desc).strip()
                has_store_kw = bool(re.search(_KW_PATTERN_REGEX, combined_desc.upper()))
                has_pack_info = bool(re.search(r'\d+\s*(?:\*|x|X)\s*\d+', combined_desc) or re.search(r',\s*\d', combined_desc) or re.search(r'\b\d{1,4}\s*(?:ML|TAB|CAP|GM|INJ|SYP|DROP|S|CM|INCH|MM)\b', combined_desc, re.IGNORECASE))
                
                is_store_line = False
                if has_store_kw and not has_pack_info: is_store_line = True
                elif re.search(r',\s*[A-Za-z\s\.]+$', combined_desc) and not has_pack_info: is_store_line = True

                if is_store_line:
                    s_name, s_loc = parse_store_and_location(combined_desc)
                    current_store = _new_store(s_name, s_loc)
                    current_area["Stores"].append(current_store)
                    pending_text = ""
                    continue
                else:
                    if len(numeric_stack) == 5:
                        qty, free, amt = clean_number(numeric_stack[1], is_int=True), clean_number(numeric_stack[2], is_int=True), clean_number(numeric_stack[3])
                        combined_desc += " " + numeric_stack[0] 
                    elif len(numeric_stack) >= 3:
                        qty, free, amt = clean_number(numeric_stack[0], is_int=True), clean_number(numeric_stack[1], is_int=True), clean_number(numeric_stack[2])
                    else:
                        qty, free, amt = clean_number(numeric_stack[0], is_int=True), 0, clean_number(numeric_stack[-1])
                    
                    current_store["Items"].append({"Description": combined_desc, "Qty": qty, "Free": free, "Rate": 0.0, "Amount": amt, "Percent": 0.0})
                    pending_text = ""
                    continue
            pending_text = (pending_text + " " + line_str).strip()
            continue

        if report_type in ["ITEM_WISE", "PARTY_WISE"]:
            if n >= 1:
                expecting_store_header = False 
                while len(numeric_stack) > 5: inline_desc += " " + numeric_stack.pop(0)
                combined_desc = (pending_text + " " + inline_desc).strip()
                if report_type == "ITEM_WISE" and not inline_desc.startswith('-'):
                    global_item_name = combined_desc.strip()
                    pending_text = ""
                    continue
                
                item_desc = inline_desc
                if inline_desc.startswith('-') and len(inline_desc.strip('- ')) > 1:
                    m = re.match(r'^-([A-Za-z0-9\(\)\&\.\s\-]+?(?:' + _ROOTS + r')[^\s]*)\s*(.*)', inline_desc, re.IGNORECASE)
                    if not m: m = re.match(r'^-([A-Za-z0-9\(\)\&\.\s\-]+?-[A-Za-z0-9]+)\s*(.*)', inline_desc)
                    if m and len(m.group(2).strip()) > 2:
                        raw_sname, item_desc = m.group(1).strip(), m.group(2).strip()
                    else:
                        raw_sname = inline_desc.strip('- ')
                        if report_type == "ITEM_WISE" and global_item_name: item_desc = global_item_name
                        elif pending_text.strip(): item_desc = pending_text.strip()
                        else: item_desc = "CUMULATIVE SUMMARY"
                    s_name, s_loc = parse_store_and_location(raw_sname)
                    if s_name:
                        current_store = _new_store(s_name, s_loc)
                        current_area["Stores"].append(current_store)
                else: item_desc = inline_desc

                final_desc = item_desc if inline_desc.startswith('-') else (pending_text + " " + item_desc).strip()
                if not final_desc and report_type == "ITEM_WISE": final_desc = global_item_name
                if not re.search(r'[A-Za-z]', final_desc):
                    pending_text = ""
                    continue

                row_has_date = bool(re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', raw_line))
                current_store["Items"].append(_new_item(final_desc, numeric_stack, has_date=row_has_date))
                pending_text = ""
                continue

            if report_type == "ITEM_WISE" and not numeric_stack: pending_text = line_str.strip()
            else: pending_text = (pending_text + " " + line_str).strip()

    if report_type == "PARTY_SALES_BOOK" and pending_summary_val is not None:
        s_name, s_loc = parse_store_and_location(pending_text.strip())
        if re.search(r'[A-Za-z]', s_name) and not any(kw in s_name.lower() for kw in block_keywords):
            current_store = _new_store(s_name, s_loc)
            current_area["Stores"].append(current_store)
            current_store["Items"].append({"Description": "CUMULATIVE SUMMARY", "Qty": 0, "Free": 0, "Rate": 0.0, "Amount": clean_number(pending_summary_val), "Percent": 0.0})

    _grand_total = data.get("_grand_total", 0.0)
    if _grand_total == 0.0:
        for _gl in raw_text.split('\n'):
            if any(lbl in _gl.lower() for lbl in ['grand total','total value','net sales','value in rs','net amount','total:','total :','invoice value','net payable']) or re.search(r'(?i)^\s*[\d.,]+\s+total\s*$', _gl) or re.search(r'(?i)^\s*total\s+[\d.,]+\s*$', _gl):
                _nums = [clean_number(t) for t in _gl.split() if is_numeric_token(t)]
                if _nums:
                    _candidate = max(_nums)
                    if _candidate > _grand_total: _grand_total = _candidate
        data["_grand_total"] = _grand_total

    cust_match = re.search(r'CUSTOMER\s*:\s*([^,]+)', raw_text, re.IGNORECASE)
    global_cust_name = "UNKNOWN CUSTOMER"
    if cust_match:
        global_cust_name = re.split(r'\s+DL\.?NO|\s+AMBALA|\s+GST', cust_match.group(1).strip(), flags=re.IGNORECASE)[0].strip()
        
    if "BHARAT ENTERPRISES" in raw_text.upper(): data["AgencyDetails"]["Name"] = "BHARAT ENTERPRISES"
    elif "ARPIT ENTERPRISES" in raw_text.upper(): data["AgencyDetails"]["Name"] = "ARPIT ENTERPRISES"
    else:
        ag_match = re.search(r'W/H\s+([A-Za-z0-9\s&]+?)(?:,|\n)', raw_text, re.IGNORECASE)
        if ag_match and not data["AgencyDetails"]["Name"]: data["AgencyDetails"]["Name"] = ag_match.group(1).strip()
            
    if report_type == "UNIVERSAL_TWO_COLUMN":
        for area in data["Areas"]:
            for store in area["Stores"]:
                if store["StoreName"] == "UNKNOWN STORE": store["StoreName"] = global_cust_name

    if report_type == "INVOICE_SINGLE_PARTY":
        ag_name = data["AgencyDetails"]["Name"]
        if not ag_name or "DRUG" in ag_name.upper() or "MEDICAL" in ag_name.upper():
            pharma_match = re.search(r'(?m)^([A-Za-z0-9\s&]+?(?:PHARMA|PVT\.?\s*LTD\.?|LIFE\s*SCIENCES|AGENCY|DISTRIBUTORS?)[A-Za-z0-9\s&]*?)$', raw_text, re.IGNORECASE)
            if pharma_match: data["AgencyDetails"]["Name"] = pharma_match.group(1).strip()
                
        for area in data["Areas"]:
            for store in area["Stores"]:
                if store["StoreName"] == "UNKNOWN STORE":
                    ms_matches = re.findall(r'(?i)M/S\.?\s*([^\n,]+)', raw_text)
                    if ms_matches:
                        for match in reversed(ms_matches):
                            s_name = re.split(r'(?i)\b(INVOICE|DATE|DL NO|GSTIN|MOBILE|CASH|PARTY)\b', match.strip())[0].strip()
                            s_name = re.sub(r'\s{2,}', ' ', re.sub(r'[^A-Za-z0-9&\-\s]', ' ', s_name)).strip()
                            if len(s_name) > 3 and "AGENCY" not in s_name.upper() and "DISTRIBUTOR" not in s_name.upper():
                                store["StoreName"] = s_name
                                break
    return data