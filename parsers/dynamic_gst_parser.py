import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class DynamicGSTInvoiceParser:
    def __init__(self, format_meta=None):
        self.format_meta = format_meta or {}
        # ULTIMATE REGEX: Decouples the messy front part from the highly structured back part
        # (Kept EXACTLY as you tested it, since it perfectly skips GST and catches math)
        self.line_item_pattern = re.compile(
            r"(?P<sno>\d+)\.\s*"                 
            r"(?P<front_text>.+?)\s+"            
            r"(?P<hsn>\d{2,8})\s+"               
            r"(?P<mrp>[\d\.,]+)\s+"                   
            r"(?P<rate>[\d\.,]+)\s+"                  
            r"(?P<discount>[\d\.,]+)\s+"              
            r"(?P<sgst>[\d\.,]+)\s+"                  
            r"(?P<cgst>[\d\.,]+)\s+"                  
            r"(?P<amount>[\d\.,]+)"
        )

    def extract(self, raw_lines: List[str]) -> Dict:
        
        # =====================================================================
        # PHASE 1: PRECISE HEADER EXTRACTION (Line-by-Line)
        # =====================================================================
        seller_name = ""
        seller_addr = ""
        seller_phone = ""
        seller_gstin = ""
        buyer_name = ""
        buyer_addr = ""
        doc_date = ""
        bill_no = ""

        all_header_text = " ".join(raw_lines[:30])

        for line in raw_lines:
            line_str = str(line).strip()
            if not line_str: continue

            # 1. Agency Name
            if not seller_name:
                s_match = re.search(r"([A-Z][A-Za-z0-9\s\.\-]+?(?:AGENCY|DISTRIBUTORS?|TRADERS?))", line_str)
                if s_match: seller_name = s_match.group(1).strip()

            # 2. Buyer Name
            if not buyer_name:
                b_match = re.search(r"(?:M/S\.?|M/s)\s*([A-Z][A-Za-z0-9\s\.\-]+?(?:PHARMACY|MEDICAL STORE|CLINIC|HOSPITAL))", line_str, re.IGNORECASE)
                if b_match: buyer_name = b_match.group(1).strip()

            # 3. Bill Date — prefer complete dates (4-digit year) over truncated ones
            d_match = re.search(r'(?:Date|Dt\.?|Due Date)\s*[:\-]?\s*(\d{2}[/\-]\d{2}[/\-]\d{2,4})', line_str, re.IGNORECASE)
            if d_match:
                candidate = d_match.group(1)
                if bool(re.search(r'\d{4}$', candidate)):
                    doc_date = candidate
                elif not doc_date:
                    doc_date = candidate

            # 4. Bill No
            if not bill_no:
                b_no_match = re.search(r'(?:Invoice No|Bill No)\s*[:\.]?\s*([A-Z0-9\-/]+)', line_str, re.IGNORECASE)
                if b_no_match: bill_no = b_no_match.group(1)

            # 5. GSTIN
            if not seller_gstin:
                gstin_match = re.search(r'GSTIN\s*:\s*([A-Z0-9]+)', line_str, re.IGNORECASE)
                if gstin_match: seller_gstin = gstin_match.group(1).strip()

            # 6. Phone
            if not seller_phone:
                phone_match = re.search(r'Phone\s*:\s*([0-9,\s]+)', line_str, re.IGNORECASE)
                if phone_match: seller_phone = phone_match.group(1).strip()

            # 7. Seller Address — collect address-like lines before the data table
            if not seller_gstin:
                addr_clean = line_str.strip()
                junk_keywords = r'(Invoice|GSTIN|Phone|Email|Page|M/S|AGENCY|DISTRIBUTOR|Sales Man|Due Date|Licence No|D\.L\.|E-Mail)'
                if (len(addr_clean) > 20 and re.search(r'[A-Za-z]{3,}', addr_clean)
                    and not re.search(junk_keywords, addr_clean, re.IGNORECASE)
                    and re.search(r'\d', addr_clean)
                    and re.search(r'(?:ROAD|PIN|DIST|WARD|NO[ -]|BENGAL|NADIA|SANTIPUR)', addr_clean, re.IGNORECASE)):
                    seller_addr = (seller_addr + ", " + addr_clean).strip(", ") if seller_addr else addr_clean

        if not doc_date or not re.search(r'\d{4}$', doc_date):
            for m in re.finditer(r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b', all_header_text):
                doc_date = m.group(1)
                break

        seller_name = seller_name if seller_name else "UNKNOWN AGENCY"
        buyer_name = buyer_name if buyer_name else "UNKNOWN STORE"

        # =====================================================================
        # PHASE 2: GROUP LINES BY PAGE, EXTRACT ONLY DATA TABLE ROWS
        # =====================================================================
        # Split raw_lines into page groups (separated by \x0c)
        pages = [[]]
        for line in raw_lines:
            if '\x0c' in line:
                before, after = line.split('\x0c', 1)
                if before.strip():
                    pages[-1].append(before)
                pages.append([after] if after.strip() else [])
            else:
                pages[-1].append(line)

        data_lines_text = ""
        for page_lines in pages:
            in_table = False
            found_data_on_page = False
            for line in page_lines:
                s = line.strip()
                if not s:
                    continue
                # Detect column header
                if re.search(r'Sn\.?\s+Qty', s, re.IGNORECASE) or re.search(r'Sn\.?\s+Qty\.?\s+Pack', s, re.IGNORECASE):
                    in_table = True
                    continue
                # Stop at summary/total lines (only after at least one data row found on this page,
                # to avoid premature stops from carry-forward lines like "TOTAL B/F")
                if in_table and re.search(r'^(?:CLASS|GST\s+\d+\.\d+|TOTAL(?!\s+B/F)|GRAND|SUB\s+TOTAL)', s, re.IGNORECASE):
                    if found_data_on_page:
                        in_table = False
                        continue
                if in_table and s and re.match(r'\d+\.', s):
                    data_lines_text += s + "\n"
                    found_data_on_page = True

        raw_text = data_lines_text.replace('\n', ' ')
        raw_text = re.sub(r'\s+', ' ', raw_text).strip()

        extracted_items = []
        
        # =====================================================================
        # PHASE 3: ITERATE, TOKENIZE, AND MAP TO EXACT CSV SCHEMA
        # =====================================================================
        for match in self.line_item_pattern.finditer(raw_text):
            data = match.groupdict()
            
            front_text = data["front_text"].strip()
            tokens = front_text.split()
            
            qty = 0.0
            pack = ""
            expiry = ""
            batch = ""
            clean_tokens = []
            expiry_prefix = ""

            for t in tokens:
                upper_t = t.upper()

                # A. Expiry — extract date portion from merged batch+expiry token
                if '/' in t and sum(c.isdigit() for c in t) >= 2:
                    date_match = re.search(r'(\d{1,2}/\d{2,4})', t)
                    if date_match:
                        expiry = date_match.group(1)
                        pref = t[:date_match.start()].strip()
                        if pref:
                            expiry_prefix = pref
                    else:
                        expiry = re.sub(r'[^0-9/]', '', t)
                    continue

                # B. Pack (Must contain a number AND letters like TAB, CAP, or 'S)
                is_pack = False
                if re.search(r'\d', t):
                    if any(x in upper_t for x in ['TAB', 'CAP', 'SYP', 'INJ', 'DROP', 'ML', 'GM', 'LTR']):
                        is_pack = True
                    elif upper_t.endswith("'S") or (upper_t.endswith("S") and len(t) <= 4):
                        is_pack = True
                
                if is_pack:
                    pack = upper_t
                    continue

                # C. Qty (Purely numeric, first one found)
                if re.match(r'^\d+(\.\d+)?$', t) and qty == 0.0:
                    qty = float(t)
                    continue

                clean_tokens.append(t)

            # D. Batch Isolation
            # Priority 1: batch recovered from merged expiry token
            if expiry_prefix and re.search(r'[A-Za-z]', expiry_prefix) and re.search(r'\d', expiry_prefix):
                batch = expiry_prefix
            # Priority 2: check BOTH ends of clean_tokens
            elif clean_tokens:
                # Check last token first (batch often appears after product name)
                if re.search(r'\d', clean_tokens[-1]) or '-' in clean_tokens[-1]:
                    batch = clean_tokens.pop(-1)
                # Fallback: check first token
                elif re.search(r'\d', clean_tokens[0]) or '-' in clean_tokens[0]:
                    batch = clean_tokens.pop(0)

            # E. Brand Name (Whatever words are left)
            brand_name = " ".join(clean_tokens).strip()
            brand_name = re.sub(r'[\$\\\^]+', '', brand_name).strip()

            # F. Dosage Assignment — detect and REMOVE from brand name
            dosage = ""
            dose_words = ["TAB", "CAP", "SYP", "INJ", "DROP", "CREAM", "OINT", "GEL"]
            name_tokens = brand_name.split()
            for d in dose_words:
                if d in [t.upper() for t in name_tokens]:
                    dosage = d
                    name_tokens = [t for t in name_tokens if t.upper() != d]
                    brand_name = " ".join(name_tokens).strip()
                    break

            sgst = float(data["sgst"].replace(',', ''))
            cgst = float(data["cgst"].replace(',', ''))
            tax_amount = sgst + cgst

            extracted_items.append({
                "Brand_Name": brand_name,
                "Dosage": dosage,
                "Packaging": pack,
                "Batch": batch,
                "Expiry": expiry,
                "Qty": qty,
                "Free": 0.0,
                "MRP": float(data["mrp"].replace(',', '')),
                "Rate": float(data["rate"].replace(',', '')),
                "Amount": float(data["amount"].replace(',', '')),
                "Discount Amount": float(data["discount"].replace(',', '')),
                "Tax Amount": tax_amount,
                "Description": ""
            })

        return {
            "AgencyDetails": {
                "Name": seller_name,
                "Address": seller_addr,
                "Phone": seller_phone,
                "GSTIN": seller_gstin
            },
            "ReportDetails": {
                "DetectedFormat": "STANDARD_GST_INVOICE",
                "ParserMode": "INVOICE",
                "FromDate": doc_date,
                "ToDate": doc_date,
                "Company": seller_name,
                "Reason": "Dynamic YAML Match"
            },
            "Areas": [
                {
                    "AreaName": "UNKNOWN AREA",
                    "Stores": [
                        {
                            "StoreName": buyer_name,
                            "StoreLocation": buyer_addr,
                            "Items": extracted_items
                        }
                    ]
                }
            ]
        }