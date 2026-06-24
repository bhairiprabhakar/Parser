import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class DynamicGSTInvoiceParser:
    def __init__(self):
        # ULTIMATE REGEX: Decouples the messy front part from the highly structured back part
        self.line_item_pattern = re.compile(
            r"(?P<sno>\d+)\.\s*"                 
            r"(?P<front_text>.+?)\s+"            # Captures Qty, Pack, Batch, Product dynamically
            r"(?P<hsn>\d{2,8})\s+"               # The anchor: 7 trailing numbers begin here
            r"(?P<mrp>[\d\.,]+)\s+"                   
            r"(?P<rate>[\d\.,]+)\s+"                  
            r"(?P<discount>[\d\.,]+)\s+"              
            r"(?P<sgst>[\d\.,]+)\s+"                  
            r"(?P<cgst>[\d\.,]+)\s+"                  
            r"(?P<amount>[\d\.,]+)"
        )

    def extract(self, raw_lines: List[str]) -> Dict:
        # 1. Flatten to destroy hidden newlines and bad layout wraps
        raw_text = " ".join([line.strip() for line in raw_lines if line.strip()])
        raw_text = re.sub(r'\s+', ' ', raw_text)  # Normalize spaces
        
        extracted_items = []
        
        # 2. Iterate over the flattened text using the Head/Tail pattern
        for match in self.line_item_pattern.finditer(raw_text):
            data = match.groupdict()
            
            front_text = data["front_text"].strip()
            tokens = front_text.split()
            
            qty = 0.0
            pack = ""
            expiry = ""
            batch = ""

            # SMART TOKEN HUNTING (Order Independent)

            # A. Hunt for Expiry (Look for the slash '/')
            for t in tokens[:4]:
                if '/' in t:
                    expiry = re.sub(r'[^0-9/]', '', t) # Cleans OCR noise like $7/27$
                    tokens.remove(t)
                    break

            # B. Hunt for Qty (The first purely numeric token)
            for t in tokens[:3]:
                if re.match(r'^\d+(\.\d+)?$', t):
                    qty = float(t)
                    tokens.remove(t)
                    break

            # C. Hunt for Pack (Token with letters/symbols like '10TAB', '10'S')
            for t in list(tokens[:2]):
                if re.search(r'[A-Za-z\']', t.upper()):
                    pack = t
                    tokens.remove(t)
                    break

            # D. Hunt for Batch (Usually alphanumeric/hyphens before product name)
            if tokens and (re.search(r'\d', tokens[0]) or '-' in tokens[0] or len(tokens[0]) > 4):
                batch = tokens.pop(0)
                
            # E. The remaining tokens make up the Product Name
            product_name = " ".join(tokens).strip()
            
            # 3. Clean up commas from numbers and append
            extracted_items.append({
                "Product": product_name,
                "Batch": batch,
                "Expiry": expiry,
                "Qty": qty,
                "MRP": float(data["mrp"].replace(',', '')),
                "PTR": float(data["rate"].replace(',', '')),
                "HSN": data["hsn"],
                "Amount": float(data["amount"].replace(',', ''))
            })
            
        # 4. Extract Seller and Buyer dynamically
        seller_name = "UNKNOWN_SELLER"
        buyer_name = "UNKNOWN_BUYER"
        
        seller_match = re.search(r"([A-Z][A-Za-z0-9\s\.\-]+?(?:AGENCY|DISTRIBUTORS?|TRADERS?))", raw_text)
        if seller_match:
            seller_name = seller_match.group(1).strip()
            
        buyer_match = re.search(r"(?:M/S\.?|M/s)\s*([A-Z][A-Za-z0-9\s\.\-]+?(?:PHARMACY|MEDICAL STORE|CLINIC|HOSPITAL))", raw_text, re.IGNORECASE)
        if buyer_match:
            buyer_name = buyer_match.group(1).strip()

        # 5. Standardize Output
        return {
            "ReportDetails": {
                "Format": "STANDARD_GST_INVOICE",
                "ReportType": "INVOICE",
                "Reason": "Dynamic YAML Match"
            },
            "status": "success",
            "AgencyName": seller_name,
            "Areas": [
                {
                    "Area": "UNKNOWN AREA", 
                    "Stores": [
                        {
                            "StoreName": buyer_name,
                            "Items": extracted_items
                        }
                    ]
                }
            ]
        }