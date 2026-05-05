import re

# ══════════════════════════════════════════════════════════════════════════════
#  ITEM DESCRIPTION PARSER (WITH INCH/SYMBOL FIX)
# ══════════════════════════════════════════════════════════════════════════════
def parse_item_description(raw_desc: str) -> dict:
    if not raw_desc or not isinstance(raw_desc, str):
        return {"Brand_Name": "", "Dosage": "", "Packaging": "", "Original_OCR": ""}

    pack_pattern = r'\b(\d+\s*[xX\*]\s*\d+[A-Z]*|\d+\'?[sS])\b'
    pack_match = re.search(pack_pattern, raw_desc)
    packaging = pack_match.group(1).upper() if pack_match else ""

    dose_pattern = r'\b(\d+(?:\.\d+)?\s*(?:MG|ML|GM|MCG|TAB|CAP|INJ|SYP|DROP|IU|STRIP|CM|INCH|MTR|MM|"|\'\'?|”|″))(?=\s|$|\W)'
    dose_match = re.search(dose_pattern, raw_desc, re.IGNORECASE)
    dosage = dose_match.group(1).upper() if dose_match else ""

    brand_name = raw_desc
    if pack_match:
        brand_name = brand_name.replace(pack_match.group(0), '')
    if dose_match:
        brand_name = brand_name.replace(dose_match.group(0), '')

    brand_name = re.sub(r'\b\d+\s*(?:"|\'\'?|”|″|INCH|CM|MM)', ' ', brand_name, flags=re.IGNORECASE)

    brand_name = re.sub(r'\b\d+\b', ' ', brand_name) 
    brand_name = re.sub(r'[^A-Za-z0-9\s]', ' ', brand_name) 
    brand_name = re.sub(r'\s{2,}', ' ', brand_name).strip().upper() 

    return {
        "Brand_Name": brand_name,
        "Dosage": dosage,
        "Packaging": packaging,
        "Original_OCR": raw_desc
    }