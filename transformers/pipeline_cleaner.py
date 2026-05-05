import re
from config import _KW_PATTERN_REGEX, _AGENCY_PATTERN_REGEX
from transformers.entity_scrubber import parse_store_and_location
from transformers.item_parser import parse_item_description

# ══════════════════════════════════════════════════════════════════════════════
#  STRICT ENTITY CLEANING WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_strict_entity_cleaning(text: str, is_location: bool = False) -> str:
    if not text:
        return ""
    
    cleaned = str(text).upper()
    cleaned = re.sub(r'\b\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}\b', ' ', cleaned)
    
    if is_location:
        cleaned = re.sub(r'[^A-Z\s]', ' ', cleaned)
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
#  POST-PROCESSING & REPARENTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def post_process_extracted_data(data: dict) -> dict:
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
            s_name = store.get("StoreName", "")
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

    return data