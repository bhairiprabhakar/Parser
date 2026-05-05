import os
import re
import logging
from config import _KW_PATTERN_REGEX, _CORP_SUFFIX_REGEX

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC BRAND & CORPORATE SCRUBBER (STRICT FILE MODE)
# ══════════════════════════════════════════════════════════════════════════════

def load_dynamic_brands(filename="brand_keywords.txt"):
    # Target the root directory of the pipeline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) 
    filepath = os.path.join(root_dir, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            brands = [re.escape(line.strip()) for line in f if line.strip()]
            
        if brands:
            brands.sort(key=len, reverse=True)
            return r'(?i)\b(?:' + '|'.join(brands) + r')\b'
            
    except FileNotFoundError:
        log.warning("⚠️ '%s' not found in %s! Brand scrubbing will be skipped.", filename, root_dir)
        
    return None

# Compile the massive regex string ONCE at startup to keep the script ultra-fast
_DYNAMIC_BRAND_REGEX = load_dynamic_brands()

def parse_store_and_location(name_str: str) -> tuple:
    name_str = re.sub(r'\[.*?\]', '', name_str)
    name_str = name_str.strip('- ')
    
    name_str = re.sub(r'([A-Za-z])(MEDICALS?|PHARMACY|PHARMA|MEDICOS?|AGENCIES|AGENCY|CLINIC|HOSPITAL|DRUGS?|CHEMISTS?)\b', r'\1 \2', name_str, flags=re.IGNORECASE)
    
    name_str = re.sub(_CORP_SUFFIX_REGEX, '', name_str)
    name_str = re.sub(r'\s{2,}', ' ', name_str).strip()
    
    def clean_loc(loc: str) -> str:
        if _DYNAMIC_BRAND_REGEX:
            loc = re.sub(_DYNAMIC_BRAND_REGEX, '', loc)
        loc = re.sub(_CORP_SUFFIX_REGEX, '', loc)
        return loc.strip(' -.,()')

    if "," in name_str:
        parts = name_str.split(",", 1)
        s_name = re.sub(r'\.+', '.', parts[0]).strip()
        s_loc = clean_loc(parts[1])
        return s_name, s_loc

    if re.match(r'(?i)^(?:DR\.?\s+|DOCTOR\b)', name_str):
        degree_pattern = r'(?i)\b(?:M\s*B\s*B\s*S|M\.?\s*D\.?|M\.?\s*S\.?|B\.?\s*D\.?\s*S\.?|B\.?\s*A\.?\s*M\.?\s*S\.?|B\.?\s*H\.?\s*M\.?\s*S\.?)\b'
        matches = list(re.finditer(degree_pattern, name_str))
        if matches:
            last_match = matches[-1]
            doc_name = name_str[:last_match.end()].strip()
            location = clean_loc(name_str[last_match.end():])
            if not re.search(r'[A-Za-z]', location) or len(location) < 3:
                location = ""
            return doc_name, location
        else:
            parts = name_str.rsplit(' ', 1)
            if len(parts) == 2 and len(parts[1]) >= 3:
                return parts[0].strip(), clean_loc(parts[1])
            return name_str.strip(), ""

    matches = list(re.finditer(_KW_PATTERN_REGEX, name_str, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        store_name = name_str[:last_match.end()].strip()
        location = clean_loc(name_str[last_match.end():])
        if not re.search(r'[A-Za-z]', location) or len(location) < 3:
            location = ""
        return store_name, location
        
    if '-' in name_str:
        parts = name_str.rsplit('-', 1)
        sname = parts[0].strip()
        sloc = clean_loc(parts[1])
        if re.search(r'[A-Za-z]', sloc) and len(sloc) >= 3:
            return sname, sloc

    return name_str.strip(), ""