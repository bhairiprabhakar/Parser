import re

# ══════════════════════════════════════════════════════════════════════════════
#  PRE-PROCESSING & OCR CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

_CID_RE  = re.compile(r'\(cid:\d+\)')
_CTRL_RE = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]')

def preprocess_line(line: str) -> str:
    line = _CID_RE.sub(' ', line)          
    line = _CTRL_RE.sub('', line)  
    
    line = re.sub(r'={2,}', ' ', line)
    line = re.sub(r'-{3,}', ' ', line)
    
    line = re.sub(r'([A-Za-z])(\d{1,2}[-/]\d{2,4})\b', r'\1 \2', line)
    line = re.sub(r'([A-Za-z])(\d+[\.,]\d{1,2})\b', r'\1 \2', line)
    line = re.sub(r'\b(\d+[\.,]\d{1,2})([A-Za-z])', r'\1 \2', line)
    line = re.sub(r'(\d+\.\d{1,2})[OoQq]\b', r'\g<1>0', line)
    
    line = re.sub(r'\b(comer|custmer|cstmer|custmr|cstomer)\b', 'CUSTOMER', line, flags=re.IGNORECASE)
             
    return line.strip()

# ══════════════════════════════════════════════════════════════════════════════
#  NUMBER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def clean_number(val: str, is_int: bool = False):
    val = str(val).strip()
    
    if re.search(r'\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b', val):
        return 0 if is_int else 0.0
        
    if val.endswith(('-', '−')): val = '-' + val[:-1]
    val = (val.replace('Cr', '').replace('Dr', '')
              .replace('CR', '').replace('DR', '')
              .replace('(', '-').replace(')', ''))
    val = re.sub(r'(?<=\d),(?=\d)', '', val)

    cleaned = re.sub(r'[^\d.\-]', '', val)
    if cleaned in ('-', '', '.', '-.'): return 0 if is_int else 0.0

    try: 
        num = int(float(cleaned)) if is_int else float(cleaned)
        if abs(num) > 999_999_999.0: 
            return 0 if is_int else 0.0
        # Round monetary amounts to 2 decimal places to prevent float-drift
        return round(num, 2) if not is_int else num
    except ValueError: 
        return 0 if is_int else 0.0


def is_numeric_token(val: str) -> bool:
    val = str(val).strip()
    if not val: return False

    if re.search(r'\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b', val):
        return False
    
    clean_test = val.upper().replace('CR', '').replace('DR', '')
    if re.search(r'[A-Z\*]', clean_test):
        return False

    if val.endswith(('-', '−')): val = '-' + val[:-1]
    val = (val.replace('Cr', '').replace('Dr', '')
              .replace('CR', '').replace('DR', '')
              .replace('(', '-').replace(')', ''))
    val = re.sub(r'(?<=\d),(?=\d)', '', val)
               
    cleaned = re.sub(r'[^\d.\-]', '', val)
    if cleaned in ('-', '', '.', '-.'): return False
    
    if len(cleaned.split('.')[0].replace('-', '')) > 12:
        return False
        
    try:
        float(cleaned)
        return True
    except ValueError:
        return False