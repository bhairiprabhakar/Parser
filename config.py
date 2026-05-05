import re

# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC ENTITY PATTERN GENERATOR (LEGO-BLOCK REGEX)
# ══════════════════════════════════════════════════════════════════════════════

# BLOCK 1: The Core Medical Words
_ROOTS = r"(?:MEDICALS?|MEDICAAL|MEDICAL\s*&|MEDICINES?|MEDICOS?E?|MEDICARE|MEDICATES?|MEDCLS?|MEDI|MED\.?|PHARMACY|PHARMAA?|PHRM|PHARMACEUTICALS?|DRUGS?|DRUGGISTS?|CHEMISTS?|AYURVEDIC|SURGICALS?|VETERINARY|VET|DENTAL|AUSHADHI|AUSHADHALAYA|LIFE\s*CARE|HEALTH\s*CARE|CLINIC|HOSPITAL|POLYCLINIC|DAWAKHANA)"

# BLOCK 2: Agency Roots
_AGENCY_ROOTS = r"(?:AGENC(?:Y|IES)|COMPANY|CO\.?|TRADERS?|TRADING|DISTRIBUTORS?|DISTRIBUTERS?|DISTRIBUTION|DIST\.?|SUPPLIERS?|SUPPLY|ENTERPRISES?|CORPORATION|CORP\.?|SONS|BROTHERS|BROS|TRUST|FOUNDATION|INSTITUTE|ASSOCIATES?|SERVICES?)"

# BLOCK 3: The Middle Connectors 
_MODIFIERS = r"(?:\s*(?:&|AND|OR|\\+|-)?\s*(?:GENERAL|GENARAL|GENREAL|GENRAL|GEN\.?|FANCY|FANCEY|FANC|PROVISION|PROV|SUPER\s*SPECIALITY))?"

# BLOCK 4 & 5: Suffixes
_SUFFIXES_OPTIONAL = r"(?:\s*(?:STORES?|STORTE?|STORE?S?|STOR|STO|ST|SHOPPES?|SHOP|MART|POINT|CENT(?:ER|RE)|DEPOT|CORNER|HUB|NETWORK|HALL|BHANDAR|KENDRA|PALACE))?"
_SUFFIXES_REQUIRED = r"(?:STORES?|STORTE?|STORE?S?|STOR|STO|ST|SHOPPES?|SHOP|MART|POINT|CENT(?:ER|RE)|DEPOT|CORNER|HUB|NETWORK|HALL|BHANDAR|KENDRA|PALACE)"

# BLOCK 6: Trailing Punctuation Eater
_PUNCT_SUFFIX = r"(?:[';/sS]+|/S|'S|;S|S)?"

# COMBINED SUPER-REGEX 
_CORE = r"(?:" + _ROOTS + _MODIFIERS + _SUFFIXES_OPTIONAL + r"|" + _AGENCY_ROOTS + r"|" + _SUFFIXES_REQUIRED + r")"
_KW_PATTERN_REGEX = r'(?i)\b(' + _CORE + _PUNCT_SUFFIX + r')(?:\b|(?=[A-Z0-9]))'

# AGENCY PATTERNS
_AGENCY_PREFIX = r"(?:PHARMA\s+|PHARMACEUTICALS\s+)?"
_AGENCY_PATTERN_REGEX = r'(?i)\b' + _AGENCY_PREFIX + _AGENCY_ROOTS + r'\b(?:(?:\s+(?:PVT|PRIVATE)\s+LTD)|(?:\s+LTD))?\b'

# UNIVERSAL CORPORATE SUFFIXES
_CORP_SUFFIX_REGEX = r'(?i)\b(?:LTD\.?|PVT\.?|LIMITED|PRIVATE|COMPANY|CO\.?|GEN|GENERICS?|LIFE\s*SCIENCES?|PHARMA(?:CEUTICALS?)?|INC\.?|CORP(?:ORATION)?\.?)\b'


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_SKIP_HEADERS = frozenset([
    "continued..", "page no", "report for", "company :", "company:",
    "d e s c r i", "description", "shop no", "area / item", "summary from",
    "party/item", "party / item", "sale summary", "party wise sale",
    "name of the party", "party name", "amt.october", "amt.november", "amt.december", 
    "party/customer", "party / customer", "free amt", "packs", "party-productwise",
    "party-productwise sales", "party wise sales book", "amount name", "amount party",
    "sales analysis", "all parties", "account details", "product name", "inv/dm no",
    "company wise customer sales", "netamount", "town name", "name town", "net amount",
    "amount n a m e", "marg erp", "computerise your", "www.margcompusoft.com",
    "customer name", "goods value", "gst amount", "proddisc amt", "cdis amt"
])

CSV_HEADERS = [
    'Agency Name', 'Agency Address', 'GSTIN',
    'From Date', 'To Date', 'Company',
    'Area', 'Store Name', 'Store Location', 'Description',
    'Brand Name', 'Dosage', 'Packaging', 
    'Qty', 'Free', 'Rate', 'Amount', 'Percent',
    'Tax Amount', 'Discount Amount', 'Doc Type',
    'Bill Date', 'Bill No', 'Taxable', 'Tax', 'Sur Tax', 'Free Amt', 'Exempted', 'Round Off',
    'Source File'
]