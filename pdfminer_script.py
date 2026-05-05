import os
from pdfminer.high_level import extract_text

# Input PDF path
pdf_path = '1767614848_Mfr___Customer_wise_sales_summary_05012026160119_292.pdf'

# Extract text
text = extract_text(pdf_path)

# Create output txt file name (same name as PDF)
txt_path = os.path.splitext(pdf_path)[0] + '.txt'

# Write text to file
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Text saved to: {txt_path}")