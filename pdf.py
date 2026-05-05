import pdf
import os

pdf_path = '1760512198_RELIABO Sep1.pdf'

text = ""

# Extract text page by page
with pdf.open(pdf_path) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

# Save as .txt with same name
txt_path = os.path.splitext(pdf_path)[0] + '.txt'

with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Saved: {txt_path}")