import os
import csv
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Optional pandas import for the Excel summary report
try:
    import pandas as pd
except ImportError:
    pd = None

# ── Clean absolute import from config.py ──
from config import CSV_HEADERS

# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT WRITERS & MATH RECONCILIATION
# ══════════════════════════════════════════════════════════════════════════════

def _is_invoice_adjustment_row(desc: str, store_name: str = "") -> bool:
    text = f"{desc} {store_name}".lower()
    adjustment_keywords = [
        "cgst", "sgst", "igst", "c.g.s.t", "s.g.s.t", "i.g.s.t",
        "add gst", "gst tax", "gst heads", "tax details", "taxable",
        "round off", "roundoff", "tcs", "net payable", "total balance",
        "rupees ", "less c.d", "less cd", "discount", "disc.", "disc ",
    ]
    return any(keyword in text for keyword in adjustment_keywords)


def write_json(data: dict, json_path: str) -> None:
    path = Path(json_path)
    path.write_text(json.dumps(data, indent=4), encoding='utf-8')
    log.info("JSON written → %s", path.resolve())


def _qa_report_path(csv_path: str) -> str:
    base = os.path.splitext(csv_path)[0]
    return base + "_qa_report.csv"

def write_csv(data: dict, csv_path: str, source_filename: str) -> float:
    ag = data.get('AgencyDetails', {})
    rd = data.get('ReportDetails', {})

    ag_name  = ag.get('Name', '')
    ag_addr  = ag.get('Address', '')
    ag_gstin = ag.get('GSTIN', '')
    r_from   = rd.get('FromDate', '')
    r_to     = rd.get('ToDate', '')
    r_co     = rd.get('Company', '')
    
    doc_type = "STATEMENT"
    doc_grand_total = data.get('_grand_total', 0.0)
    
    all_extracted_items = []
    has_gst_values = False
    
    for area in data.get('Areas', []):
        for store in area.get('Stores', []):
            for item in store.get('Items', []):
                all_extracted_items.append(item)
                desc = item.get('Description', '').lower()
                store_name = store.get('StoreName', '').lower()
                if any(k in desc or k in store_name for k in ["cgst", "sgst", "igst", "c.g.s.t"]):
                    has_gst_values = True
                    
    if rd.get("ParserMode") == "PAGED_STORE_INVOICE" or (r_from and r_from == r_to) or has_gst_values:
        doc_type = "INVOICE"

    # ── Round all amounts to 2 decimals to avoid float-drift accumulation ──
    for item in all_extracted_items:
        amt = item.get('Amount', 0.0)
        if isinstance(amt, float):
            item['Amount'] = round(amt, 2)

    # ── Grand Total Row Remover ──
    if len(all_extracted_items) > 1:
        last_item = all_extracted_items[-1]
        sum_of_everything_else = sum(i.get('Amount', 0.0) for i in all_extracted_items[:-1])
        
        if abs(last_item.get('Amount', 0.0) - sum_of_everything_else) <= 2.0:
            log.info("🧮 Math Check: Last row is the Grand Total. Tagging for removal.")
            last_item['IS_GRAND_TOTAL'] = True
            
    valid_items = [i for i in all_extracted_items if not i.get('IS_GRAND_TOTAL')]
    current_csv_sum = round(sum(i.get('Amount', 0.0) for i in valid_items), 2)
    
    # ── Auto-Reconciliation Engine (Rounding Adjustments) ──
    if doc_grand_total > 0 and len(valid_items) > 0:
        difference = round(doc_grand_total - current_csv_sum, 2)
        
        if 0.00 < abs(difference) <= 5.00:
            log.info(f"⚖️ Reconciliation: Minor gap of {difference:.2f} detected. Auto-adjusting highest row.")
            highest_item = max(valid_items, key=lambda x: x.get('Amount', 0.0))
            highest_item['Amount'] = round(highest_item['Amount'] + difference, 2)
            highest_item['Description'] = highest_item['Description'] + " [ROUNDING ADJ]"

    row_count = 0
    total_extracted_value = 0.0
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_HEADERS)

            for area in data.get('Areas', []):
                area_name = area.get('AreaName', 'UNKNOWN')
                for store in area.get('Stores', []):
                    items = store.get('Items', [])
                    if not items: continue
                    store_name = store.get('StoreName', 'UNKNOWN')
                    store_loc  = store.get('StoreLocation', '')
                    
                    for item in items:
                        if item.get('IS_GRAND_TOTAL'):
                            continue

                        amt = item.get('Amount', 0.0)
                        desc_lower = item.get('Description', '').lower()
                        store_lower = store_name.lower()

                        if doc_type == "INVOICE" and _is_invoice_adjustment_row(desc_lower, store_lower):
                            continue
                        
                        if any(kw in desc_lower for kw in ["grand total", "grandtotal", "g.total", "total amount", "page total", "net amount"]):
                            continue

                        if amt > 0 and amt == doc_grand_total and len(valid_items) > 1:
                            log.info("🚫 Dropping row '%s' because amount %.2f matches the Document Grand Total.", item.get('Description', ''), amt)
                            continue
                        
                        tax_amt = 0.0
                        disc_amt = 0.0
                        
                        if any(k in desc_lower or k in store_lower for k in ["cgst", "sgst", "igst", "c.g.s.t", "s.g.s.t", "i.g.s.t"]):
                            tax_amt = amt
                        elif any(k in desc_lower or k in store_lower for k in ["discount", "disc.", "less", "disc "]):
                            disc_amt = amt
                            
                        total_extracted_value += round(amt, 2)
                            
                        parser_format = data.get("ReportDetails", {}).get("DetectedFormat", "")
                        qa_flags = item.get('qa_flags', [])
                        if isinstance(qa_flags, list):
                            qa_flags_str = '; '.join(qa_flags)
                        else:
                            qa_flags_str = str(qa_flags) if qa_flags else ''
                        writer.writerow([
                            ag_name, ag_addr, ag_gstin,
                            r_from, r_to, r_co,
                            area_name, store_name, store_loc,
                            item.get('Description', ''),
                            item.get('Brand_Name', ''), 
                            item.get('Dosage', ''),     
                            item.get('Packaging', ''),  
                            item.get('Qty', 0),
                            item.get('Free', 0),
                            item.get('Rate', 0.0),
                            amt,
                            item.get('Percent', 0.0),
                            tax_amt, disc_amt, doc_type,
                            # ── FORMAT_27 Sales Book columns (blank for other formats) ──
                            item.get('Bill_Date', ''),
                            item.get('Bill_No', ''),
                            item.get('Taxable', ''),
                            item.get('Tax', ''),
                            item.get('Sur_Tax', ''),
                            item.get('Free_Amt', ''),
                            item.get('Exempted', ''),
                            item.get('Round_Off', ''),
                            source_filename,
                            parser_format,
                            qa_flags_str
                        ])
                        row_count += 1

    except PermissionError:
        log.error("Cannot write '%s'. Close Excel and try again.", csv_path)
        return 0.0

    # ── QA sidecar report ──
    try:
        qa_path = _qa_report_path(csv_path)
        flagged_items = []
        for area in data.get("Areas", []):
            for store in area.get("Stores", []):
                for item in store.get("Items", []):
                    if item.get('qa_flags'):
                        row = {h: '' for h in CSV_HEADERS}
                        row['Store Name'] = store.get('StoreName', '')
                        row['Store Location'] = store.get('StoreLocation', '')
                        row['Description'] = item.get('Description', '')
                        row['Amount'] = item.get('Amount', 0)
                        row['Qty'] = item.get('Qty', 0)
                        qf = item.get('qa_flags', [])
                        row['QA Flags'] = '; '.join(qf) if isinstance(qf, list) else str(qf)
                        flagged_items.append(row)
        if flagged_items:
            with open(qa_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(flagged_items)
            log.info("QA report written → %s  (%d flagged items)", qa_path, len(flagged_items))
    except Exception as e:
        log.warning("Could not write QA sidecar report: %s", e)

    log.info("CSV written → %s  (%d data rows)", Path(csv_path).resolve(), row_count)
    return total_extracted_value


def generate_summary_report(summary_records: list, output_dir: str = ".") -> None:
    if not summary_records:
        log.info("✨ No files were processed successfully to generate a report.")
        return
        
    report_path_xlsx = os.path.join(output_dir, "Extraction_Summary_Report.xlsx")
    report_path_csv = os.path.join(output_dir, "Extraction_Summary_Report.csv")
    
    try:
        if pd is not None:
            df = pd.DataFrame(summary_records)
            df.to_excel(report_path_xlsx, index=False, sheet_name="Master Summary")
            log.info("📊 Master summary report generated successfully: %s", report_path_xlsx)
            return
    except Exception as e:
        log.warning("⚠️ Could not generate Excel report (%s). Falling back to CSV...", e)
        
    try:
        with open(report_path_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["File Name", "Status", "Actual Amount", "CSV Amount", "Missing Gap Amount", "Remarks"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_records)
        log.info("📊 Master summary report generated (CSV fallback): %s", report_path_csv)
    except Exception as e:
        log.error("❌ Failed to write master summary report: %s", e)