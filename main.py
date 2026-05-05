import os
import sys
import glob
import time
import logging
from pathlib import Path
import re

# ── Force Python to recognize the current folder ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging Configuration ──
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Pipeline Phase Imports (Absolute) ──
from extractors.text_extractor import extract_raw_text
from parsers.universal_router import route_and_parse
from transformers.pipeline_cleaner import post_process_extracted_data, safe_clean_store_entities
from transformers.cleaners import clean_number, is_numeric_token
from loaders.writers import write_json, write_csv, generate_summary_report

def process_document(filepath: str, output_dir: str = ".") -> dict:
    """
    The Single Entry Point for the Pipeline.
    Can be called directly from your Dashboard backend.
    """
    total_t0 = time.time()
    target_file = Path(filepath)
    
    if not target_file.exists():
        log.error("File not found: %s", target_file.resolve())
        return {
            "File Name": target_file.name,
            "Status": "File Not Found",
            "Actual Amount": 0.0,
            "CSV Amount": 0.0,
            "Missing Gap Amount": 0.0,
            "Remarks": "File does not exist"
        }

    log.info("=" * 60)
    
    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────
    log.info("Step 1/4  —  Text extraction  [%s]", target_file.name)
    ext_t0 = time.time()
    try:
        raw_text = extract_raw_text(str(target_file))
    except Exception as e:
        log.error("Failed to extract text from %s: %s", target_file.name, e)
        return {
            "File Name": target_file.name,
            "Status": "Extraction Error",
            "Actual Amount": 0.0,
            "CSV Amount": 0.0,
            "Missing Gap Amount": 0.0,
            "Remarks": str(e)
        }
    log.info("⏱️ Extraction Time: %.2f seconds", time.time() - ext_t0)

    # Grab the expected grand total from raw text (Math Reconciliation Shield)
    import re
    expected_grand_total = None
    for line in raw_text.split('\n'):
        if any(lbl in line.lower() for lbl in ['grand total', 'total value', 'net sales', 'value in rs', 'net amount', 'total:', 'total :', 'invoice value', 'net payable']) or re.search(r'(?i)^\s*[\d.,]+\s+total\s*$', line) or re.search(r'(?i)^\s*total\s+[\d.,]+\s*$', line):
            nums = [clean_number(t) for t in line.split() if is_numeric_token(t)]
            if nums:
                max_num = max(nums)
                if expected_grand_total is None or max_num > expected_grand_total:
                    expected_grand_total = max_num

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: PARSING (Schema Inference + Legacy State Machine)
    # ──────────────────────────────────────────────────────────────────────────
    log.info("Step 2/4  —  Parsing document structure …")
    parse_t0 = time.time()
    
    data = route_and_parse(raw_text, str(target_file))
    
    rd = data["ReportDetails"]
    if rd.get("ParserMode") == "PAGED_STORE_INVOICE" and data.get("_grand_total", 0.0) > 0:
        expected_grand_total = data["_grand_total"]

    if rd.get("FromDate") and not rd.get("ToDate"):
        log.info("📄 This is not a statement, it is an INVOICE copy because it doesn't have a From Date to To Date range.")
        rd["ToDate"] = rd["FromDate"]

    log.info("⏱️ Parsing/Logic Time: %.2f seconds", time.time() - parse_t0)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3: TRANSFORM & CLEAN (NLP Sanitization)
    # ──────────────────────────────────────────────────────────────────────────
    log.info("Step 2.5/4 — Post-processing and sanitizing extracted data ...")
    data = post_process_extracted_data(data)
    data = safe_clean_store_entities(data)

    area_count  = len(data.get("Areas", []))
    store_count = sum(len(a.get("Stores", [])) for a in data.get("Areas", []))
    item_count  = sum(len(s.get("Items", [])) for a in data.get("Areas", []) for s in a.get("Stores", []))
    log.info("Parsed  →  %d areas,  %d stores,  %d items", area_count, store_count, item_count)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3.5: UNRECOGNIZED FORMAT INVESTIGATOR (RAW TEXT DUMP)
    # ──────────────────────────────────────────────────────────────────────────
    detected_format = data.get("ReportDetails", {}).get("DetectedFormat", "")
    
    # Trigger the dump if 0 items are extracted OR if it hits the absolute fallback
    if item_count == 0 or detected_format == "LEGACY_UNIVERSAL_TWO_COLUMN":
        log.warning("⚠️ Unrecognized Format or 0 items extracted! Triggering Investigation Dump...")
        
        dump_dir = os.path.join(output_dir, "investigation_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(dump_dir, f"{target_file.stem}_RAW_TEXT.txt")
        
        try:
            with open(dump_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            log.info("🕵️‍♂️ Raw pdfplumber text saved for investigation: %s", dump_path)
        except Exception as e:
            log.error("Could not save raw text dump: %s", e)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4: LOAD & EXPORT (CSV / JSON Output)
    # ──────────────────────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, target_file.with_suffix('.json').name)
    csv_path = os.path.join(output_dir, target_file.with_suffix('.csv').name)

    write_t0 = time.time()
    log.info("Step 3/4  —  Writing JSON …")
    write_json(data, json_path)

    log.info("Step 4/4  —  Writing CSV …")
    total_val = write_csv(data, csv_path, target_file.name)
    log.info("⏱️ Writing/Updation Time: %.4f seconds", time.time() - write_t0)

    # ──────────────────────────────────────────────────────────────────────────
    # RECONCILIATION SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("✅  Done. %d rows extracted in ⏱️ %.2f seconds total.", item_count, time.time() - total_t0)
    log.info("💰  CSV TOTAL EXTRACTED: %s", f"{total_val:,.2f}")
    
    summary_record = {
        "File Name": target_file.name,
        "Status": "Processed",
        "Actual Amount": expected_grand_total if expected_grand_total is not None else 0.0,
        "CSV Amount": total_val,
        "Missing Gap Amount": 0.0,
        "Remarks": ""
    }

    if expected_grand_total is not None and expected_grand_total > 0:
        diff = abs(total_val - expected_grand_total)
        summary_record["Missing Gap Amount"] = diff
        
        if diff < 1.00:
            log.info("🎯  100%% MATCH! Extracted total perfectly matches Document Grand Total (%s)!", f"{expected_grand_total:,.2f}")
            summary_record["Status"] = "Perfect Match"
            summary_record["Remarks"] = "Values matching"
        else:
            log.warning("⚠️  Document Grand Total is %s, but script extracted %s. Missing Gap: %s",
                        f"{expected_grand_total:,.2f}", f"{total_val:,.2f}", f"{diff:,.2f}")
            summary_record["Status"] = "Mismatch Detected"
            summary_record["Remarks"] = f"Check final amount / Gap found: {diff:.2f}"
    else:
        log.info("ℹ️  Could not auto-detect 'GRAND TOTAL' in the Document. Please verify manually.")
        summary_record["Status"] = "Grand Total Not Found"
        summary_record["Remarks"] = "Unable to auto-verify Grand Total"

    return summary_record


if __name__ == "__main__":
    # ORIGINAL CLI / BULK RUNNER LOGIC PRESERVED
    if len(sys.argv) > 1:
        potential_file = " ".join(sys.argv[1:])
        if os.path.exists(potential_file):
            paths_to_process = [potential_file]
        else:
            paths_to_process = sys.argv[1:]
            
        files = []
        for p in paths_to_process:
            path_obj = Path(p)
            if path_obj.is_dir():
                files.extend([f for f in path_obj.glob('*') if f.suffix.lower() in ['.pdf', '.csv', '.xlsx', '.xls', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
            elif path_obj.is_file():
                files.append(path_obj)
        
        if not files:
            log.error("No valid supported files found in the provided path(s).")
            sys.exit(1)
            
        log.info("📁 BULK MODE: Found %d files to process.", len(files))
        all_records = []
        for f in files:
            record = process_document(str(f))
            if record:
                all_records.append(record)
            log.info("\n" + "#" * 80 + "\n")
            
        generate_summary_report(all_records)
        log.info("✅ Bulk Run Complete! Processed %d total files.", len(all_records))
            
    else:
        print("\n" + "="*60)
        print("🚀 MARG ERP EXTRACTION PIPELINE - TESTING/INTERACTIVE MODE")
        print("="*60)
        print("Please select an option:")
        print("1. Process a Single File (from current or specific path)")
        print("2. Process a Folder (Bulk Upload from directory)")
        
        choice = input("\nEnter 1 or 2: ").strip()
        
        if choice == '1':
            filepath = input("Enter the full file name or path (e.g., test.pdf): ").strip()
            if not os.path.exists(filepath):
                log.error("❌ File not found: %s", filepath)
            else:
                log.info("➡️ Processing single file: %s", filepath)
                record = process_document(filepath, output_dir=os.path.dirname(os.path.abspath(filepath)))
                if record:
                    generate_summary_report([record], os.path.dirname(os.path.abspath(filepath)))
                    log.info("✅ Single file processing complete.")
                    
        elif choice == '2':
            INPUT_DIR = input("Enter folder path (Press Enter for default './input_files'): ").strip()
            if not INPUT_DIR:
                INPUT_DIR = "./input_files"
                
            os.makedirs(INPUT_DIR, exist_ok=True)
            log.info("🚀 Starting Bulk Multi-Format Extraction Pipeline from '%s'...\n", INPUT_DIR)
            
            files = glob.glob(f"{INPUT_DIR}/*.*")
            if not files:
                log.warning("⚠️ No files found in '%s'. Please add files and run again.", INPUT_DIR)
            else:
                supported_extensions = ['pdf', 'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png', 'bmp', 'tiff']
                valid_files = [f for f in files if f.lower().split('.')[-1] in supported_extensions]
                
                if not valid_files:
                    log.warning("⚠️ No valid supported files found in '%s'.", INPUT_DIR)
                else:
                    total_files = len(valid_files)
                    log.info("📁 BULK MODE: Found %d files to process.", total_files)
                    all_records = []
                    for idx, filepath in enumerate(valid_files, start=1):
                        log.info("➡️ Processing file %d of %d...", idx, total_files)
                        try:
                            record = process_document(filepath, output_dir=INPUT_DIR)
                            if record:
                                all_records.append(record)
                            log.info("\n" + "#" * 80 + "\n")
                        except Exception as e:
                            log.error("❌ Fatal loop error processing %s: %s", os.path.basename(filepath), str(e))
                            all_records.append({
                                "File Name": os.path.basename(filepath),
                                "Status": "Crash",
                                "Actual Amount": 0.0,
                                "CSV Amount": 0.0,
                                "Missing Gap Amount": 0.0,
                                "Remarks": str(e)
                            })
                    
                    generate_summary_report(all_records, INPUT_DIR)
                    log.info("✅ Bulk Run Complete! Processed %d total files.", len(all_records))
        else:
            log.error("Invalid choice. Please run the script again and select 1 or 2.")