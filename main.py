import os
import sys
import glob
import time
import logging
from pathlib import Path
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

from extractors.text_extractor import extract_raw_text
from parsers.universal_router import route_and_parse, _AI_AVAILABLE, _ai_format_inference
from parsers.erp_template_engine import ERPTemplateEngine
from transformers.pipeline_cleaner import post_process_extracted_data, safe_clean_store_entities
from transformers.cleaners import clean_number, is_numeric_token
from loaders.writers import write_json, write_csv, generate_summary_report

_GRAND_TOTAL_LINE_RE = re.compile(
    r'(?i)(grand\s*total|total\s*value|net\s*sales|value\s*in\s*rs|'
    r'net\s*amount|total\s*[:.]|invoice\s*value|net\s*payable|'
    r'total\s+[\d.,]+\s*$|^\s*[\d.,]+\s+total\s*$)'
)


def _make_empty_data() -> dict:
    return {
        "AgencyDetails": {"Name": "", "Address": "", "Phone": "", "Email": "", "GSTIN": "", "TIN": ""},
        "ReportDetails": {"FromDate": "", "ToDate": "", "Company": "", "DetectedFormat": "", "ParserMode": ""},
        "Areas": [],
    }


def _new_area(name: str) -> dict:
    return {"AreaName": name, "Stores": []}


def _new_store(name: str, location: str = "") -> dict:
    return {"StoreName": name, "StoreLocation": location, "Items": []}


def process_document(filepath: str, output_dir: str = ".") -> dict:
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
    log.info("Step 1/4  —  Text extraction  [%s]", target_file.name)
    ext_t0 = time.time()

    try:
        raw_text = extract_raw_text(str(target_file))
    except Exception as exc:
        log.error("Failed to extract text from %s: %s", target_file.name, exc)
        return {
            "File Name": target_file.name,
            "Status": "Extraction Error",
            "Actual Amount": 0.0,
            "CSV Amount": 0.0,
            "Missing Gap Amount": 0.0,
            "Remarks": str(exc)
        }

    log.info("Extraction Time: %.2f seconds", time.time() - ext_t0)

    if target_file.suffix.lower() != '.txt':
        try:
            raw_text_path = target_file.parent / f"{target_file.stem}_raw_extracted.txt"
            with open(raw_text_path, 'w', encoding='utf-8') as fh:
                fh.write(raw_text)
            log.info("OCR/Raw text saved to %s", raw_text_path.name)
        except Exception as exc:
            log.warning("Failed to save raw text file: %s", exc)

    # MultiPageContinuityPass — runs before grand-total scan and parsing
    log.info("Step 1.5/4 — Running MultiPageContinuityPass ...")
    try:
        from image_processing.continuity_pass import MultiPageContinuityPass
        _continuity = MultiPageContinuityPass()
        raw_text = _continuity.process_pages(raw_text.split('\x0c'))
    except Exception as exc:
        log.debug("MultiPageContinuityPass not available: %s", exc)

    # Grand-total scan (on continuity-cleaned text)
    expected_grand_total = None
    for line in raw_text.split('\n'):
        if _GRAND_TOTAL_LINE_RE.search(line):
            nums = [clean_number(t) for t in line.split() if is_numeric_token(t)]
            if nums:
                max_num = max(nums)
                if expected_grand_total is None or max_num > expected_grand_total:
                    expected_grand_total = max_num

    log.info("Step 2/4  —  Parsing document structure ...")
    parse_t0 = time.time()

    # ERPTemplateEngine detection — try pluggable schemas first
    _erp_template_engine = None
    try:
        _erp_template_engine = ERPTemplateEngine.instance()
    except Exception:
        pass

    lines_sample = raw_text.split('\n')
    _erp_schema = _erp_template_engine.detect(lines_sample[:40]) if _erp_template_engine else None

    if _erp_schema:
        log.info("ERPTemplateEngine matched schema: %s / %s — using template parser",
                 _erp_schema.erp_name, _erp_schema.format_id)
        erp_rows = _erp_template_engine.extract_rows(lines_sample, _erp_schema, skip_header_lines=0)
        data = _make_empty_data()
        data["ReportDetails"]["DetectedFormat"] = _erp_schema.format_id
        data["ReportDetails"]["ParserMode"] = _erp_schema.erp_name
        area = _new_area("UNKNOWN AREA")
        store = _new_store("UNKNOWN STORE", "")
        area["Stores"].append(store)
        data["Areas"].append(area)
        for row in erp_rows:
            if row.get("_row_type") == "grand_total":
                continue
            amt = clean_number(row.get("amount", "0"))
            rate = clean_number(row.get("rate", "0"))
            qty = clean_number(row.get("qty", "0"), is_int=True)
            free = clean_number(row.get("free_qty", "0"), is_int=True)
            desc = (row.get("product_name") or row.get("party_name") or "").strip()
            if not desc:
                continue
            store["Items"].append({
                "Description": desc,
                "Brand_Name": "",
                "Dosage": "",
                "Packaging": row.get("packing", ""),
                "Qty": qty,
                "Free": free,
                "Rate": rate,
                "Amount": amt,
                "Percent": 0.0,
            })
        if not store["Items"]:
            log.info("ERPTemplateEngine matched but produced 0 rows; falling back to MARG state machine.")
            from parsers.legacy_machine import parse_text
            data = parse_text(raw_text)
    else:
        data = route_and_parse(raw_text, str(target_file))

    # Store AI inference result for QA report (non-blocking)
    try:
        if _AI_AVAILABLE and _ai_format_inference is not None:
            _ai_res = _ai_format_inference.infer(
                raw_text.split('\n')[:40], raw_text
            )
            data["_ai_format_result"] = {
                "format_id": _ai_res.format_id,
                "confidence": round(_ai_res.confidence, 3),
                "tier": _ai_res.tier,
                "source": _ai_res.source,
            }
    except Exception:
        pass

    if data.get("ReportDetails", {}).get("ParserMode") == "SCHEMA_INFERRED":
        try:
            from transformers.schema_cleaner import clean_schema_output
            data = clean_schema_output(data)
        except Exception:
            pass

    data = post_process_extracted_data(data)
    data = safe_clean_store_entities(data)

    rd = data["ReportDetails"]
    if rd.get("ParserMode") == "PAGED_STORE_INVOICE" and data.get("_grand_total", 0.0) > 0:
        expected_grand_total = data["_grand_total"]

    if rd.get("FromDate") and not rd.get("ToDate"):
        log.info("This is not a statement, it is an INVOICE copy (no date range).")
        rd["ToDate"] = rd["FromDate"]

    log.info("Parsing/Logic Time: %.2f seconds", time.time() - parse_t0)

    area_count = len(data.get("Areas", []))
    store_count = sum(len(a.get("Stores", [])) for a in data.get("Areas", []))
    item_count = sum(len(s.get("Items", [])) for a in data.get("Areas", []) for s in a.get("Stores", []))
    log.info("Parsed: %d areas, %d stores, %d items", area_count, store_count, item_count)

    detected_format = data.get("ReportDetails", {}).get("DetectedFormat", "")
    if item_count == 0 or detected_format == "LEGACY_UNIVERSAL_TWO_COLUMN":
        log.warning("Unrecognized Format or 0 items extracted! Triggering Investigation Dump...")
        dump_dir = os.path.join(output_dir, "investigation_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(dump_dir, f"{target_file.stem}_RAW_TEXT.txt")
        try:
            with open(dump_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            log.info("Raw text saved for investigation: %s", dump_path)
        except Exception as e:
            log.error("Could not save raw text dump: %s", e)

    json_path = os.path.join(output_dir, target_file.with_suffix('.json').name)
    csv_path = os.path.join(output_dir, target_file.with_suffix('.csv').name)

    write_t0 = time.time()
    log.info("Step 3/4  —  Writing JSON ...")
    write_json(data, json_path)

    log.info("Step 4/4  —  Writing CSV ...")
    total_val = write_csv(data, csv_path, target_file.name)
    log.info("Writing/Updation Time: %.4f seconds", time.time() - write_t0)

    log.info("=" * 60)
    log.info("Done. %d rows extracted in %.2f seconds total.", item_count, time.time() - total_t0)
    log.info("CSV TOTAL EXTRACTED: %s", f"{total_val:,.2f}")

    status = "Processed"
    remarks = ""
    if target_file.suffix.lower() == '.pdf' and item_count == 0 and len(raw_text) > 150:
        status = "UNSUPPORTED_FORMAT"
        remarks = "Native Text Extracted but Regex Failed"

    summary_record = {
        "File Name": target_file.name,
        "Status": status,
        "Actual Amount": expected_grand_total if expected_grand_total is not None else 0.0,
        "CSV Amount": total_val,
        "Missing Gap Amount": 0.0,
        "Remarks": remarks
    }

    if status != "UNSUPPORTED_FORMAT":
        if expected_grand_total is not None and expected_grand_total > 0:
            diff = abs(total_val - expected_grand_total)
            summary_record["Missing Gap Amount"] = diff
            if diff < 1.00:
                log.info("100%% MATCH! Extracted total matches Document Grand Total (%s)!",
                         f"{expected_grand_total:,.2f}")
                summary_record["Status"] = "Perfect Match"
                summary_record["Remarks"] = "Values matching"
            else:
                log.warning("Document Grand Total is %s, but script extracted %s. Missing Gap: %s",
                            f"{expected_grand_total:,.2f}", f"{total_val:,.2f}", f"{diff:,.2f}")
                summary_record["Status"] = "Mismatch Detected"
                summary_record["Remarks"] = f"Check final amount / Gap found: {diff:.2f}"
        else:
            log.info("Could not auto-detect 'GRAND TOTAL' in the Document. Please verify manually.")
            summary_record["Status"] = "Grand Total Not Found"
            summary_record["Remarks"] = "Unable to auto-verify Grand Total"

    return summary_record


if __name__ == "__main__":
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