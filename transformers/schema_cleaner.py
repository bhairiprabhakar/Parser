# transformers/schema_cleaner.py
from transformers.item_parser import parse_item_description

def clean_schema_output(data: dict) -> dict:
    for area in data.get("Areas", []):
        for store in area.get("Stores", []):
            for item in store.get("Items", []):
                desc = item.get("Description", "")
                
                # Prevent parse_item_description from parsing "BILL AMOUNT" [cite: 667, 668]
                if desc == 'BILL AMOUNT':
                    item['Description'] = 'PARTY BILL'
                    item['Brand_Name'] = ''
                    item['Dosage'] = ''
                    item['Packaging'] = ''
                else:
                    parsed = parse_item_description(desc)
                    item.update(parsed) [cite: 669]
    return data