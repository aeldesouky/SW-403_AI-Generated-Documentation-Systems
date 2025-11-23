import json
import pandas as pd
from datasets import load_dataset
import os

def create_experiment_dataset():
    data = []

    # 1. Get Modern Data (Python) from CodeSearchNet
    print("Loading CodeSearchNet (Python)...")
    try:
        # Streaming allows loading without downloading the whole massive dataset
        dataset = load_dataset("code_search_net", "python", split="test", streaming=True, trust_remote_code=True)
        
        # Take first 20 examples
        count = 0
        for entry in dataset:
            if count >= 20: break
            data.append({
                "id": f"py_{count}",
                "language": "Python",
                "code": entry['func_code_string'],
                "ground_truth": entry['func_documentation_string']
            })
            count += 1
    except Exception as e:
        print(f"Error loading CodeSearchNet: {e}")
        print("Falling back to synthetic Python examples only.")

    # 2. Add Synthetic Legacy Data (COBOL)
    print("Adding Legacy Data (COBOL)...")
    cobol_sample_1 = {
        "id": "cob_01",
        "language": "COBOL",
        "code": "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. HELLO.\n       PROCEDURE DIVISION.\n           DISPLAY 'HELLO WORLD'.\n           STOP RUN.",
        "ground_truth": "A simple COBOL program that prints 'HELLO WORLD' to the console and terminates execution."
    }
    cobol_sample_2 = {
        "id": "cob_02",
        "language": "COBOL",
        "code": "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. CALC.\n       DATA DIVISION.\n       WORKING-STORAGE SECTION.\n       01 NUM1 PIC 9(2) VALUE 10.\n       01 NUM2 PIC 9(2) VALUE 20.\n       01 TOTAL PIC 9(3).\n       PROCEDURE DIVISION.\n           ADD NUM1 TO NUM2 GIVING TOTAL.\n           DISPLAY 'SUM IS: ' TOTAL.\n           STOP RUN.",
        "ground_truth": "This COBOL program defines two variables NUM1 and NUM2, calculates their sum, stores it in TOTAL, and displays the result."
    }
    
    data.append(cobol_sample_1)
    data.append(cobol_sample_2)
    
    # Save to JSONL
    output_path = "data/processed/experiment_set.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Dataset created with {len(data)} samples at {output_path}")

if __name__ == "__main__":
    create_experiment_dataset()
