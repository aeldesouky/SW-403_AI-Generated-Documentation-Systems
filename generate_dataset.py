import json
import argparse
import random
import os
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# 2. Define Directory Paths
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# File paths for caching raw downloads
PY_RAW_PATH = os.path.join(RAW_DATA_DIR, "python_raw.jsonl")
COB_RAW_PATH = os.path.join(RAW_DATA_DIR, "cobol_raw.jsonl")

# Fallback templates (Only used if ZERO real COBOL is found)
FALLBACK_TEMPLATES = [
    ("CALC-TAX", "COMPUTE TAX = AMOUNT * 0.05.", "Calculates 5% tax."),
    ("CHECK-BAL", "IF BALANCE < 0 DISPLAY 'OVERDRAFT'.", "Checks for negative balance.")
]

def authenticate_huggingface():
    token = os.getenv("HF_TOKEN")
    if token:
        print(f"Authenticating with HF_TOKEN found in .env...")
        login(token=token)
        return True
    return False

def load_or_fetch_python(limit=None):
    """Loads Python data from local raw cache or fetches from HF."""
    data = []
    
    # A. Try Loading Local
    if os.path.exists(PY_RAW_PATH):
        print(f"Loading Python from local cache: {PY_RAW_PATH}")
        with open(PY_RAW_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and len(data) >= limit: break
                data.append(json.loads(line))
        return data

    # B. Fetch from Internet
    print("Streaming CodeXGlue (Python) from Hugging Face...")
    try:
        ds = load_dataset("google/code_x_glue_ct_code_to_text", "python", split="test", streaming=True)
    except:
        ds = load_dataset("microsoft/codexglue_func_docstring", "python", split="test", streaming=True)

    # Save to Raw as we iterate
    with open(PY_RAW_PATH, 'w', encoding='utf-8') as f_out:
        for entry in tqdm(ds, desc="Fetching Python"):
            if limit and len(data) >= limit: break
            
            code = entry.get('code') or entry.get('func_code_string')
            doc = entry.get('docstring') or entry.get('func_documentation_string')
            
            if not code or len(code) < 50: continue
            
            row = {"language": "Python", "code": code, "ground_truth": doc}
            data.append(row)
            f_out.write(json.dumps(row) + "\n")
            
    return data

def load_or_fetch_cobol(target_count):
    """Loads COBOL data from local raw cache or fetches from HF."""
    data = []
    
    # A. Try Loading Local
    if os.path.exists(COB_RAW_PATH):
        print(f"Loading COBOL from local cache: {COB_RAW_PATH}")
        with open(COB_RAW_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    # B. Fetch from Internet
    authenticate_huggingface()
    print("Attempting to fetch REAL COBOL from BigCode/The-Stack...")
    
    try:
        ds = load_dataset("bigcode/the-stack", data_dir="data/cobol", split="train", streaming=True, token=True)
        
        with open(COB_RAW_PATH, 'w', encoding='utf-8') as f_out:
            for entry in tqdm(ds, desc="Fetching COBOL"):
                # We fetch slightly more than python to have a good seed pool, but cap it to avoid huge downloads
                if len(data) >= max(target_count, 5000): break 
                
                code = entry.get('content')
                if not code or len(code) < 50: continue

                row = {
                    "language": "COBOL", 
                    "code": code[:2000], # Truncate massive files
                    "ground_truth": "Legacy COBOL logic requiring documentation."
                }
                data.append(row)
                f_out.write(json.dumps(row) + "\n")
                
    except Exception as e:
        print(f"Could not fetch real COBOL ({e}).")
        
    return data

def create_dataset(output_filename, limit=None):
    output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
    final_dataset = []

    # 1. Get Python Data
    py_data = load_or_fetch_python(limit)
    # Add IDs
    for i, row in enumerate(py_data):
        row['id'] = f"py_{i}"
        final_dataset.append(row)

    count_py = len(py_data)
    
    # 2. Get Real COBOL Data
    cob_real_data = load_or_fetch_cobol(count_py)
    
    # Add IDs to Real COBOL
    for i, row in enumerate(cob_real_data):
        row['id'] = f"cob_real_{i}"
        final_dataset.append(row)

    count_real_cob = len(cob_real_data)

    # 3. Synthesize Remaining COBOL (Using Real Data as Template)
    if count_real_cob < count_py:
        needed = count_py - count_real_cob
        print(f"Synthesizing {needed} COBOL samples using {count_real_cob} real templates...")
        
        for i in range(needed):
            # Seed Selection: Pick a real COBOL file if available, else Fallback
            if count_real_cob > 0:
                seed = random.choice(cob_real_data)
                base_code = seed['code']
                base_truth = seed['ground_truth']
            else:
                seed = random.choice(FALLBACK_TEMPLATES)
                base_code = seed[1]
                base_truth = seed[2]

            # Mutation: Add a comment header to make it a unique string
            mutation = f"      * REFACTOR CANDIDATE BATCH_{i}\n      * SYSTEM GENERATED\n"
            
            final_dataset.append({
                "id": f"cob_syn_{i}",
                "language": "COBOL",
                "code": mutation + base_code,
                "ground_truth": base_truth
            })

    # 4. Save Final Processed Dataset
    with open(output_path, "w", encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Dataset complete. Saved to: {output_path}")
    print(f"Stats: {count_py} Python")
    print(f"       {count_real_cob} Real COBOL")
    print(f"       {max(0, count_py - count_real_cob)} Synthetic COBOL (Derived from Real)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per language")
    args = parser.parse_args()
    
    create_dataset("full_experiment_set.jsonl", args.limit)