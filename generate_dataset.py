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

def create_dataset(output_filename, limit=None):
    output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
    final_dataset = []

    # Get Python Data
    py_data = load_or_fetch_python(limit)
    # Add IDs
    for i, row in enumerate(py_data):
        row['id'] = f"py_{i}"
        final_dataset.append(row)

    count_py = len(py_data)

    # Save Final Processed Dataset
    with open(output_path, "w", encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Dataset complete. Saved to: {output_path}")
    print(f"Stats: {count_py} Python samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per language")
    args = parser.parse_args()
    
    create_dataset("full_experiment_set.jsonl", args.limit)