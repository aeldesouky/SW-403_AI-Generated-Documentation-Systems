import sys
import os
import pandas as pd
import json
import time
import argparse
import random
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generator import generate_documentation
from src.evaluator import calculate_metrics
from src.analysis import detect_hallucination

# --- CONFIGURATION ---
LOG_DIR = "experiments/logs"
RESULTS_DIR = "experiments/results"
CHECKPOINT_FILE = os.path.join(LOG_DIR, "checkpoint_log.csv")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiment_sequential(input_file, output_file, model="gpt-3.5-turbo", sample_size=None, base_delay=2.0):
    print(f"Starting Experiment on {input_file}")
    print(f"Logging checkpoints to: {CHECKPOINT_FILE}")
    print(f"Final Results will save to: {output_file}")
    
    # 1. Load Data
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # 2. Apply Sampling
    if sample_size and sample_size < len(dataset):
        print(f"Downsampling dataset from {len(dataset)} to {sample_size} random samples.")
        random.seed(42)
        dataset = random.sample(dataset, sample_size)
    else:
        print(f"Processing full dataset ({len(dataset)} items).")

    # 3. Check for existing progress
    processed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        try:
            df_existing = pd.read_csv(CHECKPOINT_FILE)
            if 'id' in df_existing.columns:
                processed_ids = set(df_existing['id'].tolist())
                print(f"Resuming... {len(processed_ids)} samples already processed.")
        except Exception:
            print("Checkpoint file unreadable. Starting fresh.")

    dataset_to_run = [d for d in dataset if d['id'] not in processed_ids]
    
    if not dataset_to_run:
        print("All items already processed.")
        return

    results = []
    
    # 4. Sequential Loop with Backoff
    pbar = tqdm(dataset_to_run, desc="Processing")
    
    for entry in pbar:
        success = False
        attempt = 0
        max_retries = 3
        current_sleep = 10 
        
        while not success and attempt < max_retries:
            try:
                # A. Generate
                gen_doc = generate_documentation(entry['code'], entry['language'], model_name=model)
                
                if gen_doc.startswith("Error:"):
                    raise Exception(gen_doc)

                # B. Evaluate
                metrics = calculate_metrics(entry['ground_truth'], gen_doc)
                
                # C. Detect Hallucination
                hallucination_data = detect_hallucination(entry['code'], gen_doc)
                
                # D. Format Row for Rubric
                # Columns: | Input | Model Output | Expected | Error Type | Hallucination? | Root Cause Hypothesis |
                row = {
                    "id": entry['id'],
                    "language": entry['language'],
                    
                    # Rubric Required Columns
                    "Input": entry['code'],
                    "Model Output": gen_doc,
                    "Expected": entry['ground_truth'],
                    "Error Type": hallucination_data['error_type'],
                    "Hallucination?": hallucination_data['has_hallucination'],
                    "Root Cause Hypothesis": hallucination_data['root_cause'],
                    
                    # Metrics (Keep for Deliverable 3)
                    "BLEU": metrics['bleu'],
                    "BERTScore": metrics['bert_similarity']
                }
                
                results.append(row)
                success = True
                
                # E. Save Checkpoint (every 5)
                if len(results) >= 5:
                    df_chunk = pd.DataFrame(results)
                    header = not os.path.exists(CHECKPOINT_FILE)
                    df_chunk.to_csv(CHECKPOINT_FILE, mode='a', header=header, index=False)
                    results = [] 
                
                # F. Success Delay
                time.sleep(base_delay)

            except Exception as e:
                attempt += 1
                error_msg = str(e)
                if "429" in error_msg or "Rate limit" in error_msg:
                    pbar.set_description(f"Rate Limit. Sleeping {current_sleep}s...")
                    time.sleep(current_sleep)
                    current_sleep *= 2 
                else:
                    # Non-retryable error
                    break 

    # Save remaining
    if results:
        df_chunk = pd.DataFrame(results)
        header = not os.path.exists(CHECKPOINT_FILE)
        df_chunk.to_csv(CHECKPOINT_FILE, mode='a', header=header, index=False)

    # 5. Final Save to Results Dir (Consolidating everything)
    print("Consolidating logs into final results file...")
    if os.path.exists(CHECKPOINT_FILE):
        final_df = pd.read_csv(CHECKPOINT_FILE)
        final_df.to_csv(output_file, index=False)
        print(f"Experiment Complete. Final Report: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="Ignored")
    parser.add_argument("--sample", type=int, default=500, help="Number of samples to process")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds to sleep between requests")
    args = parser.parse_args()
    
    run_experiment_sequential(
        input_file="data/processed/full_experiment_set.jsonl", 
        output_file=os.path.join(RESULTS_DIR, "final_hallucination_report.csv"),
        sample_size=args.sample,
        base_delay=args.delay
    )