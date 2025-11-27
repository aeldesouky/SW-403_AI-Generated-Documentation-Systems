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

CHECKPOINT_FILE = "experiments/results/checkpoint_log.csv"

def run_experiment_sequential(input_file, output_file, model="gpt-3.5-turbo", sample_size=None, base_delay=2.0):
    print(f"Starting Sequential Experiment on {input_file}")
    print(f"Mode: Sequential with Smart Backoff (Base Delay: {base_delay}s)")
    
    # 1. Load Data
    with open(input_file, 'r') as f:
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
        current_sleep = 10 # Start with 10s if we hit an error
        
        # Retry Loop for Rate Limits
        while not success and attempt < max_retries:
            try:
                # A. Generate
                gen_doc = generate_documentation(entry['code'], entry['language'], model_name=model)
                
                # CRITICAL FIX: Check if the generator returned an error string
                if gen_doc.startswith("Error:"):
                    raise Exception(gen_doc) # Force flow into the except block to trigger backoff

                # B. Evaluate
                metrics = calculate_metrics(entry['ground_truth'], gen_doc)
                
                # C. Detect Hallucination
                hallucination_data = detect_hallucination(entry['code'], gen_doc)
                
                row = {
                    "id": entry['id'],
                    "language": entry['language'],
                    "bleu_score": metrics['bleu'],
                    "bert_sim": metrics['bert_similarity'],
                    "has_hallucination": hallucination_data['has_hallucination'],
                    "error_type": hallucination_data['error_type'],
                    "root_cause": hallucination_data['root_cause'],
                    "output_doc": gen_doc
                }
                results.append(row)
                success = True # Exit retry loop
                
                # D. Save Checkpoint
                if len(results) >= 5:
                    df_chunk = pd.DataFrame(results)
                    header = not os.path.exists(CHECKPOINT_FILE)
                    df_chunk.to_csv(CHECKPOINT_FILE, mode='a', header=header, index=False)
                    results = [] 
                
                # E. Success Delay (Standard pacing)
                time.sleep(base_delay)

            except Exception as e:
                attempt += 1
                error_msg = str(e)
                
                # Check for Rate Limit keywords
                if "429" in error_msg or "Rate limit" in error_msg:
                    pbar.set_description(f"Rate Limit Hit. Sleeping {current_sleep}s...")
                    time.sleep(current_sleep)
                    current_sleep *= 2 # Exponential Backoff (10s -> 20s -> 40s)
                else:
                    # If it's another error (like context length), just log and skip
                    print(f"\nNon-Retryable Error on ID {entry['id']}: {e}")
                    break 

    # Save remaining
    if results:
        df_chunk = pd.DataFrame(results)
        header = not os.path.exists(CHECKPOINT_FILE)
        df_chunk.to_csv(CHECKPOINT_FILE, mode='a', header=header, index=False)

    print(f"Run Complete. Final results in {CHECKPOINT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="Ignored")
    parser.add_argument("--sample", type=int, default=500, help="Number of samples to process")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to sleep between requests")
    args = parser.parse_args()

    os.makedirs("experiments/results", exist_ok=True)
    
    run_experiment_sequential(
        input_file="data/processed/full_experiment_set.jsonl", 
        output_file="experiments/results/final_results.csv",
        sample_size=args.sample,
        base_delay=args.delay
    )