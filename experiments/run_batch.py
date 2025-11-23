import sys
import os
import pandas as pd
from tqdm import tqdm # Progress bar
import json

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generator import generate_documentation
from src.evaluator import calculate_metrics
from src.analysis import detect_hallucination

def run_experiment(input_file, output_file, model="gpt-3.5-turbo"):
    print(f"🚀 Starting Batch Experiment using {model}...")
    
    # Load Data
    results = []
    with open(input_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    # Loop with Progress Bar
    for entry in tqdm(dataset):
        code = entry['code']
        ground_truth = entry['ground_truth']
        lang = entry['language']
        
        # 1. Generate
        gen_doc = generate_documentation(code, lang, model_name=model)
        
        # 2. Evaluate Metrics (BLEU/ROUGE)
        metrics = calculate_metrics(ground_truth, gen_doc)
        
        # 3. Detect Hallucinations (The "Above and Beyond" step)
        # We only run this on the first 10 to save API costs, or all if you are brave.
        hallucination_data = detect_hallucination(code, gen_doc)
        
        # 4. Log Data
        row = {
            "id": entry['id'],
            "language": lang,
            "model": model,
            "bleu_score": metrics['bleu'],
            "rouge_l": metrics['rouge_l'],
            "bert_sim": metrics['bert_similarity'],
            "has_hallucination": hallucination_data['has_hallucination'],
            "error_type": hallucination_data['error_type'],
            "root_cause": hallucination_data['root_cause'],
            "input_code_snippet": code[:50] + "...", # Truncate for CSV readability
            "output_doc": gen_doc
        }
        results.append(row)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"✅ Experiment Complete! Results saved to {output_file}")
    
    # Print Summary Stats for immediate gratification
    print("\n--- Early Results Summary ---")
    print(f"Average BLEU: {df['bleu_score'].mean():.4f}")
    print(f"Average BERT Sim: {df['bert_sim'].mean():.4f}")
    print(f"Hallucination Rate: {df['has_hallucination'].mean() * 100:.2f}%")

if __name__ == "__main__":
    # Ensure folders exist
    os.makedirs("experiments/results", exist_ok=True)
    
    # Run
    run_experiment(
        input_file="data/processed/experiment_set.jsonl", 
        output_file="experiments/results/batch_run_v1.csv"
    )