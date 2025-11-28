import sys
import os
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generator import generate_documentation
from src.evaluator import calculate_metrics

RESULTS_DIR = "experiments/results"
SWEEP_FILE = os.path.join(RESULTS_DIR, "parameter_sweep.csv")

def run_sweep(input_file, sample_size=30):
    print(f"Starting Parameter Sweep (Temperatures: 0.2, 0.5, 0.8) on {sample_size} samples...")
    
    # Load a tiny subset of data
    with open(input_file, 'r', encoding='utf-8') as f:
        full_data = [json.loads(line) for line in f]
    
    # Take the first N samples
    dataset = full_data[:sample_size]
    
    temperatures = [0.2, 0.5, 0.8]
    results = []

    for temp in temperatures:
        print(f"Testing Temperature: {temp}")
        for entry in tqdm(dataset):
            try:
                # Generate with specific temp
                gen_doc = generate_documentation(
                    entry['code'], 
                    entry['language'], 
                    model_name="gpt-3.5-turbo", 
                    temperature=temp
                )
                
                # Check for errors
                if gen_doc.startswith("Error:"):
                    continue

                # Calculate BERTScore (Metric of interest)
                metrics = calculate_metrics(entry['ground_truth'], gen_doc)
                
                results.append({
                    "Temperature": temp,
                    "Language": entry['language'],
                    "BERTScore": metrics['bert_similarity'],
                    "BLEU": metrics['bleu']
                })
            except Exception as e:
                print(f"Skipping ID {entry['id']}: {e}")

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(SWEEP_FILE, index=False)
    print(f"Sweep complete. Data saved to {SWEEP_FILE}")
    
    # Generate the Comparison Graph immediately
    generate_sweep_graph(df)

def generate_sweep_graph(df):
    sns.set_theme(style="whitegrid")
    
    # Comparison Plot: Temperature vs BERTScore
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Temperature", y="BERTScore", hue="Language", marker="o")
    
    plt.title("Impact of Temperature on Semantic Consistency")
    plt.ylabel("BERTScore")
    plt.xlabel("Temperature Parameter")
    plt.legend(title="Language")
    
    output_img = os.path.join(RESULTS_DIR, "param_comparison.png")
    plt.savefig(output_img)
    print(f"Graph saved to {output_img}")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_sweep("data/processed/full_experiment_set.jsonl")