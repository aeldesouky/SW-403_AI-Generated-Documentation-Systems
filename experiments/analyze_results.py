import pandas as pd
import os

RESULTS_FILE = "experiments/results/final_hallucination_report.csv"

def print_analysis():
    if not os.path.exists(RESULTS_FILE):
        print("No results file found yet. Run experiments first.")
        return

    df = pd.read_csv(RESULTS_FILE)
    
    print("--- 1. BASELINE METRICS & ACCURACY ---")
    # Group by language to see Baseline (Python) vs Experiment (COBOL)
    metrics = df.groupby('language')[['BLEU', 'BERTScore']].mean()
    print(metrics)
    
    print("\n--- 2. FAILURE CATEGORIES ---")
    # Filter for failures only
    failures = df[df['Hallucination?'] == True]
    
    if len(failures) > 0:
        breakdown = failures.groupby(['language', 'Error Type']).size().reset_index(name='Count')
        # Calculate percentages
        total_failures = failures.groupby('language').size()
        print(breakdown)
    else:
        print("No hallucinations detected (Perfect Run).")

if __name__ == "__main__":
    print_analysis()