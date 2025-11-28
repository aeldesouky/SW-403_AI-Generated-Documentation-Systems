import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def generate_simulated_data():
    """Creates a mock dataset to simulate the FULL run results."""
    np.random.seed(42)
    data = []
    
    # Simulate 50 Python samples (High accuracy)
    for i in range(50):
        data.append({
            "language": "Python",
            "bert_sim": np.random.normal(0.88, 0.05),
            "error_type": np.random.choice(["No Error", "Omission"], p=[0.96, 0.04])
        })
        
    # Simulate 50 COBOL samples (Lower accuracy, higher hallucinations)
    for i in range(50):
        data.append({
            "language": "COBOL",
            "bert_sim": np.random.normal(0.69, 0.1),
            "error_type": np.random.choice(
                ["No Error", "Fabricated Logic", "Omission", "Hallucinated Var"], 
                p=[0.72, 0.10, 0.10, 0.08]
            )
        })
    return pd.DataFrame(data)

def plot_results():
    # Load data (Simulated for now, replace with 'pd.read_csv' for real)
    df = generate_simulated_data()
    
    sns.set_theme(style="whitegrid")
    
    # 1. Semantic Similarity Distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='language', y='bert_sim', data=df, palette="muted", inner="quartile")
    plt.title('Semantic Accuracy Gap: Modern vs. Legacy Code', fontsize=14)
    plt.ylabel('BERTScore (Similarity)', fontsize=12)
    plt.savefig("experiments/results/full_metric_dist.png")
    
    # 2. Hallucination Breakdown
    plt.figure(figsize=(10, 6))
    # Filter out "No Error" to focus on failure modes
    error_df = df[df['error_type'] != "No Error"]
    sns.countplot(y='error_type', hue='language', data=error_df, palette="Reds_d")
    plt.title('Distribution of Hallucination Types by Language', fontsize=14)
    plt.xlabel('Count of Errors', fontsize=12)
    plt.savefig("experiments/results/full_error_breakdown.png")
    
    print("Visuals generated: full_metric_dist.png, full_error_breakdown.png")

if __name__ == "__main__":
    plot_results()