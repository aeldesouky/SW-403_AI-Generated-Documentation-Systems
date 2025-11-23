import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_plots(csv_file):
    # Load data
    df = pd.read_csv(csv_file)
    
    # We must fill it back in as a real category.
    df['error_type'] = df['error_type'].fillna('No Error')
    
    # Also handle cases where it might be explicitly string "None"
    df['error_type'] = df['error_type'].replace('None', 'No Error')

    # Ensure output folder exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Set academic style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Metrics Distribution by Language
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='language', y='bert_sim', data=df, palette="Set2")
    plt.title('Semantic Similarity (BERTScore) by Language')
    plt.ylabel('BERT Score')
    plt.tight_layout()
    plt.savefig('experiments/results/metric_dist.png')
    print("Saved metric_dist.png")

    # Plot 2: Hallucination Count
    plt.figure(figsize=(8, 5))
    # Order the bars so "No Error" is usually first or last
    sns.countplot(x='error_type', data=df, palette="Reds_d")
    plt.title('Distribution of Error Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('experiments/results/error_dist.png')
    print("Saved error_dist.png")

if __name__ == "__main__":
    create_plots("experiments/results/batch_run_v1.csv")