import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

def generate_simulated_data():
    """Creates a mock dataset to simulate the FULL run results."""
    np.random.seed(42)
    data = []
    
    # Simulate 50 Python samples (High accuracy)
    for i in range(50):
        bert_score = np.random.normal(0.88, 0.05)
        data.append({
            "language": "Python",
            "bert_sim": bert_score,
            "error_type": np.random.choice(["No Error", "Omission"], p=[0.96, 0.04]),
            "complexity": np.random.choice(["Low", "Medium", "High"], p=[0.3, 0.5, 0.2]),
            "doc_length": int(np.random.normal(150, 30)),
            "confidence_score": bert_score + np.random.normal(0, 0.03)
        })
        
    # Simulate 50 COBOL samples (Lower accuracy, higher hallucinations)
    for i in range(50):
        bert_score = np.random.normal(0.69, 0.1)
        data.append({
            "language": "COBOL",
            "bert_sim": bert_score,
            "error_type": np.random.choice(
                ["No Error", "Fabricated Logic", "Omission", "Hallucinated Var"], 
                p=[0.72, 0.10, 0.10, 0.08]
            ),
            "complexity": np.random.choice(["Low", "Medium", "High"], p=[0.2, 0.4, 0.4]),
            "doc_length": int(np.random.normal(180, 40)),
            "confidence_score": bert_score + np.random.normal(0, 0.05)
        })
    return pd.DataFrame(data)

def plot_results():
    # Load data (Simulated for now, replace with 'pd.read_csv' for real)
    df = generate_simulated_data()
    
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Semantic Similarity Distribution with Statistical Annotations
    plt.subplot(2, 3, 1)
    parts = plt.violinplot([df[df['language']=='Python']['bert_sim'], 
                            df[df['language']=='COBOL']['bert_sim']], 
                           positions=[0, 1], showmeans=True, showmedians=True)
    plt.xticks([0, 1], ['Python', 'COBOL'])
    plt.ylabel('BERTScore (Similarity)', fontsize=11)
    plt.title('Semantic Accuracy Distribution\n(Violin Plot)', fontsize=12, fontweight='bold')
    
    # Add statistical test
    python_scores = df[df['language']=='Python']['bert_sim']
    cobol_scores = df[df['language']=='COBOL']['bert_sim']
    t_stat, p_value = stats.ttest_ind(python_scores, cobol_scores)
    plt.text(0.5, 0.95, f'p-value: {p_value:.2e}', transform=plt.gca().transAxes, 
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Error Rate Comparison (Percentage)
    plt.subplot(2, 3, 2)
    df['has_error'] = (df['error_type'] != "No Error").astype(int)
    error_rates = df.groupby('language')['has_error'].mean() * 100
    bars = plt.bar(error_rates.index, error_rates.values, color=['#2ecc71', '#e74c3c'], alpha=0.7)
    plt.ylabel('Error Rate (%)', fontsize=11)
    plt.title('Overall Hallucination Rate', fontsize=12, fontweight='bold')
    plt.ylim(0, 35)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Hallucination Breakdown (Stacked Bar)
    plt.subplot(2, 3, 3)
    error_df = df[df['error_type'] != "No Error"]
    error_counts = error_df.groupby(['language', 'error_type']).size().unstack(fill_value=0)
    error_counts.plot(kind='bar', stacked=True, ax=plt.gca(), 
                     colormap='YlOrRd', width=0.6)
    plt.ylabel('Count of Errors', fontsize=11)
    plt.title('Hallucination Type Breakdown', fontsize=12, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=0)
    plt.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 4. Complexity vs Accuracy Scatter
    plt.subplot(2, 3, 4)
    complexity_order = {"Low": 0, "Medium": 1, "High": 2}
    df['complexity_num'] = df['complexity'].map(complexity_order)
    for lang in ['Python', 'COBOL']:
        lang_data = df[df['language'] == lang]
        plt.scatter(lang_data['complexity_num'], lang_data['bert_sim'], 
                   label=lang, alpha=0.6, s=50)
    plt.xticks([0, 1, 2], ['Low', 'Medium', 'High'])
    plt.xlabel('Code Complexity', fontsize=11)
    plt.ylabel('BERTScore', fontsize=11)
    plt.title('Accuracy vs Code Complexity', fontsize=12, fontweight='bold')
    plt.legend()
    
    # Add trend lines
    for lang, color in [('Python', 'blue'), ('COBOL', 'red')]:
        lang_data = df[df['language'] == lang]
        z = np.polyfit(lang_data['complexity_num'], lang_data['bert_sim'], 1)
        p = np.poly1d(z)
        plt.plot([0, 1, 2], p([0, 1, 2]), color=color, linestyle='--', alpha=0.3, linewidth=2)
    
    # 5. Documentation Length Distribution
    plt.subplot(2, 3, 5)
    plt.hist([df[df['language']=='Python']['doc_length'], 
              df[df['language']=='COBOL']['doc_length']], 
             label=['Python', 'COBOL'], bins=15, alpha=0.7, color=['#3498db', '#e67e22'])
    plt.xlabel('Documentation Length (words)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Generated Documentation Length', fontsize=12, fontweight='bold')
    plt.legend()
    
    # 6. Confidence vs Actual Accuracy
    plt.subplot(2, 3, 6)
    for lang, marker in [('Python', 'o'), ('COBOL', 's')]:
        lang_data = df[df['language'] == lang]
        plt.scatter(lang_data['confidence_score'], lang_data['bert_sim'], 
                   label=lang, alpha=0.5, s=40, marker=marker)
    plt.plot([0.6, 1.0], [0.6, 1.0], 'k--', alpha=0.3, label='Perfect Calibration')
    plt.xlabel('Model Confidence Score', fontsize=11)
    plt.ylabel('Actual BERTScore', fontsize=11)
    plt.title('Model Calibration Analysis', fontsize=12, fontweight='bold')
    plt.legend()
    
    # Calculate correlation
    corr = df['confidence_score'].corr(df['bert_sim'])
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
             fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("experiments/results/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    print("✓ Comprehensive analysis saved: comprehensive_analysis.png")
    
    # Generate additional focused plots
    generate_detailed_error_analysis(df)
    generate_statistical_summary(df)

def generate_detailed_error_analysis(df):
    """Creates detailed error analysis visualizations"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error rate by complexity
    df['has_error'] = (df['error_type'] != "No Error").astype(int)
    error_by_complexity = df.groupby(['language', 'complexity'])['has_error'].mean().reset_index()
    error_by_complexity['error_rate'] = error_by_complexity['has_error'] * 100
    error_by_complexity = error_by_complexity.drop('has_error', axis=1)
    
    complexity_order = ["Low", "Medium", "High"]
    sns.barplot(data=error_by_complexity, x='complexity', y='error_rate', 
                hue='language', ax=axes[0], order=complexity_order, palette='Set2')
    axes[0].set_title('Error Rate by Code Complexity', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Error Rate (%)', fontsize=11)
    axes[0].set_xlabel('Code Complexity', fontsize=11)
    
    # Accuracy score distribution with KDE
    for lang in ['Python', 'COBOL']:
        lang_data = df[df['language'] == lang]['bert_sim']
        axes[1].hist(lang_data, bins=20, alpha=0.5, label=f'{lang} (μ={lang_data.mean():.3f})', density=True)
        kde = gaussian_kde(lang_data)
        x_range = np.linspace(lang_data.min(), lang_data.max(), 100)
        axes[1].plot(x_range, kde(x_range), linewidth=2)
    
    axes[1].set_title('Accuracy Distribution with KDE', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('BERTScore', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].legend()
    axes[1].axvline(x=0.80, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    plt.tight_layout()
    plt.savefig("experiments/results/detailed_error_analysis.png", dpi=300, bbox_inches='tight')
    print("✓ Detailed error analysis saved: detailed_error_analysis.png")

def generate_statistical_summary(df):
    """Generates and saves statistical summary"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = []
    for lang in ['Python', 'COBOL']:
        lang_data = df[df['language'] == lang]
        summary_data.append([
            lang,
            f"{lang_data['bert_sim'].mean():.4f}",
            f"{lang_data['bert_sim'].std():.4f}",
            f"{lang_data['bert_sim'].median():.4f}",
            f"{(lang_data['error_type'] != 'No Error').sum() / len(lang_data) * 100:.1f}%",
            f"{len(lang_data)}"
        ])
    
    # Add difference row
    python_mean = df[df['language']=='Python']['bert_sim'].mean()
    cobol_mean = df[df['language']=='COBOL']['bert_sim'].mean()
    diff = python_mean - cobol_mean
    
    summary_data.append([
        'Δ (Python - COBOL)',
        f"{diff:.4f} ({diff/cobol_mean*100:.1f}%)",
        '-',
        '-',
        '-',
        '-'
    ])
    
    table = ax.table(cellText=summary_data,
                     colLabels=['Language', 'Mean Score', 'Std Dev', 'Median', 'Error Rate', 'Samples'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style the difference row
    for i in range(6):
        table[(3, i)].set_facecolor('#FFF9C4')
        table[(3, i)].set_text_props(weight='bold')
    
    plt.title('Statistical Summary: Model Performance Comparison', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig("experiments/results/statistical_summary.png", dpi=300, bbox_inches='tight')
    print("✓ Statistical summary saved: statistical_summary.png")
    
    # Print summary to console
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    for row in summary_data:
        print(f"{row[0]:20} | Mean: {row[1]:10} | Error Rate: {row[4]}")
    print("="*60)

if __name__ == "__main__":
    plot_results()