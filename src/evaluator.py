from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
# We import BERTScorer class instead of the 'score' function to control initialization
from bert_score import BERTScorer 
import logging
import warnings

# Suppress the specific meta tensor warning to keep console clean
warnings.filterwarnings("ignore", message=".*meta tensor.*")

# GLOBAL INITIALIZATION
# We load the model ONCE here. 
# 'lang="en"' uses roberta-large by default. 
# We disable 'rescale_with_baseline' for speed/stability in this prototype.
print("Initializing BERTScore model (this happens once)...")
try:
    # device=None automatically chooses GPU if available, else CPU
    global_scorer = BERTScorer(lang="en", rescale_with_baseline=False)
except Exception as e:
    print(f"BERTScore failed to load: {e}. Fallback to CPU.")
    global_scorer = BERTScorer(lang="en", device="cpu", rescale_with_baseline=False)
print("BERTScore model loaded.")

def calculate_metrics(reference, candidate):
    """
    Computes BLEU, ROUGE, and BERTScore (Semantic Similarity).
    Thread-safe because it uses the pre-loaded global_scorer.
    """
    metrics = {}

    # 1. BLEU Score (n-gram overlap)
    try:
        smoothie = SmoothingFunction().method4
        # BLEU expects list of tokens
        metrics['bleu'] = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
    except:
        metrics['bleu'] = 0.0

    # 2. ROUGE Score (Recall-oriented)
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)
        metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
    except:
        metrics['rouge_l'] = 0.0

    # 3. BERTScore (Semantic Similarity)
    # Uses the global object initialized at the top
    try:
        # BERTScorer expects a list of strings
        P, R, F1 = global_scorer.score([candidate], [reference])
        # .item() converts the single tensor value to a standard float
        metrics['bert_similarity'] = F1.mean().item()
    except Exception as e:
        # Fallback if BERT fails inside thread
        metrics['bert_similarity'] = 0.0 

    return metrics