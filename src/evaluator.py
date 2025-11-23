import logging
import transformers

# Set transformers logging to error only to hide the "weights not initialized" warning
transformers.logging.set_verbosity_error()
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def calculate_metrics(reference, candidate):
    """
    Computes BLEU, ROUGE, and BERTScore (Semantic Similarity).
    """
    metrics = {}

    # 1. BLEU Score (n-gram overlap)
    smoothie = SmoothingFunction().method4
    # Basic tokenization
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if not ref_tokens or not cand_tokens:
        metrics['bleu'] = 0.0
    else:
        metrics['bleu'] = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)

    # 2. ROUGE Score (Recall-oriented)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure

    # 3. BERTScore (Semantic Similarity / Cosine)
    # We use a lightweight model to keep it fast for the prototype (distilbert is common default)
    try:
        P, R, F1 = score([candidate], [reference], lang='en', verbose=False)
        metrics['bert_similarity'] = F1.mean().item()
    except Exception as e:
        print(f"BERTScore Warning: {e}")
        metrics['bert_similarity'] = 0.0

    return metrics
