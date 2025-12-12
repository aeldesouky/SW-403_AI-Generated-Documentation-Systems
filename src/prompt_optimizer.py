"""
Prompt Optimizer Module
=======================

This module implements multiple prompting strategies to reduce hallucinations
in AI-generated documentation, addressing RQ2 of the research project.

Features:
- Zero-shot, Few-shot, Chain-of-Thought, and Structured prompting strategies
- RAG (Retrieval-Augmented Generation) for context-aware generation
- Strategy comparison for A/B testing

Author: AI Documentation Systems Team
"""

import os
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Conditional imports for RAG
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: scikit-learn not installed. RAG features disabled.")

# Import existing generator
from generator import generate_documentation


class PromptStrategy(Enum):
    """Available prompting strategies for documentation generation."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STRUCTURED_OUTPUT = "structured_output"
    RAG_ENHANCED = "rag_enhanced"


@dataclass
class PromptResult:
    """Container for prompt generation results."""
    strategy: PromptStrategy
    prompt: str
    generated_doc: str
    metadata: Dict[str, Any]


# Pre-defined few-shot examples for different languages
FEW_SHOT_EXAMPLES = {
    "python": [
        {
            "code": """def calculate_discount(price, percentage):
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")
    return price * (1 - percentage / 100)""",
            "docstring": '''"""Calculate discounted price.

Args:
    price: Original price of the item.
    percentage: Discount percentage (0-100).

Returns:
    The discounted price.

Raises:
    ValueError: If percentage is not between 0 and 100.
"""'''
        },
        {
            "code": """def merge_sorted_lists(list1, list2):
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result""",
            "docstring": '''"""Merge two sorted lists into a single sorted list.

Args:
    list1: First sorted list.
    list2: Second sorted list.

Returns:
    A new sorted list containing all elements from both input lists.
"""'''
        }
    ],
    "cobol": [
        {
            "code": """       COMPUTE TOTAL-AMOUNT = QUANTITY * UNIT-PRICE.
       IF TOTAL-AMOUNT > 1000
           COMPUTE DISCOUNT = TOTAL-AMOUNT * 0.10
       ELSE
           MOVE 0 TO DISCOUNT
       END-IF.
       COMPUTE FINAL-AMOUNT = TOTAL-AMOUNT - DISCOUNT.""",
            "docstring": "Calculates the total amount by multiplying quantity and unit price. Applies a 10% discount if the total exceeds 1000, otherwise no discount. Stores the result in FINAL-AMOUNT."
        },
        {
            "code": """       PERFORM VARYING WS-INDEX FROM 1 BY 1 
           UNTIL WS-INDEX > WS-TABLE-SIZE
           IF WS-TABLE-ITEM(WS-INDEX) = WS-SEARCH-KEY
               MOVE 'Y' TO WS-FOUND-FLAG
               MOVE WS-INDEX TO WS-FOUND-INDEX
               EXIT PERFORM
           END-IF
       END-PERFORM.""",
            "docstring": "Performs a linear search through a table to find a matching key. Sets the found flag to 'Y' and records the index if found, then exits the loop."
        }
    ]
}


class PromptOptimizer:
    """
    Optimizes prompts for AI documentation generation using various strategies.
    
    This class provides methods to:
    - Build optimized prompts using different strategies
    - Retrieve similar code for context (RAG)
    - Compare strategy effectiveness
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the PromptOptimizer.
        
        Args:
            dataset_path: Optional path to JSONL dataset for RAG retrieval.
        """
        self.dataset_path = dataset_path
        self.code_corpus: List[Dict] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Load dataset for RAG if path provided and sklearn available
        if dataset_path and RAG_AVAILABLE:
            self._load_corpus(dataset_path)
    
    def _load_corpus(self, path: str):
        """Load code corpus for RAG retrieval."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if 'code' in entry:
                        self.code_corpus.append(entry)
            
            if self.code_corpus:
                # Build TF-IDF index
                codes = [entry['code'] for entry in self.code_corpus]
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.tfidf_matrix = self.vectorizer.fit_transform(codes)
                print(f"RAG corpus loaded: {len(self.code_corpus)} samples indexed.")
        except FileNotFoundError:
            print(f"Dataset not found: {path}. RAG disabled.")
        except Exception as e:
            print(f"Error loading corpus: {e}")
    
    def retrieve_similar_code(self, query_code: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve similar code snippets from the corpus using TF-IDF similarity.
        
        Args:
            query_code: The code to find similar examples for.
            top_k: Number of similar examples to retrieve.
            
        Returns:
            List of similar code entries with similarity scores.
        """
        if not RAG_AVAILABLE or self.vectorizer is None:
            return []
        
        try:
            query_vec = self.vectorizer.transform([query_code])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            
            # Get top-k indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    entry = self.code_corpus[idx].copy()
                    entry['similarity'] = float(similarities[idx])
                    results.append(entry)
            
            return results
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return []
    
    def get_few_shot_examples(self, language: str, n_examples: int = 2) -> List[Dict]:
        """
        Get few-shot examples for a given language.
        
        Args:
            language: Programming language (python, cobol, etc.)
            n_examples: Number of examples to return.
            
        Returns:
            List of example dictionaries with 'code' and 'docstring' keys.
        """
        lang_key = language.lower()
        examples = FEW_SHOT_EXAMPLES.get(lang_key, FEW_SHOT_EXAMPLES.get('python', []))
        return examples[:n_examples]
    
    def build_prompt(self, code: str, language: str, 
                     strategy: PromptStrategy) -> Tuple[str, str, Dict]:
        """
        Build an optimized prompt based on the selected strategy.
        
        Args:
            code: Source code to document.
            language: Programming language.
            strategy: The prompting strategy to use.
            
        Returns:
            Tuple of (system_prompt, user_prompt, metadata)
        """
        metadata = {"strategy": strategy.value}
        
        if strategy == PromptStrategy.ZERO_SHOT:
            return self._build_zero_shot(code, language, metadata)
        elif strategy == PromptStrategy.FEW_SHOT:
            return self._build_few_shot(code, language, metadata)
        elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            return self._build_chain_of_thought(code, language, metadata)
        elif strategy == PromptStrategy.STRUCTURED_OUTPUT:
            return self._build_structured_output(code, language, metadata)
        elif strategy == PromptStrategy.RAG_ENHANCED:
            return self._build_rag_enhanced(code, language, metadata)
        else:
            return self._build_zero_shot(code, language, metadata)
    
    def _build_zero_shot(self, code: str, language: str, 
                         metadata: Dict) -> Tuple[str, str, Dict]:
        """Standard zero-shot prompt (baseline)."""
        system_prompt = f"""You are an expert {language} developer. 
Write a clear, accurate docstring for the given code.
Return ONLY the docstring, no explanations.
Be precise - do not invent functionality that doesn't exist in the code."""
        
        user_prompt = f"```{language}\n{code}\n```"
        
        return system_prompt, user_prompt, metadata
    
    def _build_few_shot(self, code: str, language: str, 
                        metadata: Dict) -> Tuple[str, str, Dict]:
        """Few-shot prompt with examples."""
        examples = self.get_few_shot_examples(language)
        metadata['n_examples'] = len(examples)
        
        system_prompt = f"""You are an expert {language} developer.
Study the following examples of good documentation, then document the new code in the same style.
Be precise - only describe what the code actually does."""
        
        # Build examples section
        examples_text = ""
        for i, ex in enumerate(examples, 1):
            examples_text += f"\n### Example {i}:\nCode:\n```{language}\n{ex['code']}\n```\n"
            examples_text += f"Documentation:\n{ex['docstring']}\n"
        
        user_prompt = f"""{examples_text}
### Now document this code:
```{language}
{code}
```

Documentation:"""
        
        return system_prompt, user_prompt, metadata
    
    def _build_chain_of_thought(self, code: str, language: str, 
                                metadata: Dict) -> Tuple[str, str, Dict]:
        """Chain-of-thought prompt for step-by-step reasoning."""
        system_prompt = f"""You are an expert {language} developer and code analyst.
Your task is to analyze code step-by-step before writing documentation.

Follow this process:
1. IDENTIFY: List all variables, functions, and data structures used
2. TRACE: Follow the execution flow from start to end
3. SUMMARIZE: Describe what the code accomplishes
4. DOCUMENT: Write precise documentation based on your analysis

Only describe what you can verify from the code. Never invent functionality."""
        
        user_prompt = f"""Analyze and document this {language} code:

```{language}
{code}
```

Step 1 - IDENTIFY (variables and structures):
Step 2 - TRACE (execution flow):
Step 3 - SUMMARIZE (what it does):
Step 4 - FINAL DOCUMENTATION:"""
        
        metadata['reasoning_steps'] = 4
        return system_prompt, user_prompt, metadata
    
    def _build_structured_output(self, code: str, language: str, 
                                 metadata: Dict) -> Tuple[str, str, Dict]:
        """Structured output prompt with explicit format requirements."""
        system_prompt = f"""You are an expert {language} developer.
Generate documentation in the following EXACT structure:

PURPOSE: [One sentence describing the main goal]
INPUTS: [List each parameter/input with type and description]
OUTPUTS: [What the code returns or modifies]
LOGIC: [Brief description of the algorithm/approach]
EDGE CASES: [Any special conditions handled]

Be factual - only describe what is explicitly in the code."""
        
        user_prompt = f"""Document this code using the structured format:

```{language}
{code}
```

PURPOSE:"""
        
        metadata['output_format'] = 'structured'
        return system_prompt, user_prompt, metadata
    
    def _build_rag_enhanced(self, code: str, language: str, 
                            metadata: Dict) -> Tuple[str, str, Dict]:
        """RAG-enhanced prompt with retrieved similar examples."""
        similar_examples = self.retrieve_similar_code(code, top_k=2)
        metadata['rag_examples'] = len(similar_examples)
        
        if not similar_examples:
            # Fall back to few-shot if no similar examples found
            return self._build_few_shot(code, language, metadata)
        
        system_prompt = f"""You are an expert {language} developer.
I will show you similar code examples from the same codebase with their documentation.
Use these as reference for style and detail level, then document the new code.

Important: Only describe what the new code actually does. Don't copy documentation 
from examples if it doesn't apply."""
        
        # Build context from retrieved examples
        context_text = ""
        for i, ex in enumerate(similar_examples, 1):
            context_text += f"\n### Similar Code {i} (similarity: {ex.get('similarity', 0):.2f}):\n"
            context_text += f"```{ex.get('language', language)}\n{ex['code'][:500]}...\n```\n"
            if 'ground_truth' in ex:
                context_text += f"Documentation: {ex['ground_truth']}\n"
        
        user_prompt = f"""{context_text}
### New code to document:
```{language}
{code}
```

Documentation:"""
        
        return system_prompt, user_prompt, metadata
    
    def generate_with_strategy(self, code: str, language: str, 
                               strategy: PromptStrategy,
                               model_name: str = "gpt-3.5-turbo",
                               temperature: float = 0.2) -> PromptResult:
        """
        Generate documentation using a specific prompting strategy.
        
        Args:
            code: Source code to document.
            language: Programming language.
            strategy: The prompting strategy to use.
            model_name: LLM model to use.
            temperature: Generation temperature.
            
        Returns:
            PromptResult containing the generated documentation and metadata.
        """
        system_prompt, user_prompt, metadata = self.build_prompt(code, language, strategy)
        
        # Combine prompts for the existing generator
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        metadata['model'] = model_name
        metadata['temperature'] = temperature
        
        # Generate using existing infrastructure
        # We need to call the API directly for custom prompts
        generated_doc = self._call_llm(system_prompt, user_prompt, model_name, temperature)
        
        return PromptResult(
            strategy=strategy,
            prompt=full_prompt,
            generated_doc=generated_doc,
            metadata=metadata
        )
    
    def _call_llm(self, system_prompt: str, user_prompt: str, 
                  model_name: str, temperature: float) -> str:
        """Call LLM with custom prompts."""
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        BYTEZ_KEY = os.getenv("BYTEZ_KEY")
        
        use_bytez = BYTEZ_KEY is not None and BYTEZ_KEY.strip() != ""
        
        try:
            if use_bytez:
                from bytez import Bytez
                sdk = Bytez(BYTEZ_KEY)
                model = sdk.model(f"openai/{model_name}")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                output, error, _ = model.run(messages)
                if error:
                    return f"Error: {str(error)}"
                return output.get("content", "").strip()
            else:
                import openai
                client = openai.OpenAI(api_key=OPENAI_KEY)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def compare_strategies(self, code: str, language: str,
                          strategies: Optional[List[PromptStrategy]] = None,
                          model_name: str = "gpt-3.5-turbo") -> Dict[str, PromptResult]:
        """
        Compare multiple prompting strategies on the same code.
        
        Args:
            code: Source code to document.
            language: Programming language.
            strategies: List of strategies to compare. If None, uses all.
            model_name: LLM model to use.
            
        Returns:
            Dictionary mapping strategy names to their results.
        """
        if strategies is None:
            strategies = [
                PromptStrategy.ZERO_SHOT,
                PromptStrategy.FEW_SHOT,
                PromptStrategy.CHAIN_OF_THOUGHT,
                PromptStrategy.STRUCTURED_OUTPUT
            ]
            # Only include RAG if corpus is loaded
            if self.tfidf_matrix is not None:
                strategies.append(PromptStrategy.RAG_ENHANCED)
        
        results = {}
        for strategy in strategies:
            print(f"Testing strategy: {strategy.value}...")
            result = self.generate_with_strategy(code, language, strategy, model_name)
            results[strategy.value] = result
        
        return results


def demo():
    """Demonstrate the prompt optimizer capabilities."""
    print("=" * 70)
    print("PROMPT OPTIMIZER DEMO")
    print("=" * 70)
    
    # Sample code for testing
    test_code = """def find_median(numbers):
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 0:
        return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    return sorted_nums[mid]"""
    
    optimizer = PromptOptimizer()
    
    print("\n📝 Test Code:")
    print(test_code)
    
    print("\n🔄 Comparing Prompting Strategies...")
    print("-" * 70)
    
    # Compare strategies
    results = optimizer.compare_strategies(test_code, "python")
    
    for strategy_name, result in results.items():
        print(f"\n### Strategy: {strategy_name.upper()}")
        print(f"Generated Documentation:\n{result.generated_doc}")
        print("-" * 50)


if __name__ == "__main__":
    demo()
