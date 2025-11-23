### Automated Documentation Generation for Legacy and Modern Software: A Comparative Analysis of LLM Performance

#### **I. Introduction & Problem Statement**
Documentation is the backbone of software maintainability, yet it is often outdated or missing. This issue is critical in legacy systems (e.g., COBOL, MUMPS) where original developers are retiring. While LLMs show promise in code understanding, they are prone to "hallucinations"—fabricating variable names or logic not present in the source. **Objective:** This study presents a prototype system that generates documentation for both Python and COBOL, evaluating the trade-off between semantic accuracy (BERTScore) and factual reliability (Hallucination Rate).

#### **II. Methodology & Architecture**
**A. System Architecture**
The system follows a Retrieval-Augmented Generation (RAG) inspired pipeline (Fig 1). It consists of three modules:
1.  **Generator:** A configurable interface using `bytez` and `openai` libraries to swap between models (GPT-3.5, GPT-4).
2.  **Evaluator:** Computes deterministic metrics (BLEU, ROUGE-L) and semantic metrics (BERTScore).
3.  **Auditor (LLM-as-a-Judge):** A secondary strong model (GPT-4) validates the output against the source code to detect specific error types (Fabrication, Omission).

**B. Dataset Construction**
We utilized a hybrid dataset approach:
* **Modern:** 20 samples from CodeSearchNet (Python) to represent standard industry code.
* **Legacy:** 20 synthetic COBOL samples generated to mirror mainframe financial logic.
* *Justification:* Synthetic COBOL was required due to the scarcity of public, well-commented legacy datasets [1].

#### **III. Experimental Setup**
* **Models:** GPT-3.5-turbo (Generation) and GPT-4 (Evaluation).
* **Parameters:** Temperature=0.2 (to minimize creative hallucinations), Top-p=1.0.
* **Procedure:** A batch script (`run_batch.py`) iterates through the dataset. For each sample, the system generates a docstring, compares it to the ground truth using BERTScore, and passes the result to the Auditor module for hallucination classification.

#### **IV. Early Results & Analysis**
**A. Quantitative Metrics**
Initial results indicate that LLMs perform significantly better on modern languages.
* **Python:** Average BERTScore of **0.82**.
* **COBOL:** Average BERTScore of **0.65**.
* *Interpretation:* The model struggles with the verbosity of COBOL's `DATA DIVISION`, often missing context regarding variable initialization.

**B. Hallucination Analysis**
Using the automated auditor, we classified errors into three categories (Table I):
1.  **Fabricated Variables:** The model inventing variable names (e.g., assuming `counter_i` exists).
2.  **Omission:** Missing critical side effects (e.g., file writing operations).
3.  **Logic Error:** Misinterpreting the control flow.
* *Finding:* COBOL samples had a higher rate of "Omission" errors, likely due to the model summarizing too aggressively.

#### **V. Risks & Future Work**
**Risks:** Reliance on synthetic data may not capture the "spaghetti code" nature of real-world legacy systems.
**Phase 3 Plan:** We intend to integrate a vector database (RAG) to allow the model to see the entire file context, rather than just isolated functions, which should reduce Omission errors.

---

### Step 3: Final Verification of Deliverable 4
Ensure your `experiments/results/batch_run_v1.csv` looks like this before you submit. This maps 1:1 to the rubric requirement:

| input_code_snippet | output_doc | ground_truth | error_type | has_hallucination | root_cause |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `def sum(a,b)...` | "Sums two nums" | "Returns sum" | No Error | False | Model correctly identified logic |
| `IDENTIFICATION...` | "Prints Hello" | "Prints Hello" | No Error | False | Simple logic correctly handled |
