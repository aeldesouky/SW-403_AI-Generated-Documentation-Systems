Required Deliverables (Detailed)

1. Prototype Implementation (Minimum Viable Product)

    A functioning prototype that integrates AI into a SWE task.

    Must include:
        ~Working code
        ~Configurable model choices
        Prompt templates
        Automated logging of prompts/responses
        Clear instructions to run

2. Experimental Setup & Dataset

    A fully described and reproducible experiment including:

        Dataset
            Source (e.g., public GitHub projects, synthetic examples)
            Size and justification
            Preprocessing steps

        Experiment Configuration
            Model + version
            Decoding parameters (temperature, top-k, etc.)
            Evaluation procedure
            Scripts/notebooks for running batches

3. Early Experimental Results

    You must provide:
        Baseline metrics
        Accuracy or correctness measurements
        ~Semantic similarity scores (e.g., BLEU, ROUGE, embedding cosines)
        Failure categories
        Samples of correct vs incorrect outputs

    Graphs/tables recommended:
        Metrics per category
        Distribution of hallucinations
        Comparison across parameter settings

4. Hallucination & Error Analysis

    A structured table such as:

        | Input | Model Output | Expected | Error Type | Hallucination? | Root Cause Hypothesis |

5. Early Research Report (3–4 pages)

    Mini-paper including:

        Problem & objective
        Methods / architecture diagrams
        Dataset & experimental setup
        Results
        Analysis of model errors & hallucinations
        Plan for Phase 3
        Risks, limitations, and mitigation

    Format: IEEE.


6. GitHub Repository
    
    Must contain:

        ~Codebase
        ~Dataset or data generator
        ~Experiment scripts
        ~Logs & metrics
        README with reproducibility instructions
        Clear commit history per student

=====================================================================

PHASE 2 GRADING RUBRIC (35% Total)

1. Technical Prototype Implementation – 30%

    (10%) Functionality: Does the prototype run end-to-end?

    (10%) Architecture & Code Quality: Modularity, clarity, structured repo, reproducibility.

    (10%) Integration with AI model(s): Proper calling, error handling, appropriate prompting.


2. Experimental Design – 25%

    (10%) Dataset quality: Representativeness, justification, clarity of preparation.

    (10%) Experimental soundness: Controlled variables, consistent procedure, baselines.

    (5%) Parameter reporting: Full description of model settings, scripts, and environment.

3. Results & Analysis – 25%

    (10%) Metric quality: Accuracy, correctness, BLEU/ROUGE/cosine, test pass rates, etc.

    (10%) Insightfulness: Interpretation of why the model behaves as it does.

    (5%) Hallucination/error breakdown: Clear categories & causes.


4. Research-Writing & Report Quality – 10%

    (5%) Writing clarity: Logical flow, academic tone, precise descriptions.

    (5%) Diagrams, tables, visuals: Architecture diagrams, experiment tables, graphs.


5. Repository Quality & Individual Contribution – 10%

    (5%) Repository organization: Proper folder structure, scripts, README.

    (5%) Individual contribution: Commits, PRs, code ownership.
