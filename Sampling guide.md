Sampling guide

Running 30,000 samples with a chain of 2 LLM calls (Generation + Evaluation) = **60,000 API calls**.

** RISK WARNING (T\&C and Wallet):**

1.  **Cost:** Even with cheap models, 60k calls can cost **$50 - $200+ USD**. If you are using GPT-4 for the "Judge" step, this could hit **$1,000+**.
2.  **Rate Limits (429 Errors):** OpenAI/Bytez have strict "Requests Per Minute" (RPM) limits. Multithreading *will* hit these immediately if you go too fast.
3.  **Rubric Reality:** You do **not** need 30,000 samples for a Phase 2 Prototype. A "Statistically Significant" sample is **384** items (95% confidence level). Running 30,000 provides diminishing returns for your grade but maximizes your risk of being banned or going broke.

### The Solution: Multithreading + Sampling

We have updated the script below with **`ThreadPoolExecutor`** to run requests in parallel as well as adding a **`--sample`** argument.

**We strongly recommend running only 500-1000 samples.**

### How to Run This Safely

1.  **The Safe Run (Recommended for Grade):**
    Run 500 samples with 5 threads. This is statistically valid for IEEE papers and finishes in \~10 minutes.

    ```bash
    python experiments/run_batch.py --sample 500 --workers 5
    ```

2.  **The "Full" Run (High Risk):**
    If you truly want to run all 30k, keep threads low to avoid 429 errors.

    ```bash
    # Setting sample to 0 or a high number to run all
    python experiments/run_batch.py --sample 30000 --workers 3
    ```

### Guidelines Checklist:

  * **Terms of Service:** You are allowed to use multithreading. However, "Flooding" the API (thousands of requests per second) is a violation.
  * **Rate Limits:**
      * **OpenAI Tier 1:** Limit is usually \~500-3,000 RPM (Requests Per Minute).
      * **Logic:** With `workers=5`, you make approx 60-100 RPM. This is **Safe**.
      * **Logic:** With `workers=50`, you make 1000+ RPM. You will likely get banned or rate-limited immediately. **Do not set workers \> 10.**