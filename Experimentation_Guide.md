# Experimentation and Reproducibility Guide

This document provides detailed instructions for running the experimental modules located in the `experiments/` directory.

## 1. Batch Execution (`run_batch.py`)

This is the primary script for Phase 2. It processes the dataset sequentially to respect API rate limits.

### Usage
```bash
python experiments/run_batch.py [FLAGS]
````

### Arguments

  * `--sample [INT]`: The number of items to process. **Recommendation:** Refer to `Sampling guide.md` for statistical significance. Default is 500.
  * `--delay [FLOAT]`: Seconds to sleep between API calls. Default is `2.0`. Increase this if you encounter 429 Errors.
  * `--workers [INT]`: *Legacy flag*. Ignored in the current sequential version but kept for backward compatibility.

### Outputs

  * **Logs:** `experiments/logs/checkpoint_log.csv` (Progress saved here).
  * **Results:** `experiments/results/final_hallucination_report.csv` (Final detailed report).

-----

## 2\. Parameter Sweep (`run_param_sweep.py`)

This script runs a smaller subset of data across multiple temperatures (0.2, 0.5, 0.8) to visualize the stability of the model.

### Usage

```bash
python experiments/run_param_sweep.py
```

### Outputs

  * **Graph:** `experiments/results/param_comparison.png`
  * **Data:** `experiments/results/parameter_sweep.csv`

-----

## 3\. Visualization (`visualize.py`)

Generates the distribution plots for Metric Accuracy and Error Types found in the main report.

### Usage

```bash
python experiments/visualize.py
```

  * *Prerequisite:* You must have run `run_batch.py` first.

### Outputs

  * `experiments/results/metric_dist.png`
  * `experiments/results/error_dist.png`