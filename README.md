# Lab 2: Decision Tree Modeling and Improvement

This repository currently implements **Hoang's assigned scope only**:

- an untuned baseline `DecisionTreeClassifier`;
- held-out classification metrics and confusion matrix;
- full-structure and readable top-level tree visualizations;
- correctly mapped feature importances;
- tree statistics, early splits, representative rules, and report-ready analysis.

It does not implement depth/minimum-sample tuning, pruning, class weighting,
improved-model comparisons, or the group conclusion.

## Setup

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Download the official [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing), then extract `bank-full.csv` to:

```text
data/raw/bank-full.csv
```

The verified file used for the recorded run has SHA-256:

```text
d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d
```

## Run Hoang's part

```bash
.venv/bin/python run_baseline.py
```

Optional paths and reproducibility settings are available through:

```bash
.venv/bin/python run_baseline.py --help
```

The default artifacts are written to `artifacts/baseline/`, and the report-ready
Markdown is written to `reports/hoang_baseline_and_tree_analysis.md`.

The split ratio and `random_state` are intentionally not CLI options. They are
frozen in `src/experiment_contract.py` so every team member uses the identical
test partition. See `docs/BASELINE_HANDOFF.md` for the shared experiment contract
and ownership boundaries.

Run the focused tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Integration note

The repository did not contain Khang's shared loading, split, or preprocessing
code when this baseline was implemented. `src/bank_data.py` is therefore the
current shared entry point, using the frozen stratified 80/20 split and
train-only-fitted one-hot encoding. Future experiments must call
`load_shared_experiment_data()` instead of creating another split or encoder.
Khang may later extend this implementation while preserving the contract.
