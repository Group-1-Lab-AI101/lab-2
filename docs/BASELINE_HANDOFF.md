# Baseline Handoff

This document freezes the experiment contract established by the first member
of the repository. Khang has not implemented his officially owned data section
yet, so the current shared pipeline is intentionally minimal. It exists to make
Hoang's baseline executable and to keep every later model on the same data.

# Shared Experiment Contract

| Setting | Frozen value |
| --- | --- |
| Dataset | `data/raw/bank-full.csv` |
| Dataset SHA-256 | `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d` |
| Target column | `y` |
| Positive class | `yes` |
| Split | 80% train / 20% test |
| Split strategy | Stratified random split on `y` |
| Training rows | 36,168 |
| Test rows | 9,043 |
| `random_state` | `42` |
| Shared entry point | `src.bank_data.load_shared_experiment_data()` |

The source of truth for these constants is `src/experiment_contract.py`. The
baseline CLI deliberately does not accept alternate split ratios or seeds.

`load_shared_experiment_data()` returns one `PreparedData` bundle containing:

- `X_train_raw`, `X_test_raw`, `y_train`, and `y_test`;
- `preprocessor`, fitted only on `X_train_raw`;
- `X_train_processed` and `X_test_processed`;
- `feature_names`, obtained from the fitted
  `ColumnTransformer.get_feature_names_out()`;
- original `train_row_indices` and `test_row_indices` for split verification.

Categorical columns are encoded by
`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Numeric columns
are passed through unchanged. The raw split happens before the preprocessor is
fitted, and the test partition is transform-only. The target is removed before
splitting predictors and is absent from raw and transformed inputs.

The generated shared records are:

- `artifacts/shared/preprocessor.joblib`;
- `artifacts/shared/split_indices.npz`;
- `artifacts/shared/split_manifest.json`.

The split fingerprints in `split_manifest.json` allow another run to verify the
exact ordered row selections, not merely the same train/test counts.

# Frozen Baseline

Hoang's baseline constructor is frozen in `src.baseline_tree.py` and reads the
immutable `FROZEN_BASELINE_PARAMETERS` mapping. Its exact parameters are:

```text
criterion='gini'
splitter='best'
max_depth=None
min_samples_split=2
min_samples_leaf=1
min_weight_fraction_leaf=0.0
max_features=None
random_state=42
max_leaf_nodes=None
min_impurity_decrease=0.0
class_weight=None
ccp_alpha=0.0
monotonic_cst=None
```

The `max_depth` arguments used by tree plotting/text functions only truncate a
presentation of the already-fitted tree. They do not train another estimator or
change the baseline.

Measured results from the frozen held-out test set:

| Result | Value |
| --- | ---: |
| Accuracy | 0.8737144753 |
| Error Rate | 0.1262855247 |
| Precision (`yes`) | 0.4611111111 |
| Recall (`yes`) | 0.4706994329 |
| F1-score (`yes`) | 0.4658559401 |
| ROC-AUC | 0.6989063852 |
| Train Accuracy | 1.0000000000 |
| Test Accuracy | 0.8737144753 |
| Train-Test Gap | 0.1262855247 |
| Tree Depth | 34 |
| Leaf Count | 2,876 |
| Node Count | 5,751 |

ROC-AUC is calculated from the `yes` column returned by `predict_proba`, not
from hard predictions. The baseline is not tuned, pruned, class-weighted, or
selected using cross-validation.

# How Future Members Should Reuse It

Every future experiment must start from the same bundle:

```python
from src.bank_data import load_shared_experiment_data

data = load_shared_experiment_data()

X_train = data.X_train_raw
X_test = data.X_test_raw
y_train = data.y_train
y_test = data.y_test
preprocessing = data.preprocessor
X_train_processed = data.X_train_processed
X_test_processed = data.X_test_processed
feature_names = data.feature_names
```

Hau, Kiet, and Trung should fit their separately owned estimator only on
`X_train_processed` and `y_train`. They must evaluate final results on the
unchanged `X_test_processed` and `y_test`, and must not call `train_test_split`,
fit a second encoder, or fit preprocessing on the complete dataset.

They may reuse generic evaluation helpers from `src/baseline_tree.py` where the
metric definition is intended to remain identical. They must construct their
own model in their own module; they must not modify `train_baseline_tree()` or
`FROZEN_BASELINE_PARAMETERS`.

Khang may move, extend, or replace the provisional implementation in
`src/bank_data.py` when completing his section. That integration must preserve
the target definition, ordered train/test row identities, train-only fitting,
and transformed feature-name alignment. If Khang intentionally changes the
group's experiment contract, all experiments—including the baseline—must be
rerun; results from different splits must not be compared.

# Ownership Boundary

| Owner | Responsibility |
| --- | --- |
| Khang | Dataset loading, EDA, categorical encoding, preprocessing, and train/test split. The current `src/bank_data.py` is provisional shared infrastructure awaiting his ownership. |
| Hoang | Frozen baseline tree, held-out metrics, confusion matrix, tree visualizations, feature importance, decision rules, tree analysis, and baseline report material. |
| Hau | `max_depth`, `min_samples_split`, and `min_samples_leaf` experiments and validation. |
| Kiet | Cost-complexity pruning using `ccp_alpha`. |
| Trung | Class-imbalance work using `class_weight`, plus his assigned comparison/conclusion material. |

# Audit Classification

## A. Shared infrastructure

- `src/experiment_contract.py`
- `src/bank_data.py`
- `src/artifact_utils.py`
- environment dependencies in `requirements.txt`
- shared generated records under `artifacts/shared/`
- shared-pipeline and leakage tests in `tests/test_baseline.py`
- the shared artifact-generation portion of `run_baseline.py`

## B. Hoang's work

- `src/baseline_tree.py`
- the baseline training/report portion of `run_baseline.py`
- `reports/hoang_baseline_and_tree_analysis.md`
- all generated records under `artifacts/baseline/`
- baseline evaluation tests in `tests/test_baseline.py`

## C. Code that should not exist yet

None found. There are no tuning searches, improvement estimators, pruning
experiments, class-weight experiments, cross-validation experiments, or final
group comparisons/conclusions in the repository.

# Files Future Members Must Not Modify

- `src/experiment_contract.py`, unless the whole group deliberately restarts all
  experiments under a new contract;
- `src/baseline_tree.py` baseline constructor and frozen configuration;
- `artifacts/baseline/` and
  `reports/hoang_baseline_and_tree_analysis.md`, except by rerunning the frozen
  baseline workflow;
- the held-out test labels or row selection.

# Correctness Notes

The automated audit found no target leakage, overlapping train/test rows, or
test-time fitting. The saved preprocessor and split fingerprints match a fresh
execution. The exact verified package versions are pinned in `requirements.txt`
and recorded with the run in `artifacts/baseline/run_manifest.json`.

The dataset's `duration` field is known only after a marketing call finishes.
Keeping it is consistent with the selected dataset baseline, but it would not
be available to a system making decisions before a call. This is a potential
deployment-time information issue, not leakage between the frozen train and
test partitions. Khang and the group should describe the intended prediction
time clearly rather than silently dropping the feature from only some models.

To reproduce the baseline and refresh both shared and Hoang artifacts:

```bash
.venv/bin/python run_baseline.py
```

To verify the contract and baseline helpers:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
