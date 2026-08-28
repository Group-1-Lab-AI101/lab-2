# Baseline Handoff

Khang's official data, EDA, split, and preprocessing implementation is now the
shared source of truth. Hoang's frozen baseline consumes those modules directly.

# Shared Experiment Contract

| Setting | Frozen value |
| --- | --- |
| Dataset | `data/bank-full.csv` |
| Dataset SHA-256 | `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d` |
| Target | `y`; mapping `no=0`, `yes=1` |
| Positive class | `yes` (`1`) |
| Split | Stratified 80% train / 20% test on `y` |
| Training / test rows | 36,168 / 9,043 |
| `random_state` | `42` |
| Split entry point | `src.data.load_and_split_data()` |
| Preprocessing entry point | `src.preprocessing.build_preprocessing_pipeline()` |

Khang's pipeline one-hot encodes the nine categorical columns with
`handle_unknown="ignore"` and sparse output. The seven numerical columns pass
through unchanged. It must be fitted on `X_train` only; `X_test` is transform-only.
The target is removed before splitting predictors. Ordered transformed names are
retrieved with `src.preprocessing.get_encoded_feature_names()`.

The frozen constants used by Hoang are in `src/experiment_contract.py`. The
baseline CLI cannot change the dataset, split ratio, or seed. Generated shared
records are stored in `outputs/shared/`:

- `preprocessor.joblib`: Khang's fitted preprocessing pipeline;
- `split_indices.npz`: original ordered train/test row indices;
- `split_manifest.json`: checksum, split fingerprints, encoding, and feature names.

# Frozen Baseline

`src.baseline_tree.train_baseline_tree()` reads the immutable
`FROZEN_BASELINE_PARAMETERS` mapping. Its exact estimator parameters are:

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

Presentation functions use `max_depth` only to truncate an image/text view of
the already-fitted tree. They do not train a smaller estimator.

Measured held-out results:

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
| Leaves / Nodes | 2,876 / 5,751 |

ROC-AUC uses the probability returned by `predict_proba` for encoded class `1`.
Reports and figures convert encoded values back to the original `no`/`yes` names.

# How Future Members Should Reuse It

```python
from src.data import load_and_split_data
from src.preprocessing import build_preprocessing_pipeline, get_encoded_feature_names

split = load_and_split_data(test_size=0.20, random_state=42)
preprocessing = build_preprocessing_pipeline()

X_train_processed = preprocessing.fit_transform(split.X_train)
X_test_processed = preprocessing.transform(split.X_test)
feature_names = get_encoded_feature_names(preprocessing)
y_train = split.y_train
y_test = split.y_test
```

Hau, Kiet, and Trung must fit separately owned estimators on
`X_train_processed, y_train` and evaluate on the unchanged
`X_test_processed, y_test`. They must not create another split or encoder.
`build_model_pipeline(estimator)` is available for fold-safe validation.

They may reuse generic metric helpers where definitions should remain identical,
but must not modify Hoang's `train_baseline_tree()` or the frozen parameter map.
If the group intentionally changes Khang's data contract, every model must be
rerun; results from different splits or encoders must not be compared.

# Ownership Boundary

| Owner | Responsibility |
| --- | --- |
| Khang | `src/data.py`, `src/preprocessing.py`, EDA, encoding, and the shared split |
| Hoang | Frozen baseline, metrics, confusion matrix, tree figures, importance, rules, and baseline report |
| Hau | `max_depth`, `min_samples_split`, `min_samples_leaf`, and validation |
| Kiet | Cost-complexity pruning with `ccp_alpha` |
| Trung | `class_weight`, comparison, and conclusion |

# Audit Classification

## Shared infrastructure

- `src/data.py`, `src/preprocessing.py`, `src/eda.py`, `src/visualization.py`;
- `src/experiment_contract.py` and `src/artifact_utils.py`;
- `tests/test_khang_pipeline.py` plus shared integration assertions in
  `tests/test_baseline.py`;
- `outputs/shared/`.

## Hoang's work

- `src/baseline_tree.py`, `src/baseline_workflow.py`, and the baseline workflow
  exposed by `run_all.py`;
- `docs/hoang_baseline_and_tree_analysis.md`;
- baseline PNGs in `outputs/figures/`, measured tables/manifests in
  `outputs/results/`, tree/model exports in `outputs/trees/`, and
  baseline-specific tests.

## Code that should not exist yet

None. There is no tuning search, alternative estimator experiment, pruning,
class-weight experiment, or final comparison implementation.

# Files Future Members Must Not Modify

- `src/experiment_contract.py` or Khang's ordered test selection unless the whole
  group deliberately restarts every experiment;
- Hoang's frozen baseline constructor and parameters;
- baseline outputs/report except by rerunning `run_all.py`;
- held-out test labels or rows.

# Correctness Notes

The audits verify disjoint train/test indices, target exclusion, training-only
preprocessor fitting, feature-name alignment, and probability-based ROC-AUC.
Compatible dependency ranges are in `requirements.txt`; exact executed versions
are recorded in `outputs/results/run_manifest.json`.

`duration` is known only after a marketing call finishes. Keeping it matches the
selected dataset baseline, but it is unavailable for pre-call prediction. This
is a deployment-time concern, not train/test leakage; all team models must treat
the feature consistently.

Reproduce both shared and baseline outputs:

```bash
.venv/bin/python run_all.py
```

Run all tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
