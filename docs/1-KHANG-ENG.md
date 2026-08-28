# Part 1 — Khang: Dataset, EDA, and Shared Preprocessing

## 1. Scope

This part completes four tasks:

1. read and validate the structure of `data/bank-full.csv`;
2. perform exploratory data analysis on the dataset, numerical variables,
   categorical variables, and target;
3. encode categorical features and the target;
4. create a stratified train/test split and package preprocessing as a shared
   pipeline for the whole team.

Every statistic in this document is generated within the single workflow by
`python run_all.py`. The corresponding implementation is in `src/data.py`,
`src/eda.py`, `src/preprocessing.py`, and `src/visualization.py`.

## 2. Dataset Description

### 2.1. Source and prediction task

The Bank Marketing dataset was created by Paulo Cortez (University of Minho)
and Sérgio Moro (ISCTE-IUL) and released in 2012. It describes telephone-based
direct-marketing campaigns conducted by a Portuguese banking institution from
May 2008 to November 2010. A client may have been contacted more than once.

The classification task is to predict whether a client will subscribe to a term
deposit. Original source and citation information is stored in
`data/bank-names.txt`. The complete `bank-full.csv` file uses a semicolon (`;`)
delimiter.

The dataset provider requests the following citation:

> S. Moro, R. Laureano and P. Cortez (2011), “Using Data Mining for Bank Direct
> Marketing: An Application of the CRISP-DM Methodology,” Proceedings of the
> European Simulation and Modelling Conference — ESM'2011, pp. 117–121.
> Document: <http://hdl.handle.net/1822/14838>.

### 2.2. Size and structure

| Property | Result |
|---|---:|
| Observations | 45,211 |
| Total columns | 17 |
| Input features | 16 |
| Numerical features | 7 |
| Categorical features | 9 |
| Target variable | `y` |
| True missing values (`NaN`) | 0 |
| Fully duplicated rows | 0 |

### 2.3. Variable definitions

| Variable | Type | Description |
|---|---|---|
| `age` | Numerical | Client age. |
| `job` | Categorical | Type of job. |
| `marital` | Categorical | Marital status; `divorced` includes divorced and widowed clients. |
| `education` | Categorical | Education level. |
| `default` | Binary | Whether the client has credit in default. |
| `balance` | Numerical | Average yearly balance in euros. |
| `housing` | Binary | Whether the client has a housing loan. |
| `loan` | Binary | Whether the client has a personal loan. |
| `contact` | Categorical | Contact communication type. |
| `day` | Numerical | Day of the month of the most recent contact. |
| `month` | Categorical | Month of the most recent contact. |
| `duration` | Numerical | Duration of the most recent contact in seconds. |
| `campaign` | Numerical | Contacts made during the current campaign, including the latest contact. |
| `pdays` | Numerical | Days since the previous campaign contact; `-1` means never contacted before. |
| `previous` | Numerical | Contacts made before the current campaign. |
| `poutcome` | Categorical | Outcome of the previous marketing campaign. |
| `y` | Target | Whether the client subscribed to a term deposit: `yes` or `no`. |

## 3. Exploratory Data Analysis

### 3.1. Target distribution

| Original label | Encoded label | Observations | Percentage |
|---|---:|---:|---:|
| `no` | 0 | 39,922 | 88.3015% |
| `yes` | 1 | 5,289 | 11.6985% |

![Target distribution](../outputs/figures/target-distribution.png)

The `no` class is approximately 7.55 times as frequent as the `yes` class. The
dataset is therefore clearly imbalanced, so accuracy should not be used as the
only evaluation metric. Later evaluation should also examine precision, recall,
F1, and ROC-AUC, particularly for the `yes` minority class.

### 3.2. Numerical summary

| Variable | Min | Mean | Median | Max |
|---|---:|---:|---:|---:|
| `age` | 18 | 40.936 | 39 | 95 |
| `balance` | -8,019 | 1,362.272 | 448 | 102,127 |
| `day` | 1 | 15.806 | 16 | 31 |
| `duration` | 0 | 258.163 | 180 | 4,918 |
| `campaign` | 1 | 2.764 | 2 | 63 |
| `pdays` | -1 | 40.198 | -1 | 871 |
| `previous` | 0 | 0.580 | 0 | 275 |

![Numerical feature distributions](../outputs/figures/numeric-distributions.png)

Large differences among the mean, median, and maximum of `balance`, `duration`,
`campaign`, `pdays`, and `previous` indicate skewed distributions and extreme
values. Decision Trees do not require feature scaling, so the numerical values
are passed through unchanged.

The value `pdays=-1` has the business meaning “the client has not been contacted
before.” It is not a missing value and is not replaced with the mean or median.

The histograms are clipped to each feature's 1st–99th percentile only to keep
the dense regions readable. This visual clipping does not alter the source data,
reported statistics, encoded matrices, or model inputs.

### 3.3. Categorical features

| Variable | Categories | Values equal to `unknown` |
|---|---:|---:|
| `job` | 12 | 288 |
| `marital` | 3 | 0 |
| `education` | 4 | 1,857 |
| `default` | 2 | 0 |
| `housing` | 2 | 0 |
| `loan` | 2 | 0 |
| `contact` | 3 | 13,020 |
| `month` | 12 | 0 |
| `poutcome` | 4 | 36,959 |

![Categorical feature overview](../outputs/figures/categorical-overview.png)

The literal string `unknown` is a category supplied by the dataset rather than
a `NaN` value. The pipeline retains it to avoid inventing information that is
not available. In particular, most `poutcome` values are `unknown`, which is
consistent with many clients having no known outcome from a prior campaign.

### 3.4. Caveat for `duration`

`duration` measures the latest call and is fully known only after that call has
ended. If the deployment goal is to predict before calling a client, using this
feature creates temporal information leakage. The current lab experiments keep
all 16 features to remain consistent with the assignment, but a model report
should state this limitation or include an additional configuration without
`duration` for a realistic pre-call prediction scenario.

## 4. Categorical Encoding

### 4.1. Method

- Target: map `no → 0` and `yes → 1`.
- Nine categorical features: apply
  `OneHotEncoder(handle_unknown="ignore")`.
- Seven numerical features: pass through without scaling.
- Fit the encoder only on `X_train`, then use that same fitted transformer for
  both train and test. This prevents data leakage.
- `handle_unknown="ignore"` allows the pipeline to process a category in test
  or future data that was not present in train.

The nine categorical variables produce 44 one-hot columns. Together with the
seven numerical variables, the transformed matrix contains 51 features.

| Matrix | Shape before encoding | Shape after encoding |
|---|---:|---:|
| Train | 36,168 × 16 | 36,168 × 51 |
| Test | 9,043 × 16 | 9,043 × 51 |

`OneHotEncoder` produces a sparse matrix to avoid unnecessary allocation as the
number of categories grows. The complete ordered list of 51 features appears in
the `encoding.feature_names` field printed by `run_all.py`.

## 5. Stratified Train/Test Split

The implementation uses `sklearn.model_selection.train_test_split` with:

```text
test_size=0.20
random_state=42
stratify=y
```

| Partition | Total | `no` / 0 | Percentage | `yes` / 1 | Percentage |
|---|---:|---:|---:|---:|---:|
| Train | 36,168 | 31,937 | 88.3018% | 4,231 | 11.6982% |
| Test | 9,043 | 7,985 | 88.3003% | 1,058 | 11.6997% |

![Class balance after stratified split](../outputs/figures/stratified-split-balance.png)

Class proportions are virtually unchanged among the full dataset, train, and
test. The two partitions have no overlapping indices, and their combined row
count remains 45,211. The split is reconstructed in memory rather than saved as
two duplicate CSV files, so every teammate uses a single source of truth.

## 6. Shared Pipeline

`src/preprocessing.py` provides:

- `build_preprocessor()`: returns an unfitted `ColumnTransformer`;
- `build_preprocessing_pipeline()`: returns a preprocessing-only pipeline for
  inspecting EDA and encoding output;
- `build_model_pipeline(estimator)`: attaches each experiment's estimator after
  the same shared preprocessing step.

Decision Tree example:

```python
from sklearn.tree import DecisionTreeClassifier

from src.data import load_and_split_data
from src.preprocessing import build_model_pipeline

split = load_and_split_data(test_size=0.20, random_state=42)
model = build_model_pipeline(
    DecisionTreeClassifier(random_state=42)
)
model.fit(split.X_train, split.y_train)
predictions = model.predict(split.X_test)
```

Combining the encoder and model in one pipeline also ensures that during
cross-validation the encoder is fitted separately on each training fold and
never observes that fold's validation data.

## 7. Reproducing the Results

After installing the dependencies described in the README, run this command
from the project root:

```bash
python run_all.py
```

The workflow prints the complete EDA, train/test class distributions, encoded
matrix dimensions, and ordered feature names as JSON. It also recreates four PNG
figures in `outputs/figures/`, then runs the baseline on the same split and
pipeline. The EDA plots visualize only statistics assigned to Khang.

To use a different figure directory or skip figure generation:

```bash
python run_all.py --figures-dir outputs/figures
python run_all.py --skip-figures
```
