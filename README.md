# Lab 2 — Decision Tree Modeling and Improvement

Dự án sử dụng bộ dữ liệu **Bank Marketing** để xây dựng, phân tích và cải thiện
mô hình Decision Tree. Repository hiện có năm phần hoàn chỉnh chạy từ cùng một
entry point:

- **Khang:** dataset, EDA, target encoding, stratified train/test split và
  preprocessing pipeline dùng chung;
- **Hoàng:** baseline Decision Tree cố định, đánh giá, trực quan hóa cây,
  feature importance và phân tích cây;
- **Hậu:** validation và combined search cho `max_depth`,
  `min_samples_split`, `min_samples_leaf`, sau đó so sánh với frozen baseline;
- **Kiệt:** Cost-Complexity Pruning, chọn `criterion` và `ccp_alpha` bằng
  stratified five-fold F1 cross-validation, rồi phân tích trade-off kích thước.
- **Trung:** kết hợp structural controls đã chọn trước với `class_weight`, chọn
  weight bằng training-only F1 cross-validation và tổng hợp kết luận của nhóm.
- **Deployment sensitivity:** vừa controlled ablation vừa retune đủ 216 cấu hình
  sau khi loại `duration`; đây không phải improvement method thứ tư.
- **Uncertainty audit:** paired stratified bootstrap 2.000 lần cho Accuracy, F1
  và chênh lệch F1 trên cùng test rows.

## Phân công

| Thành viên | Phần phụ trách |
| --- | --- |
| Khang | Dataset, EDA, encoding, preprocessing và train/test split |
| Hoàng | Frozen baseline, metrics, confusion matrix, tree visualization và tree analysis |
| Hậu | `max_depth`, `min_samples_split`, `min_samples_leaf` và validation |
| Kiệt | Cost-complexity pruning với `ccp_alpha` |
| Trung | `class_weight`, comparison, conclusion và notebook thuyết trình |

## Cấu trúc chính

```text
lab-2/
├── data/
│   ├── bank-full.csv
│   └── bank-names.txt
├── notebooks/
│   ├── khang_data_eda_preprocessing.ipynb # dataset, EDA và shared preprocessing
│   ├── 2-BASELINE-DECISION-TREE.ipynb    # English presentation notebook
│   ├── kiet_pruning_visualization.ipynb  # thí nghiệm pruning và trực quan hóa
│   └── trung_class_weight_and_comparison.ipynb # class weight và kết luận
├── docs/
│   ├── 1-KHANG.md
│   ├── 1-KHANG-ENG.md
│   ├── BASELINE_HANDOFF.md
│   ├── hoang_baseline_and_tree_analysis.md
│   ├── hau_improvement_methods.md
│   ├── kiet_pruning_and_criterion.md
│   └── trung_class_weight_and_conclusion.md
├── src/
│   ├── data.py                 # load, validate, target mapping, split
│   ├── preprocessing.py        # shared one-hot pipeline
│   ├── eda.py                  # EDA report
│   ├── visualization.py        # EDA figures
│   ├── baseline_tree.py        # baseline model and tree analysis
│   ├── evaluation.py           # shared metrics and confusion matrices
│   ├── baseline_workflow.py    # internal baseline orchestration
│   ├── hau_hyperparameter_tuning.py # Hậu training-only CV/search workflow
│   ├── precall_sensitivity.py  # controlled analysis without duration
│   ├── pruning_experiment.py   # Kiet: ccp_alpha, Gini/Entropy, size trade-off
│   ├── trung_class_weight.py   # Trung: class weighting và team comparison
│   ├── statistical_comparison.py # paired bootstrap confidence intervals
│   ├── report_tables.py        # auto-generated report metric blocks
│   └── experiment_contract.py  # frozen comparison contract
├── tests/
├── outputs/
│   ├── figures/                # EDA và figures của mọi model experiment
│   ├── shared/                 # official preprocessor and split identity
│   ├── results/                # metrics, tabular reports, audits and manifests
│   └── trees/                  # fitted trees, DOT, rules and structure data
├── requirements.txt
├── requirements-notebook.txt
├── requirements-lock.txt       # exact verified environment, including transitive packages
└── run_all.py                  # unified project entry point
```

Toàn bộ file sinh ra nằm dưới `outputs/`, toàn bộ tài liệu nằm dưới `docs/`, và
`run_all.py` là runner duy nhất của dự án.

## Yêu cầu môi trường

- Windows, macOS hoặc Linux;
- Python 3.10 trở lên (đã kiểm tra ở 3.14.4);
- Graphviz không bắt buộc: file DOT vẫn được xuất mà không cần cài Graphviz;
- khuyến nghị dùng `uv` để dependency luôn được cài trong `.venv`, không cài vào
  Python hệ thống.

Dataset chính thức nằm tại `data/bank-full.csv`, SHA-256:

```text
d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d
```

## Tạo môi trường bằng `uv` — khuyến nghị

Các lệnh này giống nhau trên Linux, macOS và Windows, chạy tại thư mục gốc dự
án. `uv` tự tìm `.venv` nên không cần activate trước khi cài hoặc chạy:

```bash
uv venv --python 3.14.4
uv pip install -r requirements-lock.txt
uv run python run_all.py
```

Nếu muốn activate thủ công:

| Nền tảng / shell | Lệnh |
| --- | --- |
| Linux hoặc macOS (`bash`, `zsh`) | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

## Cài đặt bằng Python/pip

### Linux hoặc macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python run_all.py
```

### Windows PowerShell hoặc Command Prompt

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe run_all.py
```

Tất cả cách cài đặt trên đều đưa dependency vào `.venv` của dự án.
`requirements-lock.txt` tái tạo đúng môi trường đã chạy kiểm thử; `requirements.txt`
giữ các khoảng phiên bản runtime, còn `requirements-notebook.txt` bổ sung công
cụ Jupyter để thuận tiện khi chủ động nâng cấp và kiểm tra lại.

## Chạy workflow duy nhất

`run_all.py` luôn tái tạo tuần tự phần Khang, baseline của Hoàng, thí nghiệm
hyperparameter tuning của Hậu, ablation và retuning không có `duration`, pruning
của Kiệt, hybrid class weighting của Trung, bootstrap uncertainty và các bảng
metric tự sinh trong `REPORT.md`:

Phần dữ liệu và preprocessing được giữ ở dạng module Python dùng chung; notebook
trình bày chỉ import các module này và không sao chép pipeline.

```bash
uv run python run_all.py
```

Không sinh lại hình EDA nhưng vẫn chạy toàn bộ phần dữ liệu, baseline, tuning,
pre-call sensitivity, pruning và class weighting:

```bash
uv run python run_all.py --skip-figures
```

Đổi thư mục đầu ra khi cần:

```bash
uv run python run_all.py \
  --figures-dir outputs/figures \
  --results-dir outputs/results \
  --trees-dir outputs/trees \
  --shared-output-dir outputs/shared \
  --baseline-report docs/hoang_baseline_and_tree_analysis.md \
  --hau-report docs/hau_improvement_methods.md \
  --kiet-report docs/kiet_pruning_and_criterion.md \
  --trung-report docs/trung_class_weight_and_conclusion.md
```

Kết quả mặc định:

- `outputs/figures/`: biểu đồ EDA, baseline và các improvement experiments;
- `outputs/shared/`: fitted preprocessor, split indices và manifest dùng chung;
- `outputs/results/`: metrics, classification report, feature importance, audit,
  Hậu validation/search, pre-call retuning, Kiệt pruning, Trung hybrid weighting,
  bootstrap intervals và comparison tables;
- `outputs/trees/`: fitted models, pre-call pipelines, DOT, rules, early splits
  và tree statistics;
- `docs/1-KHANG.md` và `docs/1-KHANG-ENG.md`: báo cáo phần Khang;
- `docs/hoang_baseline_and_tree_analysis.md`: báo cáo baseline của Hoàng;
- `docs/hau_improvement_methods.md`: phương pháp và kết quả tuning của Hậu;
- `docs/kiet_pruning_and_criterion.md`: phần Improvement Methods - Phương pháp 2;
- `docs/trung_class_weight_and_conclusion.md`: class weight và kết luận toàn nhóm;
- `docs/BASELINE_HANDOFF.md`: hợp đồng thí nghiệm và hướng dẫn bàn giao.

## Notebook phần Khang

Notebook đã chạy sẵn trình bày kiểm tra dữ liệu, EDA, class imbalance, target
encoding, official stratified split, one-hot preprocessing và leakage audit:

```bash
uv run jupyter lab notebooks/khang_data_eda_preprocessing.ipynb
```

Notebook gọi trực tiếp `src.data`, `src.eda`, `src.preprocessing` và
`src.visualization`. Chọn **Run All Cells** để tái tạo bốn hình EDA và kiểm tra
lại hợp đồng dữ liệu dùng chung; notebook không tạo split hoặc encoder riêng.

For presentation or video recording, open the English notebook
[`notebooks/2-BASELINE-DECISION-TREE.ipynb`](notebooks/2-BASELINE-DECISION-TREE.ipynb):

```bash
.venv/bin/jupyter lab
```

Select **Run All Cells**. The notebook directly reuses the shared `src.data` and
`src.preprocessing` modules, then presents the metrics, confusion matrix, ROC
curve, tree structure, feature importance, representative rules, and baseline
interpretation. It can also be executed without the graphical interface:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/2-BASELINE-DECISION-TREE.ipynb
```

`run_all.py` remains the canonical entry point for reproducible artifacts and
report material. The notebook is a presentation layer, not a second
preprocessing pipeline or an alternative experiment runner.

## Pipeline chung cho các thành viên tiếp theo

Không tạo `train_test_split` hoặc encoder riêng. Dùng trực tiếp:

```python
from src.data import load_and_split_data
from src.preprocessing import (
    build_preprocessing_pipeline,
    get_encoded_feature_names,
)

split = load_and_split_data(test_size=0.20, random_state=42)
preprocessing = build_preprocessing_pipeline()

X_train_processed = preprocessing.fit_transform(split.X_train)
X_test_processed = preprocessing.transform(split.X_test)
feature_names = get_encoded_feature_names(preprocessing)

y_train = split.y_train
y_test = split.y_test
```

Target được ánh xạ `no → 0`, `yes → 1`. Encoder chỉ được fit trên `X_train`;
test set chỉ được transform và đánh giá. Dùng `build_model_pipeline(estimator)`
khi validation/cross-validation cần preprocessing được fit riêng trong từng
training fold.

## Kiểm thử

Với `uv`:

```bash
uv run python -m unittest discover -s tests -v
```

Hoặc gọi Python trong `.venv` trực tiếp:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Trên Windows, thay `.venv/bin/python` bằng `.venv\Scripts\python.exe`.

## Notebook thí nghiệm phần Kiệt

Notebook đã chạy sẵn gồm alpha grid cố định, đường Mean CV F1, bảng test
Gini/Entropy, Accuracy/Error rate, Confusion Matrix và phân tích kích thước cây.
Cài môi trường và mở notebook bằng hai lệnh:

    uv pip install -r requirements-notebook.txt
    uv run jupyter lab notebooks/kiet_pruning_visualization.ipynb

Trong JupyterLab hoặc VS Code, chọn **Run All Cells** để tái tạo toàn bộ bảng và
biểu đồ. Notebook chỉ chạy lại workflow pruning khi thiếu kết quả thí nghiệm.

## Notebook phần Trung

Notebook đã chạy sẵn trình bày class imbalance, validation các `class_weight`,
Confusion Matrix, bảng `Comparison of Results` và `Conclusion`:

    uv run jupyter lab notebooks/trung_class_weight_and_comparison.ipynb

Notebook đọc các artifact do `src/trung_class_weight.py` sinh ra và chỉ chạy lại
workflow khi thiếu kết quả. Structural controls được khóa từ workflow Hậu trước
khi `class_weight` được chọn bằng F1 trên training-only cross-validation; test
set không được dùng để chọn weight.

Hợp đồng so sánh cố định:

- dataset: `data/bank-full.csv`;
- target: `y`, positive class `yes` (`1`);
- stratified split: 80/20;
- `random_state=42`;
- 36.168 training rows, 9.043 test rows;
- 51 transformed features.
