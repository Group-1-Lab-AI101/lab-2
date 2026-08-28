# Lab 2 — Decision Tree Modeling and Improvement

Dự án sử dụng bộ dữ liệu **Bank Marketing** để xây dựng, phân tích và cải thiện
mô hình Decision Tree. Hiện repository đã ghép hai phần:

- **Khang:** dataset, EDA, target encoding, stratified train/test split và
  preprocessing pipeline dùng chung;
- **Hoàng:** baseline Decision Tree cố định, đánh giá, trực quan hóa cây,
  feature importance và phân tích cây.

Chưa có code thử nghiệm cải thiện của Hậu, Kiệt hoặc Trung.

## Phân công

| Thành viên | Phần phụ trách |
| --- | --- |
| Khang | Dataset, EDA, encoding, preprocessing và train/test split |
| Hoàng | Frozen baseline, metrics, confusion matrix, tree visualization và tree analysis |
| Hậu | `max_depth`, `min_samples_split`, `min_samples_leaf` và validation |
| Kiệt | Cost-complexity pruning với `ccp_alpha` |
| Trung | Xử lý mất cân bằng bằng `class_weight`, comparison và conclusion |

## Cấu trúc chính

```text
lab-2/
├── data/
│   ├── bank-full.csv
│   └── bank-names.txt
├── docs/
│   ├── 1-KHANG.md
│   ├── 1-KHANG-ENG.md
│   └── BASELINE_HANDOFF.md
├── src/
│   ├── data.py                 # Khang: load, validate, target mapping, split
│   ├── preprocessing.py        # Khang: shared one-hot pipeline
│   ├── eda.py                  # Khang: EDA report
│   ├── visualization.py        # Khang: EDA figures
│   ├── baseline_tree.py        # Hoàng: baseline and tree analysis
│   └── experiment_contract.py  # frozen comparison contract
├── tests/
├── outputs/                    # Khang's EDA outputs
├── artifacts/
│   ├── shared/                 # fitted official preprocessor + split identity
│   └── baseline/               # Hoàng's measured baseline artifacts
├── reports/
├── run_all.py                  # reproduce Khang's section
└── run_baseline.py             # reproduce Hoàng's section
```

## Cài đặt

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Dataset chính thức đã nằm tại `data/bank-full.csv`. File đã kiểm tra có SHA-256:

```text
d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d
```

## Chạy phần Khang

```bash
.venv/bin/python run_all.py
```

Chỉ kiểm tra report JSON, không sinh lại hình:

```bash
.venv/bin/python run_all.py --skip-figures
```

Tài liệu: [tiếng Việt](docs/1-KHANG.md) và
[tiếng Anh](docs/1-KHANG-ENG.md).

## Chạy phần Hoàng

```bash
.venv/bin/python run_baseline.py
```

Kết quả mặc định được ghi vào `artifacts/baseline/`; report-ready Markdown nằm
tại `reports/hoang_baseline_and_tree_analysis.md`. Hợp đồng ghép và ranh giới
ownership nằm trong [docs/BASELINE_HANDOFF.md](docs/BASELINE_HANDOFF.md).

Baseline là `DecisionTreeClassifier` không tune, giữ nguyên mặc định và chỉ cố
định `random_state=42`. Split và pipeline được lấy trực tiếp từ code của Khang.

## Pipeline chung cho các thành viên tiếp theo

Không tạo `train_test_split` hoặc encoder riêng. Dùng trực tiếp:

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

Target được Khang ánh xạ `no → 0`, `yes → 1`. Encoder chỉ được fit trên
`X_train`; test set chỉ được transform và đánh giá. Có thể dùng
`build_model_pipeline(estimator)` khi validation/cross-validation cần đảm bảo
preprocessing được fit riêng trong từng training fold.

## Kiểm thử

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Hợp đồng so sánh cố định:

- dataset: `data/bank-full.csv`;
- target: `y`, positive class `yes` (`1`);
- stratified split: 80/20;
- `random_state=42`;
- 36,168 training rows, 9,043 test rows;
- 51 transformed features.
