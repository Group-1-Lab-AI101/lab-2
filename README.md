# Lab 2 — Decision Tree Modeling and Improvement

Dự án sử dụng bộ dữ liệu **Bank Marketing** để xây dựng và cải thiện mô hình
Decision Tree. Phần hiện đã hoàn thành là phần của **Khang**: mô tả dữ liệu,
EDA, categorical encoding, chia train/test theo stratified sampling và pipeline
tiền xử lý dùng chung. Chi tiết phương pháp và kết quả có bản
[tiếng Việt](docs/1-KHANG.md) và [tiếng Anh](docs/1-KHANG-ENG.md).

## Phân công

| Thành viên | Nhiệm vụ kỹ thuật (Code & Thực nghiệm) | Trách nhiệm Báo cáo & Thuyết trình |
|---|---|---|
| **Khang** | Tải `bank-full.csv`, EDA và mã hóa biến phân loại. Chia dữ liệu train/test bằng stratified sampling. Tạo pipeline tiền xử lý dùng chung. | Viết phần **Dataset Description**: nguồn dữ liệu, số lượng mẫu, đặc trưng và biến mục tiêu. |
| **Hoàng** | Huấn luyện baseline Decision Tree. Tính Accuracy, Error rate, Confusion Matrix, Precision, Recall, F1, ROC-AUC. Trực quan hóa cây và trích xuất feature importance. | Ghép báo cáo chính; viết **Baseline Model** và **Analysis of the Tree**. |
| **Hậu** | Thử nghiệm `max_depth`, `min_samples_split`, `min_samples_leaf` bằng validation/cross-validation. | Viết **Improvement Methods — Phương pháp 1**, tính lại tỷ lệ lỗi và giải thích hiệu quả. |
| **Kiệt** | Thử nghiệm Cost-Complexity Pruning (`ccp_alpha`). So sánh Gini và Entropy; đánh giá kích thước cây so với hiệu suất. | Viết **Improvement Methods — Phương pháp 2**, trình bày cài đặt và so sánh kết quả. |
| **Trung** | Xử lý mất cân bằng lớp bằng `class_weight="balanced"`. Đánh giá theo Recall/F1 của lớp thiểu số và tạo bảng/biểu đồ so sánh. | Quay video thuyết trình; viết **Comparison of Results** và **Conclusion**. |

## Cấu trúc dự án

```text
lab-2/
├── data/
│   ├── bank-full.csv
│   └── bank-names.txt
├── docs/
│   ├── 1-KHANG.md
│   └── 1-KHANG-ENG.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── evaluate.py          # dành cho phần đánh giá của Hoàng
│   └── utils.py
├── experiments/             # notebook của các phần sau, hiện để trống
├── outputs/
│   ├── figures/             # bốn biểu đồ EDA của Khang
│   ├── trees/               # hiện để trống
│   └── results/             # hiện để trống
├── requirements.txt
├── README.md
└── run_all.py
```

`experiments/`, `outputs/trees/` và `outputs/results/` hiện chỉ có `.gitkeep`;
chưa có notebook, cây hoặc kết quả của các thành viên khác. `outputs/figures/`
chỉ chứa bốn biểu đồ EDA thuộc phần Khang.

## Yêu cầu môi trường

- Python 3.10 trở lên
- `pandas`
- `scikit-learn`
- `matplotlib`

Không cần tạo bản sao CSV train/test. Hàm trong `src/data.py` tái tạo đúng cùng
một phép chia từ `bank-full.csv` với `test_size=0.20` và `random_state=42`.

## Cài đặt và chạy bằng `uv`

### Linux

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run python run_all.py
```

### Windows PowerShell

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv run python run_all.py
```

## Cài đặt và chạy bằng Python/pip

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_all.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` in báo cáo JSON ra terminal và tái tạo bốn biểu đồ EDA trong
`outputs/figures/`; script không ghi đè dữ liệu và không huấn luyện mô hình. Có
thể đổi đường dẫn, cấu hình chia dữ liệu hoặc thư mục hình:

```bash
python run_all.py --data data/bank-full.csv --test-size 0.2 --random-state 42
python run_all.py --figures-dir outputs/figures
```

Chỉ in JSON mà không sinh lại hình: `python run_all.py --skip-figures`.

Chạy bộ kiểm thử bằng interpreter trong môi trường ảo:

```bash
python -m unittest discover -s tests -v
```

## Dùng train/test split và pipeline chung

Các thành viên huấn luyện mô hình nên dùng đúng hai hàm dưới đây để mọi thí
nghiệm có cùng dữ liệu và cách mã hóa:

```python
from sklearn.tree import DecisionTreeClassifier

from src.data import load_and_split_data
from src.preprocessing import build_model_pipeline

split = load_and_split_data(test_size=0.20, random_state=42)
pipeline = build_model_pipeline(
    DecisionTreeClassifier(random_state=42)
)

pipeline.fit(split.X_train, split.y_train)
y_pred = pipeline.predict(split.X_test)
```

Pipeline chỉ học các category từ `X_train`; category mới ở dữ liệu kiểm tra hoặc
dữ liệu thật được bỏ qua an toàn nhờ `handle_unknown="ignore"`. Target được ánh
xạ `no → 0`, `yes → 1`.

## Nguồn dữ liệu

Bank Marketing dataset do Paulo Cortez (University of Minho) và Sérgio Moro
(ISCTE-IUL) công bố. Mô tả và trích dẫn đi kèm nằm trong
[`data/bank-names.txt`](data/bank-names.txt). Dữ liệu mô tả các chiến dịch gọi
điện tiếp thị trực tiếp của một ngân hàng Bồ Đào Nha; mục tiêu là dự đoán khách
hàng có đăng ký tiền gửi có kỳ hạn hay không.
