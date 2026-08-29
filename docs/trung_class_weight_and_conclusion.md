# Improvement Method 3 - Class Weight and Final Conclusion

## Mục tiêu

Phần này do **Trung** phụ trách. Dataset có lớp `yes` thiểu số, vì vậy Accuracy có thể che khuất việc mô hình bỏ sót khách hàng đăng ký tiền gửi. Thí nghiệm tìm `class_weight` để tăng chi phí phân loại sai lớp `yes`. The class weights are evaluated on Hau's training-CV-selected structural controls (`{'max_depth': 20, 'min_samples_split': 100, 'min_samples_leaf': 10}`), so the final model addresses both variance and class imbalance. The structural values were fixed before this weight search.

## Thiết kế thí nghiệm

- Giữ nguyên dataset, target, stratified split 80/20 và `random_state=42` của nhóm.
- Preprocessing nằm trong pipeline và được fit riêng trong từng training fold.
- Chọn weight bằng F1 lớp `yes` trên stratified 5-fold cross-validation; khi bằng nhau ưu tiên Recall cao hơn.
- Tập test chính thức chỉ được dùng một lần sau khi weight đã được khóa.
- Weight được chọn: **`no=1, yes=3`**, mean CV F1 **0.584174**, mean CV Recall **0.740488**.

![Class-weight validation](../outputs/figures/trung_class_weight_validation.png)

## Kết quả mô hình của Trung

| Metric | Result |
| --- | ---: |
| Accuracy | 0.880128 |
| Error Rate | 0.119872 |
| Precision (`yes`) | 0.491975 |
| Recall (`yes`) | 0.753308 |
| F1 (`yes`) | 0.595220 |
| ROC-AUC | 0.894534 |
| Depth | 20 |
| Leaves | 390 |
| Nodes | 779 |

![Weighted confusion matrix](../outputs/figures/trung_confusion_matrix.png)

## Comparison of Results

| Model | Accuracy | Precision yes | Recall yes | F1 yes | ROC-AUC | Depth | Leaves | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hoang Baseline | 0.873714 | 0.461111 | 0.470699 | 0.465856 | 0.698906 | 34 | 2,876 | 5,751 |
| Hau Tuned | 0.900476 | 0.595411 | 0.465974 | 0.522800 | 0.900540 | 20 | 339 | 677 |
| Kiet Pruned Entropy | 0.904456 | 0.603632 | 0.534026 | 0.566700 | 0.911403 | 19 | 143 | 285 |
| Trung Weighted + Tuned | 0.880128 | 0.491975 | 0.753308 | 0.595220 | 0.894534 | 20 | 390 | 779 |

![Team model comparison](../outputs/figures/team_model_comparison.png)

## Conclusion

So với frozen baseline, mô hình class-weight của Trung làm Recall lớp `yes` thay đổi +0.282609 và F1 thay đổi +0.129364. Class weighting hướng cây chú ý nhiều hơn tới lỗi false negative nhưng có thể đánh đổi Precision hoặc Accuracy; vì vậy mô hình phù hợp phải được chọn theo chi phí nghiệp vụ, không chỉ theo Accuracy.

Hậu cho thấy giới hạn cấu trúc có thể giảm overfitting, Kiệt cho thấy pruning giảm mạnh kích thước cây, còn Trung tập trung trực tiếp vào mất cân bằng lớp. Bảng trên dùng cùng test partition cho mọi mô hình, nhưng từng cấu hình đều được khóa bằng training-only validation trước khi xem test.
