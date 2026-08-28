# Improvement Methods - Phương pháp 2: Cost-Complexity Pruning

## Mục tiêu và nguyên lý

Phần này do **Thái Kiệt** phụ trách. Mục tiêu là giảm hiện tượng overfitting của cây baseline không giới hạn bằng Minimal Cost-Complexity Pruning và đồng thời so sánh hai tiêu chí tách `gini` và `entropy`. Thuật toán cân bằng sai số và độ phức tạp theo `R_alpha(T) = R(T) + alpha * |T|`; `ccp_alpha` càng lớn thì mức phạt cho số lá càng mạnh và cây càng nhỏ.

## Thiết kế thí nghiệm và chống rò rỉ dữ liệu

- Giữ nguyên dataset, target `no=0`/`yes=1`, stratified train/test 80%/20% và `random_state=42` của nhóm.
- Tập test chính thức gồm 9,043 dòng được giữ nguyên cho đánh giá cuối; không dùng để chọn `ccp_alpha` hay tiêu chí tách.
- Tập train chính thức được chia tiếp theo stratified split thành 28,934 dòng inner-fit và 7,234 dòng validation. Encoder được fit riêng trên inner-fit và validation chỉ được transform.
- Với từng criterion, lấy pruning path, giữ mốc `ccp_alpha=0` làm đối chứng và lấy mẫu tối đa 40 cấu hình trải trên đường cắt tỉa. Số cấu hình thực tế: Gini: 40, Entropy: 40.
- Cấu hình pruning của mỗi criterion được chọn bằng Accuracy validation cao nhất; nếu bằng nhau thì ưu tiên cây có ít lá/node hơn. Chỉ các ứng viên có `ccp_alpha > 0` mới được xem là mô hình pruned.
- Sau khi khóa `criterion` và `ccp_alpha`, mô hình được fit lại trên toàn bộ 36,168 dòng train rồi đánh giá trên test chính thức.

## Kết quả chọn `ccp_alpha` trên validation

| Criterion | Selected `ccp_alpha` | Inner-fit Accuracy | Validation Accuracy | Validation Error | Depth | Leaves | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Entropy | 0.00026500744 | 0.912041 | 0.904479 | 0.095521 | 18 | 135 | 269 |
| Gini | 0.00014140443 | 0.916223 | 0.905861 | 0.094139 | 15 | 110 | 219 |

![Quan hệ giữa pruning, kích thước cây và Accuracy](../outputs/figures/kiet_pruning_tradeoff.png)

Đồ thị cho thấy `ccp_alpha` làm giảm dần số lá và thường thu hẹp chênh lệch train-validation. Cắt tỉa quá ít vẫn giữ nhiều nhánh đặc thù của tập train; cắt tỉa quá mạnh làm mất các quy tắc hữu ích và gây underfitting. Dấu sao là cấu hình được chọn cho từng criterion.

## Kết quả trên tập test giữ lại

| Model | Criterion | `ccp_alpha` | Accuracy | Error Rate | Precision (`yes`) | Recall (`yes`) | F1 (`yes`) | ROC-AUC | Depth | Leaves | Nodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (unpruned) | Gini | 0 | 0.873714 | 0.126286 | 0.461111 | 0.470699 | 0.465856 | 0.698906 | 34 | 2,876 | 5,751 |
| Unpruned Entropy | Entropy | 0 | 0.883335 | 0.116665 | 0.501463 | 0.485822 | 0.493519 | 0.710914 | 37 | 2,744 | 5,487 |
| Pruned Gini | Gini | 0.00014140443 | 0.901250 | 0.098750 | 0.602740 | 0.457467 | 0.520150 | 0.907433 | 18 | 81 | 161 |
| Pruned Entropy | Entropy | 0.00026500744 | 0.905231 | 0.094769 | 0.604361 | 0.550095 | 0.575952 | 0.915281 | 18 | 111 | 221 |

Hai dòng unpruned tách riêng ảnh hưởng của criterion; hai dòng pruned cho thấy ảnh hưởng kết hợp của criterion và `ccp_alpha`. Accuracy và Error Rate được báo cáo theo yêu cầu đề bài, còn Precision/Recall/F1/ROC-AUC giúp tránh kết luận sai khi lớp `yes` là lớp thiểu số.

Xét riêng số liệu mô tả trên test, **Pruned Entropy** đạt Accuracy cao nhất (0.905231) và Error Rate thấp nhất (0.094769). Tuy nhiên, cấu hình chính vẫn được khóa bằng validation: Gini có Validation Accuracy 0.905861 so với 0.904479 của Entropy. Không đổi lựa chọn sau khi xem test giúp tránh tối ưu gián tiếp trên tập test.

## Cây đã cắt tỉa được chọn

Tiêu chí cuối cùng được khóa theo validation là **Gini**, với `ccp_alpha=0.00014140443`. Hình dưới chỉ hiển thị các tầng đầu để đọc được nhãn; các thống kê depth/leaves/nodes trong bảng được tính trên toàn bộ cây đã fit.

![Các tầng đầu của cây đã cắt tỉa](../outputs/figures/kiet_pruned_tree_top_levels.png)

## Phân tích và giải thích kết quả

- Pruned Gini làm Accuracy tăng 0.027535 điểm so với baseline, đồng thời giảm 97.18% số lá (2,876 xuống 81).
- Pruned Entropy làm Accuracy tăng 0.031516 điểm so với baseline, đồng thời giảm 96.14% số lá (2,876 xuống 111).

- Cấu hình được chọn theo validation đạt Accuracy test 0.901250, cao hơn baseline 0.027535 điểm; Error Rate là 0.098750.
- Gini đo mức không thuần bằng `1 - sum(p_k^2)`, còn Entropy dùng `-sum(p_k * log2(p_k))`. Hai tiêu chí có thể chọn các split khác nhau, nên pruning path, `ccp_alpha` tối ưu và kích thước cây cũng khác nhau.
- Lợi ích chính của pruning không chỉ nằm ở Accuracy: cây nhỏ hơn giảm variance, chênh lệch train-test và chi phí diễn giải. Nếu Accuracy không tăng, kết quả vẫn cho biết mức đơn giản hóa đạt được và chỉ ra rằng cắt tỉa không tự động giải quyết mất cân bằng lớp.
- Kết luận chỉ áp dụng cho split và pipeline cố định của nhóm. Không chọn lại cấu hình bằng kết quả test để tránh test leakage.

## Tệp kết quả phục vụ ghép báo cáo

- `outputs/results/kiet_pruning_validation.csv`: toàn bộ điểm trên pruning path đã lấy mẫu.
- `outputs/results/kiet_pruning_test_comparison.csv`: bảng so sánh baseline, Gini/Entropy và pruning.
- `outputs/results/kiet_pruning_metrics.json`: cấu hình chọn, metric và audit split.
- `outputs/trees/kiet_pruned_tree_model.joblib`: mô hình pruned được chọn theo validation.
