# Chạy lại repository đã tinh gọn

## 1. Chuẩn bị thư mục và môi trường

Giải nén `HW2_Submission_Refactored.zip` thành thư mục mới, không đè lên hw2 cũ.
Chạy lệnh ở thư mục có `run_adaptive.py`. Cài PyTorch CUDA phù hợp GPU/driver
trước, sau đó `python -m pip install -r requirements.txt`. Cần Graphviz `dot`
cho sơ đồ schema. Môi trường và quy trình đã kiểm tra nằm ở VALIDATION.md.

Code giữ đúng 7 lựa chọn: copy7, base, graph, temporal, additive, adaptive,
gnn_time. `addaptive` được nhận như alias của adaptive. Không cần chạy
run_gnn.py, Graph28 hay các script evaluate_adaptive_week cũ.

## 2. Dữ liệu

Tái sử dụng ZIP train_db/valid_db/test_db đã qua preprocessor mới của môn,
hoặc chạy `official/scripts/preprocess.py --raw RAW_FOLDER --out OUT_FOLDER`
cho từng nguồn rồi truyền `OUT_FOLDER/db` vào --db.

ZIP nguồn phải có đúng một file Parquet cho mỗi bảng; thư mục nguồn chứa trực
tiếp 8 Parquet. Nhiều nguồn truyền theo thời gian. Bản ZIP code mới không chứa
dataset, cache hay weights cũ. Các file này cần dung lượng lớn và không nên
đưa vào public GitHub; mọi bước tạo lại đều có code.

## 3. Kiểm tra lịch trước khi train

`configs/historical.json` giữ nguyên bản upload: H=7 ngày, train cutoff bắt
đầu 01/01/2026, stride 7 ngày, last bound 19/07 (cutoff cuối sinh ra là 16/07).
Không tự đổi về H=21 của thí nghiệm trước. `configs/pilot.json` riêng dùng H=21
cho kiểm tra luồng, không phải cấu hình full.

Nếu muốn H=21 hoặc H=28, chỉnh history_days trước prepare. Cần bảo đảm
train.first − max(14,H) >= observation_start, và nhãn mỗi split kết thúc trước
split tiếp theo. Đổi plan phải prepare và train lại ở output mới.

## 4. Chạy toàn bộ với một lệnh

```bash
bash scripts/run_historical.sh runs/full cuda:0 \
  data_inputs/train_db.zip data_inputs/valid_db.zip data_inputs/test_db.zip
```

Lệnh sẽ tạo EDA, schema, cache chung, train 7 model và chấm validation/test.
Nếu cần chọn hyperparameter bằng validation trước khi chấm test, dùng các
lệnh rời trong README. Mỗi output cần đường dẫn mới; không ghi đè run cũ.

Hoặc từng bước:

```bash
python run_adaptive.py prepare \
  --db data_inputs/train_db.zip data_inputs/valid_db.zip data_inputs/test_db.zip \
  --plan configs/historical.json --data runs/full/data

python run_adaptive.py train \
  --data runs/full/data --out runs/full/models --device cuda:0 \
  --epochs 16 --patience 4 --batch-size 256 --seed 2026

python run_adaptive.py evaluate \
  --data runs/full/data --models runs/full/models \
  --split evaluation --out runs/full/validation --device cuda:0

python run_adaptive.py evaluate \
  --data runs/full/data --models runs/full/models \
  --split test --out runs/full/test --device cuda:0
```

`--models base graph temporal additive adaptive gnn_time copy7` tùy chọn ở
train; mặc định đã gồm cả 7. Copy7 không học trọng số nhưng vẫn lưu checkpoint
để cùng đường predict. GNN dùng --gnn-hidden, --gnn-layers, --gnn-fanout,
--gnn-dropout, --time-dim, --time-periods; hidden của các model cũ dùng --hidden.
Chọn epoch bằng stop BCE, chọn ngưỡng bằng calibration. Giữ cùng các split
khi so sánh; không tune ngưỡng bằng test. Số tham số từng model được xuất cùng
summary, vì các kiến trúc có capacity khác nhau.

## 5. Kết quả để phân tích ablation

Trong validation/ và test/ có:

- summary.csv: 7 dòng model, macro/micro F1, Score, n, params.
- weekly.csv: Score từng cutoff và pooled.
- per_label.csv: AP, F1, TP/FP/FN, support theo nhãn và tuần.
- model/split_predictions.csv.gz: nhãn thật, xác suất, quyết định từng user-week.
- model/split_metrics.json: ngưỡng đã lưu và điểm ở ngưỡng .5.

GNN và các model cũ dùng chung file cache nên user, nhãn và cutoff thống nhất.
Đánh giá mặc định lấy batch-size checkpoint, tránh đổi cách sample neighbor
sau calibration. GPU khác hoặc phiên bản thư viện khác vẫn có thể tạo sai số.

## 6. Chấm riêng 27/09–04/10

```bash
python run_adaptive.py evaluate-week \
  --data runs/full/data --models runs/full/models \
  --db data_inputs/train_db.zip data_inputs/valid_db.zip data_inputs/test_db.zip \
  --cutoff 2026-09-27T00:00:00+08:00 \
  --out runs/full/sep27 --device cuda:0
```

Chấm cả bảy checkpoint và giữ nguyên ngưỡng; không cần train lại.

## 7. Xuất file dự đoán chính thức

Chọn model dựa trên validation. Ví dụ GNN:

```bash
python run_adaptive.py predict \
  --checkpoint runs/full/models/gnn_time/model.pt \
  --db data_inputs/test_db.zip data/recent/db \
  --users official/submission/test_input.csv \
  --cutoff 2026-11-22T00:00:00+08:00 \
  --out outputs/output_314561001.csv --device cuda:0

python official/scripts/evaluate.py \
  --pred outputs/output_314561001.csv \
  --input official/submission/test_input.csv
```

Thay gnn_time bằng copy7/base/graph/temporal/additive/adaptive để chọn model
khác. Đổi mã sinh viên nếu cần. Mục tiêu là 22/11 00:00 đến 29/11 00:00 giờ
Đài Loan; --split test chỉ chấm lịch sử tháng 10–11, không tạo bài nộp này.

Nguồn lịch sử cũ dừng 15/11: cần bổ sung các giờ trước cutoff bằng dữ liệu đã
qua official preprocess, bao phủ đủ H ngày trước 22/11. Tuyệt đối không đưa
nhãn/hoạt động tuần cần dự đoán vào input. File test_input đính kèm có 4465
user; kiểm tra bản giảng viên cập nhật trước khi nộp cuối.

Kèm CSV là metadata JSON giúp tái lập checkpoint, nguồn và cutoff; nộp CSV lên
E3, giữ metadata và weights cục bộ để reproduce. Repo này chưa sinh file dự
đoán cuối cùng vì cần lần train và lựa chọn model của bạn.

## 8. Gửi kết quả cho bước report/video

```bash
python scripts/pack_results.py --run runs/full \
  --out outputs/results_for_report.zip \
  --predictions outputs/output_314561001.csv outputs/output_314561001.csv.metadata.json
```

Gửi results_for_report.zip và cho biết checkpoint/model bạn chọn. Nếu cần
chia sẻ weights, thêm --include-checkpoints. Không cần gửi lại toàn cache .npz.
Phần slide/report/video sẽ viết sau khi có kết quả chạy thật, theo yêu cầu của
bạn; xem checklist để không thiếu đầu ra cuối.
