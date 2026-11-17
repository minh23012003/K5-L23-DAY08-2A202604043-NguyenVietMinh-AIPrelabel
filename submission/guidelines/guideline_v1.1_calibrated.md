# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202604043 - Nguyễn Viết Minh

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 4 (phần mask) | Người ngồi trong xe ba bánh có mui che: xử lý thế nào? | Với **mask**: người lái / hành khách ngồi bên trong khung tuk-tuk là vật riêng; cắt toàn bộ phần nhìn thấy của người khỏi mask xe; mask xe chỉ gồm khung + mái + bánh, phần khung bị người che để trống. (Ngưỡng "còn thấy ≥ 20%" cho **box** `person` giữ nguyên như v1.0.) | img_013 | CAL-02 |
| 1, 2, 3, 5, 6, 7 | (xem mục 6 của v1.0) | **Chưa chốt** trong v1.1: khi duyệt lô 24 ảnh không ghi nhận ca `can_review` nào thuộc các câu này, nên chưa có ảnh làm căn cứ để ra quy tắc. Giữ nguyên v1.0. | — | — |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_011, img_004, img_013 | Tuk-tuk (xe 3 bánh có mui) | YOLO26m gọi là `truck` (img_011, điểm 0.77) hoặc `car` (img_004, img_013); nâng `--conf` lên 0.5 không sửa được | Luôn gán `three_wheeler`; lấy lớp này từ YOLOE và merge YOLOE trước YOLO26 | Mục 2 v1.0 đã đúng, không đổi chữ; thêm các ảnh này làm ví dụ cho dòng `three_wheeler` (CAL-01) |
| img_013 | Người lái ngồi bên trong khung tuk-tuk, một phần trước khung một phần sau khung | SAM 2.1 (`sam2.1_b.pt`) với 1 click âm vẫn cho mask xe dính người lái | Cắt người khỏi mask xe | **Mục 4.2, thêm 4.2.5:** người ngồi bên trong khung xe ba bánh luôn thuộc mask `person`; mask xe không gồm bất kỳ phần nào của người, kể cả khi 4.2.4 ("vật phía trước") khó xác định (CAL-02) |
| img_027 | Xe bị cắt ở mép phải ảnh, không đếm đủ số bánh | Bẫy `loose` trên box `motorbike` | Tôi đã gán `three_wheeler` và không tick `can_review`; quyết định: chọn lớp tin nhất **và bắt buộc** tick `can_review` | **Mục 4.3 bổ sung:** quy tắc đếm bánh áp dụng cả cho `motorbike` vs `three_wheeler`; xe bị cắt mép ảnh không đếm đủ bánh → bắt buộc `can_review` (CAL-03) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Thêm 4.2.5 (người trong khung xe ba bánh không thuộc mask xe); bổ sung 4.3 (motorbike vs three_wheeler, xe cắt mép → `can_review`); thêm ảnh ví dụ tuk-tuk bị máy gọi car / truck cho mục 2 | CAL-02 (img_013), CAL-03 (img_027), CAL-01 (img_011, img_004, img_013) |
