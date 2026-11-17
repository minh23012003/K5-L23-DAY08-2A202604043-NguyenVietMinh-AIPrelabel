# Báo cáo Day 08 — 2A202604043 - Nguyễn Viết Minh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26m.pt`) | theo lớp từ `qc/threshold_sweep.md`: person 0.15 (recall 90%, 20 vật gold); motorbike 0.25 (recall 92%, 12 vật gold); bicycle / car / bus / truck giữ mặc định 0.3 | r02: bắt được person / motorbike / car / bicycle nhưng gọi tuk-tuk là truck (img_011); r03: ngưỡng chung 0.5 làm mất person và motorbike thật ở img_007 (15 box bị loại) nên phải chọn ngưỡng riêng từng lớp |
| three_wheeler | YOLOE-26s (`yoloe-26s-seg.pt`), prompt: `tuk-tuk`, `cyclo`, `cargo tricycle` → three_wheeler | 0.15 (recall 100%, 6 vật gold; từ `qc/threshold_sweep.md`) | r04: YOLOE tìm được xe ba bánh ở img_002 / img_011 / img_032 mà YOLO26 không có lớp này (r02, r03 gọi tuk-tuk là truck); vẫn có 1 box bắt nhầm truck (img_002) và sót 3 xe (img_032) nên phải duyệt kỹ |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,motorbike=0.25,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 base (`sam2.1_b.pt`). Trên cùng bộ click ở img_013, SAM 2.1 bám viền xe tốt hơn MobileSAM (MobileSAM thiếu phần viền xe và vỡ thành 2 polygon); nhược điểm là SAM 2.1 dính người lái nhiều hơn nên cần thêm click âm. Chạy được trên RTX 3050 nên không cần bản nhẹ; SAM 3 cần token HuggingFace nên không dùng.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Tuk-tuk ở img_011 bị YOLO26 gọi là `truck` với điểm 0.77: nâng `--conf` từ 0.25 lên 0.5 (r02 → r03) box vẫn còn và vẫn sai lớp, vì ngưỡng chỉ giữ / bỏ box chứ không đổi tên lớp. Sửa bằng cách đổi model: dùng YOLOE với prompt xe ba bánh cho lớp `three_wheeler` và đặt file YOLOE **trước** YOLO26 khi `merge_prelabels.py` để box `three_wheeler` thắng box `truck` / `car` trùng chỗ (IoU ≥ 0.6).
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): điểm cao mà sai: img_011 tuk-tuk → `truck` 0.77 (còn nguyên ở ngưỡng 0.5, r03); điểm thấp mà đúng: img_007 có person và motorbike thật với điểm 0.26–0.49 bị loại khi lên ngưỡng 0.5 (r02 → r03).
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: <<>>

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 pos `400,480` `600,110` + 1 neg `720,300`) | có: dính người lái | bám viền xe tốt hơn MobileSAM nhưng mask lan sang người lái (45.4% ảnh, 1 polygon): 1 click âm chưa đủ tách người lái | `tuktuk_img013_sam2.jpg` |
| img_013 | MobileSAM (`mobile_sam.pt`, SAM 1) | 3 (cùng click như trên) | ít hơn SAM 2.1 | thiếu phần viền xe; mask vỡ thành 2 polygon (35.2% ảnh) | `tuktuk_img013_sam1.jpg` |
| img_004 | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 pos `620,480` `700,560` + 1 neg `465,450`) | không ghi nhận | không ghi nhận lỗi (13.3% ảnh, 1 polygon) | `tuktuk_img004_sam2.jpg` |
| img_017 | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 pos `560,420` `680,500` + 1 neg `645,355`) | không ghi nhận | không ghi nhận lỗi (12.1% ảnh, 1 polygon) | `tuktuk_img017_sam2.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? **Giúp:** quy tắc mask xe không gồm người lái (tách bằng `--neg`) cho biết mask img_013 bản SAM 2.1 là sai vì dính người lái; quy tắc polygon bám biên phần nhìn thấy dùng để so SAM 2.1 (bám viền tốt) với MobileSAM (thiếu viền xe). **Thiếu:** guideline chưa nói ranh giới khi người lái ngồi dính sát khung xe tuk-tuk (img_013): phần nào thuộc mask xe, phần nào thuộc người.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
