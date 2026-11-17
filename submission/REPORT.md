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
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: ngưỡng được chọn và chấm trên cùng 4 ảnh gold (52 vật), nên recall ≥ 90% và precision ở ngưỡng đã chọn là con số đẹp hơn thực tế trên 24 ảnh batch. Lớp ít mẫu (bicycle 4 vật, bus 1, truck 0, đều < 5) không chọn ngưỡng riêng mà giữ mặc định 0.3. `car` không đạt recall 90% ở ngưỡng nào ≤ 0.5 (tối đa 78%, 9 vật) nên cũng dùng mặc định 0.3.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 552 | 86.4% | 639 box máy |
| edited | 13 | 2.0% | 639 box máy |
| deleted | 74 | 11.6% | 639 box máy |
| added | 32 | 5.4% | 597 đối tượng sau duyệt |

- Vì sao loại vùng bẫy: `review_stats.py` đã loại 27 đối tượng nằm quanh 12 bẫy khỏi bảng trên, vì đó là lỗi nhân tạo do `inject_traps.py` cấy vào chứ không phải lỗi của máy; giữ lại sẽ làm sai lệch AI error profile (ví dụ box `extra` bị xoá sẽ bị tính là máy vẽ thừa).

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? `three_wheeler`: 25/41 box máy bị xoá (61%, so với person 32/377 = 8.5%). YOLOE sinh nhiều box thừa / trùng và bắt nhầm: ở pilot img_002 có 1 box three_wheeler thực ra là truck, img_032 sót 3 xe ba bánh (r04 trong `model_log.csv`).
- So với gold: precision / recall của **máy** 47/89 = 52.8% (KTC 95%: 43–63%) / 47/52 = 90.4% (KTC 95%: 79–96%) (mẫu số: 52 đối tượng gold trên 4 ảnh; precision tính trên 89 box máy = 47 TP + 42 FP)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 10/12 = 83.3% (KTC 95%: 55–95%; `qc/review_stats.md`; wrong_class 3/3, extra 3/3, missing 3/3, loose 1/3); loại nào lọt: 2 bẫy `loose`: img_027 (box `motorbike` sát mép phải bị nới lỏng) và img_022 (box `person` nhỏ cao ~60 px bị nới lỏng); vì sao (bạn đã bỏ bước nào của checklist)? img_027: tôi kéo box về gần đúng chỗ (IoU 0.79 với box gốc) nhưng đổi lớp thành `three_wheeler` vì nhìn thấy 2 bánh sau song song; đáp án bẫy giữ lớp gốc của máy (`motorbike`) nên bị tính là lọt: đây là ca "sửa khác box gốc" theo REVIEW_CHECKLIST mục 4 và là CAL-03 (xe sát mép, không đếm đủ bánh, đáng lẽ phải tick `can_review`). img_022: box `person` bị nới lỏng đã bị tôi xoá thay vì kéo lại cho sát (trong file không còn box nào trùng vị trí người đó), tức là bỏ bước 4 (soi 4 cạnh và kéo lại) và xử lý như box thừa. img_028 (`missing`, 1 người ngồi trên xe máy) bị lọt ở lần duyệt đầu vì ảnh quá đông (43 vật trong lần duyệt đầu), tôi không kiểm hết từng vật (bỏ bước 0: đếm vật trước khi xem nhãn máy); đã bổ sung khi duyệt lại.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? accepted 552/639 = 86.4% cao: phần lớn box máy trông đúng nên tôi có xu hướng tin box có sẵn và lướt qua, nhất là ở ảnh đông (img_028) và với lỗi nhỏ như box lỏng (loose chỉ bắt 1/3, là loại thấp nhất). Đây là thiên kiến tự động hóa (neo nhãn + lỗi nhỏ + mệt khi duyệt liền 120 phút): các lỗi cần nhìn kỹ hình học dễ lọt hơn lỗi rõ như sai lớp hay box thừa (đều 3/3).

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 5 phút / 4 ảnh = 1.25 phút/ảnh vs duyệt 120 phút / 24 ảnh = 5 phút/ảnh. Thiên lệch (bạn gán gold trước nên đã quen cảnh): so theo ảnh không công bằng vì ảnh batch đông hơn (610 vật / 24 ảnh ≈ 25 vật/ảnh so với gold 52 vật / 4 ảnh = 13 vật/ảnh); tính theo vật: duyệt 120/610 ≈ 0.20 phút/vật vs gán tay 5/52 ≈ 0.10 phút/vật. Gán gold trước nên đã quen cảnh phố, điều này làm lô batch có lợi hơn, vậy mà duyệt vẫn chậm hơn.
- Kết luận: **có cho chất lượng, chưa tiết kiệm thời gian**: 552/639 = 86.4% box máy giữ nguyên nên máy đáng dùng cho person / motorbike / car (trừ `three_wheeler`, 25/41 box bị xoá, phải duyệt kỹ); nhưng duyệt ≈ 0.20 phút/vật chậm hơn gán tay ≈ 0.10 phút/vật, nên chưa thấy lợi về thời gian.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Ẩn nhãn máy và đếm vật trước ở 100% ảnh, ghi số đếm của từng ảnh vào log trước khi hiện nhãn.
  2. Phóng to soi 4 cạnh của mọi box và quét riêng dải sát mép ảnh (2/3 bẫy `loose` bị lọt: img_027 xe máy sát mép phải ảnh, img_022 người nhỏ cao ~60 px).
  3. Mọi xe không đếm đủ bánh hoặc bị cắt mép ảnh phải tick `can_review` (lần này 0 ca được tick); kiểm bằng số box `can_review` trong export.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 pos `400,480` `600,110` + 1 neg `720,300`) | có: dính người lái | bám viền xe tốt hơn MobileSAM nhưng mask lan sang người lái (45.4% ảnh, 1 polygon): 1 click âm chưa đủ tách người lái | `tuktuk_img013_sam2.jpg` |
| img_013 | MobileSAM (`mobile_sam.pt`, SAM 1) | 3 (cùng click như trên) | ít hơn SAM 2.1 | thiếu phần viền xe; mask vỡ thành 2 polygon (35.2% ảnh) | `tuktuk_img013_sam1.jpg` |
| img_004 | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 pos `620,480` `700,560` + 1 neg `465,450`) | không ghi nhận | không ghi nhận lỗi (13.3% ảnh, 1 polygon) | `tuktuk_img004_sam2.jpg` |
| img_017 | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 pos `560,420` `680,500` + 1 neg `645,355`) | không ghi nhận | không ghi nhận lỗi (12.1% ảnh, 1 polygon) | `tuktuk_img017_sam2.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? **Giúp:** mục 4.2.1 (polygon bám biên phần nhìn thấy) dùng để so SAM 2.1 (bám viền tốt) với MobileSAM (thiếu viền xe); quy tắc người và xe là 2 vật riêng (mục 2 dòng motorbike, mục 4.1.4, mục 5 ca xích lô) cho biết mask img_013 bản SAM 2.1 là sai vì dính người lái và phải tách bằng `--neg`. **Thiếu:** mục 4.2.4 chỉ nói vùng chạm thuộc vật ở phía trước; khi người lái ngồi bên trong khung tuk-tuk (một phần trước khung một phần sau khung như img_013) guideline chưa nói cách xác định vật nào ở phía trước.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: không thử; chỉ dùng `scripts/sam2_points.py` offline (click `--pos` / `--neg` trên GPU).

## 6. Calibration (Part 6)

- Số quyết định: 3 (`calibration_decisions.csv`: CAL-01 `loi_may`, CAL-02 `cau_hoi_mo`, CAL-03 `bay_lot`); quan trọng nhất: CAL-01 (tuk-tuk bị YOLO26 gọi `truck` / `car`): lỗi lặp có hệ thống, khi merge đã phải loại 11 box `car` và 6 box `truck` của YOLO26 nằm chồng lên box `three_wheeler` của YOLOE.
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: 4.2.5 (người ngồi trong khung xe ba bánh không thuộc mask xe, CAL-02, img_013); bổ sung 4.3 (đếm bánh áp dụng cho motorbike vs three_wheeler, xe bị cắt mép ảnh bắt buộc `can_review`, CAL-03, img_027); thêm ảnh ví dụ cho mục 2 (CAL-01). Câu hỏi mở 1, 2, 3, 5, 6, 7 chưa chốt vì không ghi nhận ca nào khi duyệt.
- Nguyên nhân gốc: CAL-01 do **máy** (YOLO26 không có lớp xe ba bánh); CAL-02 do **guideline mơ hồ** (4.2.4 không đủ cho người ngồi trong khung xe); CAL-03 do **guideline mơ hồ** kết hợp **tôi chưa áp quy tắc** (không tick `can_review` cho xe bị cắt mép).

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ngẫu nhiên, seed 42: img_003, img_006, img_014, img_031, img_035), 117 đối tượng; người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: 3/117 = 2.6% · theo **ảnh**: 3/5 = 60.0% (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: sai vị trí / kích thước 2 lỗi (img_003 box `three_wheeler` #4 chưa đúng kích cỡ; img_006 box `car` #88 rộng hơn xe); thừa 1 lỗi (img_031 box `person` #531). Không ghi nhận bỏ sót / sai lớp / sai thuộc tính trong mẫu.
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): 3/5 ảnh lỗi cho cận trên Wilson 95% là 88%; với 5 ảnh không thể chứng minh tỉ lệ ảnh lỗi ≤ 2% (kể cả 0/5 thì cận trên vẫn rất cao), mẫu chỉ đủ phát hiện lỗi lớn.
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): script chỉ đếm những lỗi người kiểm ghi vào `qc_review.csv` và so cấu trúc / số liệu; script không tự kiểm được box có ôm sát không, vật mà cả máy lẫn người duyệt cùng bỏ sót, hay người duyệt có làm đủ checklist 4 bước thật không.

**Quyết định: Rework**, ngày 2026-11-18. Lý do (đi theo Hold → Rework → Release): không Hold vì các ca mơ hồ gặp phải đã chốt trong v1.1 (CAL-02, CAL-03) và không có ảnh hỏng; Rework vì tỉ lệ ảnh lỗi 3/5 = 60% > 2% (`qc_report_auto.md`; bắt bẫy 10/12 = 83.3% đã đạt ngưỡng 80% nên không phải lý do), lỗi chủ yếu là box sai kích thước (2/3 lỗi).
Nếu Rework / Hold, việc cần làm tiếp: sửa 3 lỗi QC trên CVAT (kéo lại box #4 img_003 và #88 img_006, xoá #531 img_031); duyệt lại lô 24 ảnh theo checklist, phóng to soi 4 cạnh box (nhất là xe / người sát mép ảnh) và tick `can_review` theo v1.1; export `submission/labels/qc_fixed.json`, chạy lại `build_release.py` rồi rút mẫu QC mới.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Claude (Claude Code trong VS Code). Dùng để: hướng dẫn từng pha; chạy các script của bài (`prelabel.py`, `threshold_sweep.py`, `merge_prelabels.py`, `inject_traps.py`, `review_stats.py`, `sam2_points.py`, `build_release.py`, `qc_sample.py`, `qc_report.py`); chép số liệu từ output script vào CSV / REPORT; diễn đạt lại nhận xét của tôi vào `model_log.csv`, REPORT và guideline v1.1. Phần quan sát ảnh (`viz/`, `masks/`, ảnh QC), duyệt trên CVAT, chọn decision / quyết định calibration / Rework là của tôi; AI không mở ảnh của bài,  Kiểm tra: đối chiếu số trong REPORT với `submission/qc/*`, chạy `validate_labels.py` và `check_submission.py`.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? không: 4 ảnh gold gán tay trên CVAT trước khi chạy bất kỳ model nào; gold được commit trước khi chạy `--split all`.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (CUDA, torch 2.6.0+cu124), Python 3.12.10, CVAT chạy bằng Docker tại `localhost:8080`; cài đặt không vướng (`check_env.py` báo SẴN SÀNG).
- Dùng cache `data/cache/` thay vì tự chạy? không, tự chạy toàn bộ trên GPU.
- Lỗi / bug script đã gặp (nếu có): không gặp lỗi script. Lỗi khi nạp CVAT: (1) upload file `.json` lẻ báo `Failed to find dataset 'coco_instances' ... annotations/default.json` → sửa bằng cách đóng zip với đường dẫn `annotations/instances_default.json`; (2) `Label 'motorbike' is not registered for this task` vì task tạo ngoài Project nên không có nhãn → đưa task vào Project có đủ 8 nhãn rồi nạp lại.
- Một điều bạn làm khác lần sau: đọc kỹ các bước hướng dẫn có trong repo (GUIDE, REVIEW_CHECKLIST, label contract) và nghe hướng dẫn của lab coach kỹ hơn trước khi làm.
