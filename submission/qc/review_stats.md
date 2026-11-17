# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 639; đối tượng sau duyệt: 597; ngưỡng accepted: IoU >= 0.9
- Đã **loại 27 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 552 | 552/639 = 86.4% | 639 box máy |
| edited (sửa) | 13 | 13/639 = 2.0% | 639 box máy |
| deleted (xoá) | 74 | 74/639 = 11.6% | 639 box máy |
| added (vẽ thêm) | 32 | 32/597 = 5.4% | 597 đối tượng sau duyệt |

Trong số deleted + added, **7 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 25.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 340 | 5 | 32 | 5 |
| bicycle | 6 | 0 | 2 | 1 |
| motorbike | 137 | 5 | 8 | 15 |
| car | 51 | 0 | 4 | 1 |
| bus | 1 | 0 | 1 | 1 |
| truck | 4 | 0 | 2 | 5 |
| three_wheeler | 13 | 3 | 25 | 4 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 52 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 47 | 42 | 5 | 52.8% (KTC 95%: 43–63%) | 90.4% (KTC 95%: 79–96%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 10/12 = 83.3% (KTC 95%: 55–95%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 3/3
  - extra: 3/3
  - missing: 3/3
  - loose: 1/3
  - lọt: img_027.jpg (loose), img_022.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
