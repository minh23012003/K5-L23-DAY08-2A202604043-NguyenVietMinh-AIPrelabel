# Thông tin học viên

- Họ và tên: Nguyễn Viết Minh
- MSHV: 2A202604043
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: minh23012003
- Repo bài nộp (URL): https://github.com/minh23012003/K5-L23-Day08-NguyenVietMinh-2A202604043-AI-Assisted-Labeling
- Commit chốt bài (hash): 72e8768
- Máy: Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (CUDA)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 552 / 13 / 74 = 86.4% / 2.0% / 11.6% (mẫu số: 639 box máy); added: 32 = 5.4% (mẫu số: 597 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 52.8% (47/89) / 90.4% (47/52 đối tượng gold)
- Bắt bẫy: 10/12 = 83.3%
- Thời gian: gán tay 1.25 phút/ảnh (5 phút / 4 ảnh); duyệt có pre-label 5 phút/ảnh (120 phút / 24 ảnh)
- Lỗi QC: 60.0% theo ảnh (3/5), 2.6% theo đối tượng (3/117)
- **Quyết định:** Rework — lý do một dòng: ảnh lỗi QC 3/5 = 60% > 2% (bắt bẫy 10/12 = 83.3% đã đạt); lỗi chủ yếu là box sai kích thước
