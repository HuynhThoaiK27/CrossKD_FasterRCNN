# 📌 CrossKD với Faster R-CNN

Đây là phiên bản tinh chỉnh của **CrossKD** để hỗ trợ **Faster R-CNN** trong MMDetection.

## **📌 Các thay đổi chính**
### 🔹 Thêm mô hình CrossKD cho Faster R-CNN:
- `mmdet/models/detectors/crosskd_faster_rcnn.py`
- `mmdet/models/detectors/crosskd_two_stage.py`
- **Cập nhật** `mmdet/models/detectors/__init__.py` để đăng ký mô hình mới.

### 🔹 Cấu hình mới:
- `configs/crosskd/crosskd_fasterRCNN_101_50.py`  (CrossKD với Faster R-CNN 101)
- **Thư mục mới:** `configs/faster_rcnn` chứa:
  - `faster-rcnn_r101_fpn_1x_voc0712.py`
  - `faster-rcnn_r50_fpn_1x_voc0712.py`
- **Thêm dataset VOC0712:** `configs/_base_/datasets/voc0712_faster_rcnn.py`

---

## **📥 Cài đặt**
### 1️⃣ **Clone repo**
```bash
git clone https://github.com/HuynhThoaiK27/CrossKD_FasterRCNN.git
cd CrossKD_FasterRCNN
