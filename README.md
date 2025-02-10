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

- **Đường dẫn đến trọng số huấn luyện Faster_RCNN 101(teacher):** https://drive.google.com/file/d/19xWost3a5QFDegu65mAv7fiN8Lb7EgY1/view?usp=drive_link
- **Đường dẫn đến bộ dữ liệu VOC0712 được sử dụng trong cấu hình voc0712_faster_rcnn.py: ** https://www.kaggle.com/datasets/vijayabhaskar96/pascal-voc-2007-and-2012

---

## Get Started

### 1. Prerequisites

**Dependencies**

- Ubuntu >= 20.04
- CUDA >= 11.3
- pytorch==1.12.1
- torchvision=0.13.1
- mmcv==2.0.0rc4
- mmengine==0.7.3

Our implementation based on MMDetection==3.0.0rc6. For more information about installation, please see the [official instructions](https://mmdetection.readthedocs.io/en/3.x/).

**Step 0.** Create Conda Environment

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 1.** Install [Pytorch](https://pytorch.org)

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

**Step 2.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install "mmengine==0.7.3"
mim install "mmcv==2.0.0rc4"
```
**Step 3.** Install [CrossKD](https://github.com/HuynhThoaiK27/CrossKD_FasterRCNN.git).

```shell
git clone https://github.com/HuynhThoaiK27/CrossKD_FasterRCNN
cd CrossKD_FasterRCNN
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**Step 4.** Prepare dataset follow the [official instructions](https://mmdetection.readthedocs.io/en/3.x/user_guides/dataset_prepare.html).


### 2. Training

**Single GPU**

```shell
python tools/train.py configs/crosskd/${CONFIG_FILE} [optional arguments]
```

**Multi GPU**

```shell
CUDA_VISIBLE_DEVICES=x,x,x,x python tools/dist_train.sh \
    configs/crosskd/${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

### 3. Evaluation

```shell
python tools/test.py configs/crosskd/${CONFIG_FILE} ${CHECKPOINT_FILE}
```

