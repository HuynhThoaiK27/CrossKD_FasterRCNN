# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList

from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..utils import images_to_levels, multi_apply, unpack_gt_instances

from .two_stage import TwoStageDetector

@MODELS.register_module()
class CrossKDTwoStageDetector(TwoStageDetector):
    r"""Triển khai cơ chế Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Class này hỗ trợ **Knowledge Distillation (KD)** cho các mô hình **nhận diện hai giai đoạn**
    (Faster R-CNN, Mask R-CNN, Cascade R-CNN). KD giúp **truyền đạt kiến thức từ mô hình giáo viên
    (teacher model) sang mô hình học sinh (student model)** nhằm cải thiện hiệu suất của mô hình nhỏ hơn.

    Args:
        backbone (:obj:`ConfigDict` hoặc dict): Mô-đun backbone để trích xuất đặc trưng từ ảnh đầu vào.

        neck (:obj:`ConfigDict` hoặc dict): Mô-đun neck để kết hợp thông tin từ nhiều mức độ đặc trưng.
            Thường sử dụng FPN hoặc BiFPN.

        rpn_head (:obj:`ConfigDict` hoặc dict): Mô-đun RPN (Region Proposal Network) dùng để tạo ra
            các đề xuất hộp giới hạn.

        roi_head (:obj:`ConfigDict` hoặc dict): Mô-đun ROI Head, chịu trách nhiệm phân loại và tinh chỉnh
            hộp giới hạn từ các đề xuất của RPN.

        teacher_config (:obj:`ConfigDict` | dict | str | Path): Đường dẫn đến tệp cấu hình của mô hình giáo viên
            hoặc đối tượng cấu hình của mô hình giáo viên.

        teacher_ckpt (str, tùy chọn): Đường dẫn đến tệp checkpoint của mô hình giáo viên.
            Nếu không được cung cấp (`None`), mô hình sẽ không tải trọng số từ checkpoint.
            Mặc định là `None`.

        eval_teacher (bool): Thiết lập chế độ **training** hoặc **evaluation** cho mô hình giáo viên.
            Nếu `True`, mô hình giáo viên sẽ được cố định (`eval()` mode).
            Mặc định là `True`.

        train_cfg (:obj:`ConfigDict` hoặc dict, tùy chọn): Cấu hình huấn luyện cho mô hình hai giai đoạn.
            Bao gồm các tham số như cấu hình huấn luyện cho **RPN** và **ROI Head**. Mặc định là `None`.

        test_cfg (:obj:`ConfigDict` hoặc dict, tùy chọn): Cấu hình kiểm thử của mô hình.
            Bao gồm các tham số như ngưỡng NMS và chiến lược chọn hộp dự đoán. Mặc định là `None`.

        data_preprocessor (:obj:`ConfigDict` hoặc dict, tùy chọn): Cấu hình bộ tiền xử lý dữ liệu.
            Bao gồm các bước như chuẩn hóa hình ảnh, thay đổi kích thước và chuyển đổi định dạng dữ liệu.
            Mặc định là `None`.
    """
    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        rpn_head: ConfigType,
        roi_head: ConfigType,
        teacher_config: Union[ConfigType, str, Path],
        teacher_ckpt: Optional[str] = None,
        kd_cfg: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor)

        # Tải mô hình teacher
        if isinstance(teacher_config, (str, Path)):
            teacher_config = Config.fromfile(teacher_config)
        self.teacher = MODELS.build(teacher_config['model'])

        if teacher_ckpt is not None:
            load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')

        # In order to reforward teacher model,
        # set requires_grad of teacher model to False

        self.freeze(self.teacher) # không cập nhật trọng số cho mô hình teacher

        # Khởi tạo các KD Loss Functions
        self.loss_cls_kd = MODELS.build(kd_cfg['loss_cls_kd'])
        self.loss_reg_kd = MODELS.build(kd_cfg['loss_reg_kd'])

        self.loss_rpn_kd = MODELS.build(kd_cfg['loss_rpn_kd'])  # Loss KD cho RPN

        self.with_feat_distill = False

        if kd_cfg.get('loss_feat_kd', None):
            self.loss_feat_kd = MODELS.build(kd_cfg['loss_feat_kd'])
            self.with_feat_distill = True

        self.reused_teacher_head_idx = kd_cfg['reused_teacher_head_idx']

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Hỗ trợ chuyển teacher model sang GPU/CPU đồng bộ với student."""
        self.teacher.cuda(device=device)
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Hỗ trợ chuyển teacher model sang GPU/CPU đồng bộ với student."""
        self.teacher.to(device=device)
        return super().to(device=device)


    def train(self, mode: bool = True) -> None:
        """Giữ mô hình teacher ở chế độ eval() khi training."""
        self.teacher.train(False)
        super().train(mode)


    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
