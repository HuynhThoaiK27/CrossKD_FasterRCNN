# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn.functional as F
from torch import Tensor
import torch

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import InstanceList, OptInstanceList

from ..utils import images_to_levels, multi_apply, unpack_gt_instances
from .crosskd_two_stage import CrossKDTwoStageDetector


@MODELS.register_module()
class CrossKDFasterRCNN(CrossKDTwoStageDetector):




    def loss(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Union[dict, list]:
        """Tính toán loss từ một batch dữ liệu đầu vào và các mẫu dữ liệu.

        Tham số:
            batch_inputs (Tensor): Ảnh đầu vào có kích thước (N, C, H, W).
                Các ảnh này thường được chuẩn hóa theo trung bình (mean centered)
                và chia tỷ lệ độ lệch chuẩn (std scaled).

            batch_data_samples (list[:obj:`DetDataSample`]): Danh sách các mẫu dữ liệu trong batch.
                Thông thường, mỗi mẫu dữ liệu chứa các thông tin như:
                - `gt_instance`: Thông tin về các đối tượng thực tế trong ảnh.
                - `gt_panoptic_seg`: Dữ liệu phân đoạn toàn cảnh (nếu có).
                - `gt_sem_seg`: Dữ liệu phân đoạn theo ngữ nghĩa (nếu có).

        Trả về:
            dict: Một từ điển chứa các thành phần loss của mô hình.
        """

        """ Tính toán loss giữa Teacher & Student trong Faster R-CNN."""

        # Bước 1: Trích xuất đặc trưng từ Teacher & Student
        tea_x = self.teacher.extract_feat(batch_inputs)
        stu_x = self.extract_feat(batch_inputs)

        # Bước 2: Forward qua RPN Head của cả Teacher & Student  --> các vùng có k/n chứa đối tượng để đưa vào ROI
        tea_rpn_results = self.teacher.rpn_head.predict(tea_x, batch_data_samples, rescale=False)
        stu_rpn_results = self.rpn_head.predict(stu_x, batch_data_samples, rescale=False)


        # Bước 3: Forward qua ROI Head của cả Teacher & Student
        tea_roi_feats = self.teacher.roi_head.forward(tea_x, tea_rpn_results, batch_data_samples)
        stu_roi_feats = self.roi_head.forward(stu_x, stu_rpn_results, batch_data_samples)

        # Nếu ROI Head trả về tuple, lấy phần tử đầu tiên
        if isinstance(stu_roi_feats, tuple):
            stu_roi_feats = stu_roi_feats[0]
        if isinstance(tea_roi_feats, tuple):
            tea_roi_feats = tea_roi_feats[0]


        # Bước 4: Tái sử dụng đặc trưng từ Teacher (Knowledge Distillation)
        reused_cls_scores, reused_bbox_preds = self.reuse_teacher_head(tea_roi_feats, stu_roi_feats)


        # Bước 5: Tính Loss
        losses = self.loss_by_feat(tea_roi_feats, stu_roi_feats, batch_data_samples,
                               reused_cls_scores, reused_bbox_preds, tea_rpn_results, stu_rpn_results)

        return losses

    def reuse_teacher_head(self, tea_roi_feats, stu_roi_feats):
        """Tái sử dụng đầu ra của giáo viên để hỗ trợ học sinh trong ROI Head."""
        # Bước 1: Chuẩn hóa đặc trưng giữa Student và Teacher
        if len(stu_roi_feats.shape) == 2:  # Trường hợp (num_rois, feature_dim)
            stu_roi_feats = stu_roi_feats.view(stu_roi_feats.size(0), -1)  # Flatten
            tea_roi_feats = tea_roi_feats.view(tea_roi_feats.size(0), -1)
        reused_cls_feat = self.align_scale(stu_roi_feats, tea_roi_feats)


        # Bước 2: Truyền qua các fully connected layers của ROI Head
        if hasattr(self.roi_head.bbox_head, 'shared_fcs'):
            reused_cls_feat = reused_cls_feat.view(reused_cls_feat.size(0), -1)  # Đảm bảo dạng (N, feature_dim)
            for fc_layer in self.roi_head.bbox_head.shared_fcs:
                reused_cls_feat = fc_layer(reused_cls_feat)

        else:
            print("Warning: shared_fcs not found in bbox_head")


        # Kiểm tra lại kích thước trước khi đưa vào Fully Connected Layer
        expected_fc_input_dim = self.roi_head.bbox_head.fc_cls.in_features
        if reused_cls_feat.shape[1] != expected_fc_input_dim:
            print(f"Reshaping reused_cls_feat from {reused_cls_feat.shape} to (N, {expected_fc_input_dim})")
            reused_cls_feat = F.adaptive_avg_pool2d(reused_cls_feat.unsqueeze(-1).unsqueeze(-1), (1, 1))  # (N, C, 1, 1)
            reused_cls_feat = reused_cls_feat.view(reused_cls_feat.size(0), -1)  # Đảm bảo dạng (N, feature_dim)

        print(f"reused_cls_feat shape before FC: {reused_cls_feat.shape}")
        print(f"Expected input size for fc_cls: {self.roi_head.bbox_head.fc_cls.in_features}")
        print(f"Expected input size for fc_reg: {self.roi_head.bbox_head.fc_reg.in_features}")


        # Bước 3: Dự đoán phân loại và bbox từ đặc trưng tái sử dụng
        reused_cls_feat = reused_cls_feat.view(reused_cls_feat.size(0), -1)  # Flatten
        reused_cls_score = self.roi_head.bbox_head.fc_cls(reused_cls_feat)
        reused_bbox_pred = self.roi_head.bbox_head.fc_reg(reused_cls_feat)

        return reused_cls_score, reused_bbox_pred


    def align_scale(self, stu_feat, tea_feat):
        if len(stu_feat.shape) == 2:
            stu_feat = stu_feat.view(stu_feat.shape[0], -1, 1, 1)
        if len(tea_feat.shape) == 2:
            tea_feat = tea_feat.view(tea_feat.shape[0], -1, 1, 1)

        # Lấy batch_size của hai tensor
        N_s, C, H, W = stu_feat.size()
        N_t, _, _, _ = tea_feat.size()

        print(f"Batch size sau khi reshape - stu_feat: {N_s}, tea_feat: {N_t}")

        # Nếu batch_size của tea_feat lớn hơn stu_feat, cần align kích thước
        if N_t > N_s:
            tea_feat = tea_feat[:N_s]  # Cắt batch_size để khớp với stu_feat
        elif N_t < N_s:
            raise ValueError(f"Batch size mismatch: stu_feat={N_s}, tea_feat={N_t}")

        # Tính toán mean và std theo không gian
        stu_mean = stu_feat.mean(dim=(2, 3), keepdim=True)
        stu_std = stu_feat.std(dim=(2, 3), keepdim=True)
        tea_mean = tea_feat.mean(dim=(2, 3), keepdim=True)
        tea_std = tea_feat.std(dim=(2, 3), keepdim=True)

        # Kiểm tra kích thước trước khi nhân
        assert stu_feat.shape == tea_std.shape, f"Mismatch: stu_feat={stu_feat.shape}, tea_std={tea_std.shape}"

        # Chuẩn hóa feature của học sinh dựa trên teacher
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        stu_feat = stu_feat * tea_std + tea_mean

        return stu_feat




    def loss_by_feat(self, tea_roi_feats, stu_roi_feats, batch_data_samples,
                 reused_cls_scores, reused_bbox_preds, tea_rpn_results, stu_rpn_results):
        """Tính toán loss giữa giáo viên và học sinh trong Faster R-CNN"""

        # **Loss KD cho RPN**
        loss_rpn_kd = self.loss_rpn_kd(stu_rpn_results, tea_rpn_results)

        # **Loss KD cho ROI Head**
        loss_roi_cls_kd = self.loss_cls_kd(reused_cls_scores, self.roi_head.fc_cls(stu_roi_feats))
        loss_roi_reg_kd = self.loss_reg_kd(reused_bbox_preds, self.roi_head.fc_reg(stu_roi_feats))

        losses = dict(
        loss_rpn_kd=loss_rpn_kd,
        loss_roi_cls_kd=loss_roi_cls_kd,
        loss_roi_reg_kd=loss_roi_reg_kd
        )

        print("Loss RPN KD:", self.loss_rpn_kd)
        print("Loss ROI Cls KD:", self.loss_cls_kd)
        print("Loss ROI Reg KD:", self.loss_reg_kd)


        return losses
