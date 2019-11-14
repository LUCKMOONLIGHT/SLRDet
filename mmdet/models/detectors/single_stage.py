import torch.nn as nn

from mmdet.core import bbox2result
from mmdet.utils import draw_boxes_with_label_and_scores
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
import numpy as np


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # img_name = img_metas[0]['filename'].split('/')[-1]
        # mean = img_metas[0]['img_norm_cfg']['mean']
        # std = img_metas[0]['img_norm_cfg']['std']
        # img_array = (img.cpu().numpy()[0].transpose((1, 2, 0)) * np.array(std) + np.array(mean))
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # pos_bboxes = pos_bboxes[0].cpu().numpy()
        # pos_labels = pos_labels[0].cpu().numpy()
        # score = np.zeros_like(pos_labels)
        # draw_boxes_with_label_and_scores(img_array, pos_bboxes, score, pos_labels, img_name, 'train', 1, 0)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        # img_name = img_meta[0]['filename'].split('/')[-1]
        # mean = img_meta[0]['img_norm_cfg']['mean']
        # std = img_meta[0]['img_norm_cfg']['std']
        # img_array = (img.cpu().numpy()[0].transpose((1, 2, 0)) * np.array(std) + np.array(mean))
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        # bboxes = np.vstack(bbox_results[0])
        # labels = [  # 0-15
        #     np.full(bbox.shape[0], i + 1, dtype=np.int32)
        #     for i, bbox in enumerate(bbox_results[0])
        # ]
        # labels = np.concatenate(labels)
        # draw_boxes_with_label_and_scores(img_array, bboxes[:, :5], bboxes[:, 5], labels, img_name, 'val', 1, 0)
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
