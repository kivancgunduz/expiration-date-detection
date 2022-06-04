# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22)
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/DMY/Evaluation/predictor.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 5466 bytes
import torch
from FCOS.fcos_core.structures.bounding_box import BoxList
from FCOS.fcos_core.structures.boxlist_ops import cat_boxlist, boxlist_ml_nms, boxlist_nms, remove_small_boxes

class PostProcessor(torch.nn.Module):
    __doc__ = '\n    Performs post-processing on the outputs of the RetinaNet boxes.\n    This is only used in the testing.\n    '

    def __init__(self):
        super(PostProcessor, self).__init__()
        self.pre_nms_thresh = 0.2
        self.pre_nms_top_n = 100
        self.nms_thresh = 0.6
        self.fpn_post_nms_top_n = 100
        self.min_size = 0

    def forward_for_single_feature_map(self, locations, box_cls, box_regression, centerness, size):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape
        box_cls = box_cls.reshape(N, C, H, W)
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=(self.pre_nms_top_n))
        box_cls = box_cls * centerness[:, :, None]
        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
            detections = torch.stack([
             per_locations[:, 0] - per_box_regression[:, 0],
             per_locations[:, 1] - per_box_regression[:, 1],
             per_locations[:, 0] + per_box_regression[:, 2],
             per_locations[:, 1] + per_box_regression[:, 3]],
              dim=1)
            h, w = size[0], size[1]
            boxlist = BoxList(detections, (w, h), mode='xyxy')
            boxlist.add_field('labels', per_class)
            boxlist.add_field('scores', torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, size):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for idx_1, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(self.forward_for_single_feature_map(l, o, b, c, size))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            if DEVICE == 'cuda':
                result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            else:
                result = boxlist_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)
            number_of_detections = len(result)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field('scores')
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)

        return results
# okay decompiling ./Evaluation/predictor.pyc
