from typing import Dict, List, Optional, Tuple, Union
import os
import copy
import warnings

import PIL.Image
from torchvision import transforms
import mmcv
import json
import numpy as np
import torch
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
from mmengine.dist import master_only
from mmdet.structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from mmengine.visualization.utils import (check_type, tensor2ndarray)
from mmengine.model.base_model import BaseModel

id2class_name = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

class_name2id = {v: k for k, v in id2class_name.items()}


def extend_context(lower, upper, left, right, height, width, extend_context_value):
    """
    Extend the context of the bounding box by a certain value.
    """
    lower = lower + extend_context_value if lower + extend_context_value < height else height - 1
    upper = upper - extend_context_value if upper - extend_context_value >= 0 else 0
    right = right + extend_context_value if right + extend_context_value < width else width - 1
    left = left - extend_context_value if left - extend_context_value >= 0 else 0
    return lower, upper, left, right


def convert_to_corners_torch(bboxes):
    """
    Convert bounding boxes from (x, y, width, height) to (x_min, y_min, x_max, y_max) format using PyTorch.
    """
    bboxes_corners = torch.zeros_like(bboxes)
    bboxes_corners[:, 0] = bboxes[:, 0]  # x_min
    bboxes_corners[:, 1] = bboxes[:, 1]  # y_min
    bboxes_corners[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max
    bboxes_corners[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max
    return bboxes_corners


def calculate_iou_torch(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes using PyTorch.
    """
    x_min_inter = torch.max(bbox1[0], bbox2[0])
    y_min_inter = torch.max(bbox1[1], bbox2[1])
    x_max_inter = torch.min(bbox1[2], bbox2[2])
    y_max_inter = torch.min(bbox1[3], bbox2[3])

    intersection_area = torch.clamp(x_max_inter - x_min_inter, min=0) * torch.clamp(y_max_inter - y_min_inter, min=0)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou


def find_pairs_above_threshold_torch(bboxes1, bboxes2, threshold=0.5):
    """
    Find all pairs of bboxes where IoU is above the given threshold using PyTorch.
    """
    bboxes1_corners = convert_to_corners_torch(bboxes1)
    bboxes2_corners = convert_to_corners_torch(bboxes2)

    idx1 = set()
    idx2 = set()

    for i in range(bboxes1_corners.size(0)):
        for j in range(bboxes2_corners.size(0)):
            if calculate_iou_torch(bboxes1_corners[i], bboxes2_corners[j]) > threshold:
                idx1.add(i)
                idx2.add(j)

    return idx1, idx2


def iou_function(box1, box2):
    """
    Find IoU between bboxes
    """
    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou


class CropObject(DetLocalVisualizer):

    def __init__(self, name="visualizer", ipd=1184, img_total_number=None, img_h=578, img_w=484,
                 save_file=None, scale=1.0, max_img_h=121, max_img_w=144, save_gt=False, overlap_threshold=0.5,
                 apply_select_background=False, background_bbox_path=None, extend_context = False, extend_context_value = 5, dynamic_extend_context = False, original_coco_path=None,
                 apply_screen_iter=False, apply_select_iter=False,soft_label_model=None, soft_label_input_shape=(955, 800),
                 screen_overlap_iou=0.25, conf_threshold=0.2, screen_pipeline=None, model_type="RCNN", labels_file=None, label_type = "bbox"):
        super().__init__()
        self.ipd = ipd
        self.img_h = img_h
        self.img_w = img_w
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.img_total_number = img_total_number
        self.save_gt = save_gt
        self.overlap_iou = overlap_threshold
        self.background_bbox_path = background_bbox_path
        self.apply_select_background = apply_select_background
        self.original_coco_path = original_coco_path
        self.apply_screen_iter = apply_screen_iter
        self.apply_select_iter = apply_select_iter
        self.soft_label_input_shape = soft_label_input_shape
        self.screen_overlap_iou = screen_overlap_iou
        self.discard_thres = conf_threshold
        self.model_type = model_type
        self.labels_file = labels_file
        self.extend_context = extend_context
        self.extend_context_value = extend_context_value
        self.dynamic_extend_context = dynamic_extend_context
        self.label_type = label_type

        if self.img_total_number is None:
            raise ValueError("Must input img_total_number!")
        self.interval = int(self.img_total_number // ipd)
        if self.apply_select_background:
            assert background_bbox_path is not None, "background_bbox_path should not be None!"
            assert original_coco_path is not None, "original_coco_path should not be None!"
            self.init_background()
        if apply_screen_iter:
            assert soft_label_model is not None, "soft_label_model should not be None!"
            self.soft_label_model: BaseModel = soft_label_model
            assert screen_pipeline is not None, "screen_pipeline should not be None!"
            self.screen_pipeline = screen_pipeline
        if self.save_gt:
            self.init_json()
        self.count = 1
        self.image_id = -1
        self.init_image()
        self.save_file = save_file
        self.scale = scale

    def init_image(self):
        if hasattr(self, "large_scale_image"):
            del self.large_scale_image
        if hasattr(self, "large_scale_mask"):
            del self.large_scale_mask
        if hasattr(self, "candidates"):
            del self.candidates
        if hasattr(self, "memory"):
            del self.memory

        self.large_scale_mask = torch.zeros(1, self.img_h, self.img_w).bool().cuda()
        if self.apply_select_background:
            background_path = os.path.join(self.original_coco_path, self._background[self._b_counter][1])
            img = PIL.Image.open(background_path).convert("RGB")
            self._bg = tensor = self._transform(img) * 255  # 3, H, W
            self.large_scale_image = tensor.clone()
            self.large_scale_image_2 = tensor.clone()
            self._b_counter += 1
        else:
            self.large_scale_image = torch.ones(3, self.img_h, self.img_w).cuda() * 255
            self.large_scale_image_2 = torch.ones(3, self.img_h, self.img_w).cuda() * 255
            self._bg = torch.ones(3, self.img_h, self.img_w).cuda() * 255
        self.candidates = []
        self.memory = []
        self.image_id += 1

    def init_background(self):
        with open(self.background_bbox_path, "r") as ff:
            p = ff.readlines()
            p = [i.strip().split(", ") for i in p]
            p = [[float(i[0]), i[1]] for i in p]
            p = sorted(p, key=lambda x: x[0], reverse=True)
        self._b_counter = 0
        self._background = p
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_h, self.img_w))
        ])

    def init_json(self):
        self.large_scale_json = dict()  # include label and bbox
        self.large_scale_json["annotations"] = []  # annotations: {'image_id': xx, 'bbox': [a,b,c,d], 'category_id': xx}
        self.large_scale_json["images"] = []  # images: {'id': xx, 'full_name': xx}

    def get_image_predict_bbox(self, image):  # image ~ (3, H, W)
        img_meta = dict(img=image.cpu().numpy().transpose((1, 2, 0)), img_shape=[self.img_h, self.img_w],
                        ori_shape=[self.img_h, self.img_w])
        data = self.screen_pipeline(img_meta)
        data['inputs'] = [data['inputs']]
        data['data_samples'] = [data['data_samples']]
        output = self.soft_label_model.test_step(data=data)  # [labels, bbox, scores, probability distribution]
        return output

    def screen_iter(self, overlap_iou=0.5):
        bbox_predict = self.get_image_predict_bbox(self.large_scale_image_2) # why not self.large_scale_image? the one with masked objects? for worst case scenario?
        scores = bbox_predict[0].pred_instances.scores
        bbox_predict = bbox_predict[0].pred_instances.bboxes
        if "YOLO" == self.model_type:
            new_bbox_predict = []
            for i, score in enumerate(scores):
                if score >= 0.25:
                    new_bbox_predict.append(bbox_predict[i])
            if len(new_bbox_predict) <= 1:
                bbox_predict = bbox_predict
            else:
                bbox_predict = torch.stack(new_bbox_predict, 0)

        bbox_predict[:, 2] = bbox_predict[:, 2] - bbox_predict[:, 0]
        bbox_predict[:, 3] = bbox_predict[:, 3] - bbox_predict[:, 1]
        bbox_gt = torch.Tensor(
            [[candidate[2], candidate[0], candidate[3] - candidate[2], candidate[1] - candidate[0]] for candidate in
             self.candidates]).to(bbox_predict.device).float()

        if bbox_predict.shape[0] <= 1 or bbox_gt.shape[0] <= 1:
            divide_idxs = set([i for i in range(bbox_gt.shape[0])])
        else:
            idxs, _ = find_pairs_above_threshold_torch(bbox_gt, bbox_predict, threshold=overlap_iou) # we are not considering the classes of the objects here right?
            divide_idxs = set([i for i in range(bbox_gt.shape[0])]) - idxs

        new_divide_idxs = []

        _memory = []
        for i in range(len(self.memory)):
            x, y = self.memory[i]
            if self.large_scale_mask[0, int(x), int(y)].int().item() == 0:
                _memory.append((x, y))
        self.memory = _memory

        for d_idx in divide_idxs: # divide_idxs is the set of indices of the objects to remove
            _y_min, _x_min, _w, _h = bbox_gt[d_idx]  # x,y,h,w
            _x_max = _x_min + _h
            _y_max = _y_min + _w
            _x_min = int(torch.clip(_x_min, 0, self.img_h).item())
            _y_min = int(torch.clip(_y_min, 0, self.img_w).item())
            _x_max = int(torch.clip(_x_max, 0, self.img_h).item())
            _y_max = int(torch.clip(_y_max, 0, self.img_w).item())
            self.large_scale_mask[0, _x_min:_x_max, _y_min:_y_max] = False
            new_divide_idxs.append(d_idx)
            if round(self.large_scale_mask.float().sum().item() / self.large_scale_mask.numel(),
                     3) <= self.discard_thres: # WHY? Intuituition?
                self.large_scale_mask[0, _x_min:_x_max, _y_min:_y_max] = True
                break
            self.memory.append(((_x_min + _x_max) / 2, (_y_min + _y_max) / 2))
            self.large_scale_image_2[:, _x_min:_x_max, _y_min:_y_max] = self._bg[:, _x_min:_x_max,
                                                                        _y_min:_y_max].clone()
            self.large_scale_image[:, _x_min:_x_max, _y_min:_y_max] = self._bg[:, _x_min:_x_max, _y_min:_y_max].clone()

        self.candidates = [candidate for i, candidate in enumerate(self.candidates) if i not in new_divide_idxs]

    @torch.no_grad()
    @master_only
    def add_datasample(
            self,
            image: torch.Tensor,
            data_sample: DetDataSample = None,
            draw_gt: bool = True) -> None:
        
        image = torch.clamp(image, 0, 255).int()
        self.width, self.height = image.shape[2], image.shape[1]
        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            assert 'gt_instances' in data_sample
            gt_crops, gt_labels, gt_masks = self._get_crop(image, data_sample.gt_instances, self.scale,
                                                           classes=self.dataset_meta.get('classes', None))  # [h,w] X N
        else:
            raise NotImplementedError

        if self.apply_select_iter:
            self._insert_in_image(gt_crops, gt_labels, gt_masks) # puts objects in self.candidates if overlap_iou thresh is satisfied.
        else:
            self._insert_in_image_wout_overlap(gt_crops, gt_labels, gt_masks) # puts objects in self.candidates without checking overlap_iou thresh

        # screening process of the candidates
        if self.count % int(self.interval / 10) == 0 and self.count % self.interval != 0 and self.apply_screen_iter: 
            print("\n Passing screening...", self.count)
            print("Content Ratio [0]:",
                  round(self.large_scale_mask.float().sum().item() / self.large_scale_mask.numel(), 3))
            self.screen_iter(self.screen_overlap_iou)
            print("Content Ratio [1]:",
                  round(self.large_scale_mask.float().sum().item() / self.large_scale_mask.numel(), 3))
        # saving the large scale image
        if self.count % self.interval == 0 or self.count == self.img_total_number: # at the end of interval or at the end of all images
            print("\n Passing saving...", self.count)
            print(f"\n Save the {int(self.count // self.interval)}th image")
            full_name = self.save_large_scale_image()
            if self.save_gt:
                self.large_scale_json["images"].append({"full_name": full_name, "id": self.image_id})

                for candidate in self.candidates:
                    self.large_scale_json["annotations"].append({"image_id": self.image_id,
                                                                 "bbox": [candidate[2], candidate[0],
                                                                          candidate[3] - candidate[2],
                                                                          candidate[1] - candidate[0]],
                                                                 "category_id": candidate[4]})

                ff = open(self.labels_file, "w")
                ff.write(json.dumps(self.large_scale_json))
                ff.close()
            self.init_image()
        self.count += 1

    def _insert_in_image(self, gt_crops, gt_labels, gt_masks) -> np.ndarray:
        l_c, l_h, l_w = self.large_scale_image.shape
        candidates = copy.deepcopy(self.candidates)
        if len(gt_masks) == 0:
            mask_tag = False
            gt_masks = [None for i in range(len(gt_crops))]
        else:
            mask_tag = True

        iii = 0
        for gt_crop, gt_label, gt_mask in zip(gt_crops, gt_labels, gt_masks):
            c, h, w = gt_crop.shape
            c_h, c_w = h / 2, w / 2
            interval_h = [c_h / 2, l_h - c_h / 2]
            interval_w = [c_w / 2, l_w - c_w / 2]

            if interval_h[1] < interval_h[0]:
                interval_h = [interval_h[1], interval_h[0]]

            if interval_w[1] < interval_w[0]:
                interval_w = [interval_w[1], interval_w[0]]

            insert_tag = False
            for ii in range(40):
                if len(self.memory) > 0:
                    _idx = np.random.choice(np.arange(len(self.memory)), 1, replace=False)[0]
                    center_h, center_w = self.memory[_idx]
                    _p = torch.rand(1).float().item()
                    center_h = ((torch.rand(1) * (interval_h[1] - interval_h[0]) + interval_h[0]) * _p + (
                            1 - _p) * center_h).int()
                    center_w = ((torch.rand(1) * (interval_w[1] - interval_w[0]) + interval_w[0]) * _p + (
                            1 - _p) * center_w).int()
                else:
                    center_h = (torch.rand(1) * (interval_h[1] - interval_h[0]) + interval_h[0]).int()
                    center_w = (torch.rand(1) * (interval_w[1] - interval_w[0]) + interval_w[0]).int()
                left = center_h - c_h if center_h - c_h >= 0 else torch.Tensor([0])
                right = center_h + c_h if center_h + c_h <= l_h else torch.Tensor([l_h])
                upper = center_w - c_w if center_w - c_w >= 0 else torch.Tensor([0])
                lower = center_w + c_w if center_w + c_w <= l_w else torch.Tensor([l_w])
                left, right, upper, lower = int(left.item()), int(right.item()), int(upper.item()), int(lower.item())
                if lower <= upper:
                    lower = upper + 1
                if right <= left:
                    right = left + 1
                post_gt_mask = gt_mask[int(max(0, c_h - center_h)): int(h - max(center_h + c_h - l_h, 0)),
                               int(max(0, c_w - center_w)): int(w - max(center_w + c_w - l_w, 0))]
                post_gt_mask = torch.nn.functional.interpolate(post_gt_mask[None, None, ...].float(),
                                                               size=(right - left, lower - upper), mode='bilinear')[
                    0, 0, ...].bool().to(self.large_scale_image.device)
                overlap_iou = self.large_scale_mask[0, left:right, upper:lower][post_gt_mask].float().sum().item() / (
                        1 + post_gt_mask.float().sum().item())
                if overlap_iou <= self.overlap_iou:
                    insert_tag = True
                    self.large_scale_mask[0, left:right, upper:lower][post_gt_mask] = True
                    break

            if insert_tag:
                post_gt_crop = gt_crop[:, int(max(0, c_h - center_h)): int(h - max(center_h + c_h - l_h, 0)),
                               int(max(0, c_w - center_w)): int(w - max(center_w + c_w - l_w, 0))]
                if mask_tag:
                    post_gt_mask = gt_mask[int(max(0, c_h - center_h)): int(h - max(center_h + c_h - l_h, 0)),
                                   int(max(0, c_w - center_w)): int(w - max(center_w + c_w - l_w, 0))]
                    post_gt_mask = torch.nn.functional.interpolate(post_gt_mask[None, None, ...].float(),
                                                                   size=(right - left, lower - upper), mode='bilinear')[
                        0, 0, ...].bool().to(self.large_scale_image.device)
                    post_gt_crop = \
                        torch.nn.functional.interpolate(post_gt_crop[None, ...].float(),
                                                        size=(right - left, lower - upper),
                                                        mode='bilinear')[0].int().to(self.large_scale_image.device)
                    self.large_scale_image[:, left:right, upper:lower] = torch.where(post_gt_mask, post_gt_crop,
                                                                                     self.large_scale_image[:,
                                                                                     left:right, upper:lower].clone()) # mask
                    self.large_scale_image_2[:, left:right, upper:lower] = post_gt_crop # bbox
                else:
                    self.large_scale_image[:, left:right, upper:lower] = \
                        torch.nn.functional.interpolate(post_gt_crop[None, ...].float(),
                                                        size=(right - left, lower - upper), mode='bilinear')[0].int()
                candidates.append([left, right, upper, lower, gt_label, 0])
            iii += 1

        self.candidates = candidates

    def _insert_in_image_wout_overlap(self, gt_crops, gt_labels, gt_masks) -> np.ndarray:
        l_c, l_h, l_w = self.large_scale_image.shape
        candidates = copy.deepcopy(self.candidates)
        if len(gt_masks) == 0:
            mask_tag = False
            gt_masks = [None for i in range(len(gt_crops))]
        else:
            mask_tag = True

        iii = 0
        for gt_crop, gt_label, gt_mask in zip(gt_crops, gt_labels, gt_masks):
            c, h, w = gt_crop.shape
            c_h, c_w = h / 2, w / 2
            interval_h = [c_h / 2, l_h - c_h / 2]
            interval_w = [c_w / 2, l_w - c_w / 2]

            if interval_h[1] < interval_h[0]:
                interval_h = [interval_h[1], interval_h[0]]

            if interval_w[1] < interval_w[0]:
                interval_w = [interval_w[1], interval_w[0]]

            if len(self.memory) > 0:
                _idx = np.random.choice(np.arange(len(self.memory)), 1, replace=False)[0]
                center_h, center_w = self.memory[_idx]
                _p = torch.rand(1).float().item()
                center_h = ((torch.rand(1) * (interval_h[1] - interval_h[0]) + interval_h[0]) * _p + (
                        1 - _p) * center_h).int()
                center_w = ((torch.rand(1) * (interval_w[1] - interval_w[0]) + interval_w[0]) * _p + (
                        1 - _p) * center_w).int()
            else:
                center_h = (torch.rand(1) * (interval_h[1] - interval_h[0]) + interval_h[0]).int()
                center_w = (torch.rand(1) * (interval_w[1] - interval_w[0]) + interval_w[0]).int()
            left = center_h - c_h if center_h - c_h >= 0 else torch.Tensor([0])
            right = center_h + c_h if center_h + c_h <= l_h else torch.Tensor([l_h])
            upper = center_w - c_w if center_w - c_w >= 0 else torch.Tensor([0])
            lower = center_w + c_w if center_w + c_w <= l_w else torch.Tensor([l_w])
            left, right, upper, lower = int(left.item()), int(right.item()), int(upper.item()), int(lower.item())
            if lower <= upper:
                lower = upper + 1
            if right <= left:
                right = left + 1
            post_gt_mask = gt_mask[int(max(0, c_h - center_h)): int(h - max(center_h + c_h - l_h, 0)),
                            int(max(0, c_w - center_w)): int(w - max(center_w + c_w - l_w, 0))]
            post_gt_mask = torch.nn.functional.interpolate(post_gt_mask[None, None, ...].float(),
                                                            size=(right - left, lower - upper), mode='bilinear')[
                0, 0, ...].bool().to(self.large_scale_image.device)
            
            post_gt_crop = gt_crop[:, int(max(0, c_h - center_h)): int(h - max(center_h + c_h - l_h, 0)),
                               int(max(0, c_w - center_w)): int(w - max(center_w + c_w - l_w, 0))]
            if mask_tag:
                post_gt_mask = gt_mask[int(max(0, c_h - center_h)): int(h - max(center_h + c_h - l_h, 0)),
                                int(max(0, c_w - center_w)): int(w - max(center_w + c_w - l_w, 0))]
                post_gt_mask = torch.nn.functional.interpolate(post_gt_mask[None, None, ...].float(),
                                                                size=(right - left, lower - upper), mode='bilinear')[
                    0, 0, ...].bool().to(self.large_scale_image.device)
                post_gt_crop = \
                    torch.nn.functional.interpolate(post_gt_crop[None, ...].float(),
                                                    size=(right - left, lower - upper),
                                                    mode='bilinear')[0].int().to(self.large_scale_image.device)
                self.large_scale_image[:, left:right, upper:lower] = torch.where(post_gt_mask, post_gt_crop,
                                                                                    self.large_scale_image[:,
                                                                                    left:right, upper:lower].clone()) # mask
                self.large_scale_image_2[:, left:right, upper:lower] = post_gt_crop # bbox
            else:
                self.large_scale_image[:, left:right, upper:lower] = \
                    torch.nn.functional.interpolate(post_gt_crop[None, ...].float(),
                                                        size=(right - left, lower - upper), mode='bilinear')[0].int()
            candidates.append([left, right, upper, lower, gt_label, 0])
        self.candidates = candidates

    @master_only
    def _get_bboxes(
            self,
            bboxes: Union[np.ndarray, torch.Tensor],
    ):
        check_type('bboxes', bboxes, (np.ndarray, torch.Tensor))
        if len(bboxes.shape) == 1:
            bboxes = bboxes[None]
        assert bboxes.shape[-1] == 4, (
            f'The shape of `bboxes` should be (N, 4), but got {bboxes.shape}')
        assert (bboxes[:, 0] <= bboxes[:, 2]).all() and (bboxes[:, 1] <=
                                                         bboxes[:, 3]).all()
        if not self._is_posion_valid(bboxes.reshape((-1, 2, 2))):
            warnings.warn(
                'Warning: The bbox is out of bounds,'
                ' the drawn bbox may not be in the image', UserWarning)
        poly = torch.stack(
            (bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 1],
             bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 3]),
            dim=-1).contiguous().view(-1, 4, 2)
        poly = [p for p in poly]  # (4,2)
        return poly

    @master_only
    def _get_crop(
            self,
            image,
            instances: ['InstanceData'],
            scale: int = 1.0,
            classes: list = None
    ):
        if 'bboxes' in instances and instances.bboxes.sum() > 0:
            bboxes = instances.bboxes
            labels = instances.labels
            polys = self._get_bboxes(bboxes) # bbox with 4 vertices
            # crop
            crops = []
            crops_labels = []
            crops_masks = []

            if 'masks' in instances:
                masks = instances.masks
                if isinstance(masks, torch.Tensor):
                    masks = masks.numpy()
                elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                    masks = masks.to_ndarray()
                masks = torch.from_numpy(masks).bool().to(image.device)  # [number, h, w]
            
            # min and max areas in MS COCO based on stats
            min_area = 0
            max_area = 787151
            area_range = max_area - min_area

            for idx, poly in enumerate(polys):
                poly = poly.int()
                upper, lower = poly[0, 1], poly[2, 1] # y_min, y_max
                left, right = poly[0, 0], poly[2, 0] # x_min, x_max
                if lower == upper:
                    lower = upper + 1 if upper + 1 <= self.height else upper
                    if lower == upper:
                        upper = lower - 1
                if left == right:
                    right = left + 1 if left + 1 <= self.width else left
                    if right == left:
                        left = right - 1
                
                if self.extend_context:
                    if not self.dynamic_extend_context: # static extension
                        lower, upper, left, right = extend_context(lower, upper, left, right, self.height, self.width, self.extend_context_value)
                    else:
                        # determine the area of the current object
                        area = (lower - upper) * (right - left)
                        assert area > 0, "Area is 0 or less"
                        
                        # calculate the dynamic extend context value as a percentage of 10 pixels
                        if area_range > 0:
                            percentage = 1 - ((area - min_area) / area_range)
                            # make sure percentage is between 0 and 1
                            percentage = max(0, min(percentage, 1))
                            # assert if percentage is less than 0 or greater than 1
                            assert 0 <= percentage <= 1, "Percentage is less than 0 or greater than 1"
                            extend_value = 10 * percentage
                            # convert to int
                            extend_value = int(extend_value)
                        else:
                            extend_value = 10  # fallback if all areas are the same

                        lower, upper, left, right = extend_context(lower, upper, left, right, self.height, self.width, extend_value)

                crop = image[:, upper:lower, left:right]
                crops.append(crop)
                if 'masks' in instances:
                    crops_masks.append(masks[idx][upper:lower, left:right])

            for i, label in enumerate(labels):
                if 'label_names' in instances:
                    label_text = instances.label_names[i]
                else:
                    label_text = classes[
                        label] if classes is not None else f'class {label}'
                crop_label = class_name2id[label_text]
                crops_labels.append(crop_label)

            if scale != 1.0:
                for i in range(len(crops)):
                    c, h, w = crops[i].shape
                    new_h = int(scale * h)
                    new_w = int(scale * w)
                    if new_h == 0:
                        new_h = 1
                    if new_w == 0:
                        new_w = 1
                    crops[i] = \
                        torch.nn.functional.interpolate(crops[i][None, ...].float(), size=(new_h, new_w),
                                                        mode='bilinear')[
                            0].int()
                    if 'masks' in instances:
                        crops_masks[i] = \
                            torch.nn.functional.interpolate(crops_masks[i][None, None, ...].float(),
                                                            size=(new_h, new_w),
                                                            mode='bilinear')[
                                0, 0, ...].bool()

            if self.max_img_w is not None and self.max_img_h is not None:
                # max_img_h, max_img_w are the maximum size of the cropped image (object)
                # if the size of the cropped image is larger than max_img_h or max_img_w, resize it
                for i in range(len(crops)):
                    c, h, w = crops[i].shape
                    new_h = min(self.max_img_h, h)
                    new_w = min(self.max_img_w, w)
                    local_scale = min(new_w / w, new_h / h)
                    if local_scale != 1.0:
                        crops[i] = torch.nn.functional.interpolate(crops[i][None, ...].float(),
                                                                   size=(max(int(h * local_scale), 1),
                                                                         max(1, int(w * local_scale))))[
                            0].int()
                        if 'masks' in instances:
                            crops_masks[i] = torch.nn.functional.interpolate(crops_masks[i][None, None, ...].float(),
                                                                             size=(max(int(h * local_scale), 1),
                                                                                   max(1, int(w * local_scale))))[
                                0, 0, ...].bool()

        else:
            crops = []
            crops_labels = []
            crops_masks = []
        return crops, crops_labels, crops_masks

    def save_large_scale_image(self, ):
        large_scale_image = tensor2ndarray(self.large_scale_image).transpose((1, 2, 0))[..., ::-1]
        large_scale_image_2 = tensor2ndarray(self.large_scale_image_2).transpose((1, 2, 0))[..., ::-1]
        large_scale_mask = tensor2ndarray(self.large_scale_mask.expand(3, -1, -1).float() * 255).transpose(
            (1, 2, 0))[..., ::-1]

        if self.count == self.img_total_number:
            idx = int(self.count // self.interval)
        else:
            idx = int(self.count // self.interval) - 1
        if self.save_file is not None:
            if not os.path.exists(self.save_file):
                os.makedirs(self.save_file, exist_ok=True)
            
            if(self.label_type == 'mask'):
                mmcv.imwrite(large_scale_image, os.path.join(self.save_file, "ipc_{:05d}.jpg".format(idx)))
            elif(self.label_type == 'bbox'):
                mmcv.imwrite(large_scale_image_2,
                         os.path.join(self.save_file, "baseline_{:05d}.jpg".format(idx)))

        return "baseline_{:05d}.jpg".format(idx)
