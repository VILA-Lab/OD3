# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmcv.transforms import Compose

from mmengine.model.utils import revert_sync_batchnorm
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from mmengine.runner import load_checkpoint
from mmengine.device.utils import get_device

from mmdet.models.utils import mask2ndarray
from mmdet.registry import DATASETS, VISUALIZERS, MODELS
from mmdet.structures.bbox import BaseBoxes
from mmdet.apis import inference_detector, init_detector
from distiller import CropObject


def parse_args():
    parser = argparse.ArgumentParser(description='Distill a dataset')
    parser.add_argument('config', help='config file path', default = "configs/dd/data_synthesis/data_synthesis_faster-rcnn_r50_fpn_1x_coco.py")
    parser.add_argument('checkpoint', help='checkpoint file', default="checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth")
    parser.add_argument('labelsfile', help='path to labels files to save', default = "large_scale_bbox.json")
    parser.add_argument('--output-dir', default=None,type=str)
    parser.add_argument('--original-dir', default=None,type=str)
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument('--apply-screen-iter',  default=False, action='store_true')
    parser.add_argument('--apply-select-iter',  default=False, action='store_true')
    parser.add_argument('--extend-context',  default=False, action='store_true')
    parser.add_argument('--apply-background',  default=False, action='store_true')
    parser.add_argument('--background-path', default="./mmdetection/background.txt", type=str, help='path to text file of backgrounds')
    parser.add_argument('--extend-context-dynamic',  default=False, action='store_true')
    parser.add_argument('--extend-context-value', default=10, type=int)
    parser.add_argument('--overlap-threshold', default=0.6, type=float)
    parser.add_argument('--conf-threshold', default=0.2, type=float)
    parser.add_argument('--ipd', default=1184, type=int)
    parser.add_argument('--img-width', default=484, type=int, help='width of canvas')
    parser.add_argument('--img-height', default=578, type=int, help='height of canvas')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set original dir in config
    if args.original_dir is not None:
        cfg.data_root = args.original_dir
        
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    device = get_device()
    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    # soft label model
    soft_label_model = MODELS.build(cfg.model)
    model_type = "RCNN"
    revert_sync_batchnorm(soft_label_model)
    soft_label_model = soft_label_model.eval()
    soft_label_model = soft_label_model.to(device)
    load_checkpoint(soft_label_model, args.checkpoint, map_location=device)

    # screening pipline
    screen_pipeline = Compose(cfg.screen_pipeline)
    nbr_imgs = len(dataset)

    visualizer = CropObject(ipd=args.ipd, img_total_number=nbr_imgs, img_h=args.img_height, img_w=args.img_width, save_file=args.output_dir,
                            scale=1.0, max_img_h=121, max_img_w=144, save_gt=True, overlap_threshold=args.overlap_threshold,
                            apply_select_background=args.apply_background,background_bbox_path=args.background_path,
                            original_coco_path=osp.join(cfg.data_root, "train2017"), apply_screen_iter=args.apply_screen_iter, apply_select_iter=args.apply_select_iter, 
                            soft_label_model=soft_label_model, conf_threshold=args.conf_threshold, dynamic_extend_context=args.extend_context_dynamic,
                            screen_pipeline=screen_pipeline, model_type=model_type, labels_file = args.labelsfile, extend_context=args.extend_context, extend_context_value=args.extend_context_value)

    visualizer.dataset_meta = dataset.metainfo
    progress_bar = ProgressBar(len(dataset))

    for item in dataset:

        img = item['inputs']
        data_sample = item['data_samples'].numpy() # object containing all labels, img path, metadata, img shape...
        gt_instances = data_sample.gt_instances
        img_path = osp.basename(item['data_samples'].img_path) 

        img = img[[2, 1, 0], ...]  # bgr to rgb
        gt_bboxes = gt_instances.get('bboxes', None)
        if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
            gt_instances.bboxes = gt_bboxes.tensor
        gt_masks = gt_instances.get('masks', None)
        if gt_masks is not None:
            masks = mask2ndarray(gt_masks)
            gt_instances.masks = masks.astype(bool)
        data_sample.gt_instances = gt_instances

        visualizer.add_datasample(
            img,
            data_sample)
        progress_bar.update()


if __name__ == '__main__':
    main()
