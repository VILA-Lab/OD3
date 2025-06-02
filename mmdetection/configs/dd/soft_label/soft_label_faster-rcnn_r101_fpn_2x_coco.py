_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py',
    '../../rtmdet/rtmdet_tta.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

backend_args = None
dataset_type = 'CocoDataset'
data_root = './'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

