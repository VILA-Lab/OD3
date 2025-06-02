_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

backend_args = None
dataset_type = 'CocoDataset'
data_root = '/data/ms-coco/'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True,with_label=True),
    dict(type='Resize', scale=(578, 484), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

screen_pipeline = [
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('ori_shape', 'img_shape', 'scale_factor'))
]