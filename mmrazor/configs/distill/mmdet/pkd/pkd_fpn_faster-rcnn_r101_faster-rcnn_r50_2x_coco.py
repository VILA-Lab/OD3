_base_ = [
    '../../../_base_/datasets/mmdet/dd_coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='PKDLoss', loss_weight=6),
            loss_pkd_fpn1=dict(type='PKDLoss', loss_weight=6),
            loss_pkd_fpn2=dict(type='PKDLoss', loss_weight=6),
            loss_pkd_fpn3=dict(type='PKDLoss', loss_weight=6)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_pkd_fpn3=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=3)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=96, val_begin=4, val_interval=4)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=96,
        by_epoch=True,
        milestones=[64, 88],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    accumulative_counts=4)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=8))