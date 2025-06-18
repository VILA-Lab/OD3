#!/usr/bin/env bash

OUTPUT_DIR=$1
ORIGINAL_DIR=$2
IPD=$3
MODEL=${4:-fasterrcnn}

if [ "$MODEL" = "retinanet" ]; then
    CONFIG=./mmdetection/configs/dd/data_synthesis/data_synthesis_retinanet_101_coco.py
    CHECKPOINT=./mmdetection/checkpoints/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth
    SOFT_LABEL_CONFIG=./mmdetection/configs/dd/soft_label/soft_label_retina_net_101_fpn_2x_coco.py
elif [ "$MODEL" = "fasterrcnn" ]; then
    CONFIG=./mmdetection/configs/dd/data_synthesis/data_synthesis_faster-rcnn_r50_fpn_1x_coco.py
    CHECKPOINT=./mmdetection/checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth
    SOFT_LABEL_CONFIG=./mmdetection/configs/dd/soft_label/soft_label_faster-rcnn_r101_fpn_2x_coco.py
else
    echo "Error: Unsupported model '$MODEL'. Please use 'retinanet' or 'fasterrcnn'."
    exit 1
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/annotations
mkdir -p ${OUTPUT_DIR}/images

run data synthesis for COCO
python ./mmdetection/tools/dataset_distillation/data_synthesis.py \
        $CONFIG \
        $CHECKPOINT \
        ${OUTPUT_DIR}/annotations/large_scale_anns.json \
        --apply-select-iter --apply-screen-iter --extend-context --extend-context-dynamic \
        --apply-background --not-show --output-dir ${OUTPUT_DIR}/images/ \
        --ipd $IPD --original-dir ${ORIGINAL_DIR}

python ./mmdetection/tools/dataset_converters/images2coco.py \
       ${OUTPUT_DIR}/images/ \
       ./mmdetection/coco_classes.txt \
       ${OUTPUT_DIR}/annotations/hard_label.json

python ./mmdetection/tools/dataset_distillation/preprocess.py \
       ${OUTPUT_DIR}/annotations/hard_label.json \
       ${ORIGINAL_DIR}/annotations/instances_train2017.json \
       ${OUTPUT_DIR}/annotations/hard_label.json 

bash   ./mmdetection/tools/dist_test.sh \
       $SOFT_LABEL_CONFIG \
       $CHECKPOINT 2 \
       --cfg-options test_evaluator.format_only=True \
       data_root=${OUTPUT_DIR} \
       test_evaluator.outfile_prefix=${OUTPUT_DIR}/annotations/soft_label.json \
       val_dataloader.dataset.data_prefix.img='' \
       val_dataloader.dataset.ann_file=${OUTPUT_DIR}/annotations/hard_label.json \
       val_evaluator.ann_file=${OUTPUT_DIR}/annotations/hard_label.json \
       test_evaluator.ann_file=${OUTPUT_DIR}/annotations/hard_label.json \
       test_dataloader.dataset.ann_file=${OUTPUT_DIR}/annotations/hard_label.json \
       test_dataloader.dataset.data_prefix.img='' --tta

python ./mmdetection/tools/dataset_distillation/postprocess.py \
       ${OUTPUT_DIR}/annotations/hard_label.json \
       ${OUTPUT_DIR}/annotations/soft_label.json.bbox.json \
       ${ORIGINAL_DIR}/annotations/instances_train2017.json \
       ${OUTPUT_DIR}/annotations/final_soft_labels.json