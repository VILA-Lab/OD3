#!/bin/bash
TRAINING_CONFIG=$1
NUM_GPUS=$2
ANNOTATION=$3
IMAGES_DIR=$4
WORK_DIR=$5

PORT=29502 bash ./mmrazor/tools/dist_train.sh \
                $TRAINING_CONFIG $NUM_GPUS \
                --cfg-options train_dataloader.dataset.ann_file=$ANNOTATION \
                train_dataloader.dataset.data_prefix.img=$IMAGES_DIR \
                --work-dir $WORK_DIR