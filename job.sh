#! /bin/bash

python train.py \
    --cfg configs/resnet101_delg_8gpu.yaml \
    OUT_DIR ./output \
    PORT 13001 \
    TRAIN.WEIGHTS ./pretrained/R-101-1x64d_dds_8gpu.pyth
