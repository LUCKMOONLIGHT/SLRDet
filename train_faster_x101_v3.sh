#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=fasx101 --kill-on-bad-exit=1 \
python -u tools/train.py \
    configs/myconfig/v2.0/faster_rcnn_x101_64x4d_fpn_1x_v3.py \
    --work_dir=work_dirs/faster_rcnn_x101_64x4d_fpn_1x_v3_trainval \
    --launcher="slurm" \
    --validate \
