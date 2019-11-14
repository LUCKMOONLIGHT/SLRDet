#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=v2casx101 --kill-on-bad-exit=1 \
python -u tools/train.py \
    configs/myconfig/v2.0/cascade_rcnn_x101_64x4d_fpn_1x_v2.py \
    --work_dir=work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_trainval_v2 \
    --launcher="slurm" \
    --validate \
