#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=casr101 --kill-on-bad-exit=1 \
python -u tools/train.py \
    configs/myconfig/v2.0/cascade_rcnn_r101_fpn_1x.py \
    --work_dir=work_dirs/cascade_rcnn_r101_fpn_1x_trainval \
    --launcher="slurm" \
    --validate \
