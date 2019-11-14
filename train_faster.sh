#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 \
    --job-name=test --kill-on-bad-exit=1 \
python -u tools/train.py \
    configs/myconfig/test_faster_rcnn_r50_fpn_1x.py \
    --work_dir=work_dirs/faster_rcnn_r50_fpn_1x_noclip_trainval \
    --launcher="slurm" \
    --validate \
