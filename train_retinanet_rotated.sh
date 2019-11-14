#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=retinanet --kill-on-bad-exit=1 \
python -u tools/train.py configs/myconfig/retinanet_x101_64x4d_fpn_1x.py \
    --work_dir=work_dirs/retinanet_x101_64x4d_fpn_1x_trainval \
    --launcher="slurm" \
    --validate \
