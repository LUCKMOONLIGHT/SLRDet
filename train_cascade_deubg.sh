#!/bin/bash
set -x
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=cascade_rrpn --kill-on-bad-exit=1 \
python -u tools/train.py configs/myconfig/cascade_rcnn_hrnetv2p_w32_1x.py --work_dir=work_dirs/cascade_rcnn_hrnetv2p_w32_1x_train --launcher="slurm" --validate
