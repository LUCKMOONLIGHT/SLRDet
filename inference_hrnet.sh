#!/bin/bash                                                         
set -x                                                              
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=inference --kill-on-bad-exit=1 \
python -u demo/inference_hrnet.py \
     --config configs/hrnet/retinanet_hrnet_fpn.py \
     --checkpoint checkpoints/hrnet_epoch_84.pth \
     --out work_dirs/retinanet_hrnet_fpn_inference \
     --cropsize 800 \
     --stride 256 \
     --testImgpath data/test_rpn/test \
     --saveTxtpath work_dirs/retinanet_hrnet_fpn_inference/txt \
     --saveImgpath work_dirs/retinanet_hrnet_fpn_inference/img \
     --patchImgPath work_dirs/retinanet_hrnet_fpn_inference/patch \
