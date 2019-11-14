#!/bin/bash                                                         
set -x                                                              
srun -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=inference --kill-on-bad-exit=1 \
python -u demo/inference_cascade.py \
     --config configs/cascade_rcnn_x101_32x4d_fpn_2x.py \
     --checkpoint checkpoints/cascade_epoch_85.pth \
     --out work_dirs/retinanet_hrnet_fpn_inference_cascade \
     --cropsize 800 \
     --stride 256 \
     --testImgpath data/test_rpn/test \
     --saveTxtpath work_dirs/retinanet_hrnet_fpn_inference_cascade/txt \
     --saveImgpath work_dirs/retinanet_hrnet_fpn_inference_cascade/img \
     --patchImgPath work_dirs/retinanet_hrnet_fpn_inference_cascade/patch \
