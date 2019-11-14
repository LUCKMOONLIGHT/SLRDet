#!/bin/bash                                                         
set -x                                                              
srun -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 \
    --job-name=inference --kill-on-bad-exit=1 \
python -u demo/inference_hrnet.py \
     --config configs/myconfig/retinanet_x101_64x4d_fpn_1x.py \
     --checkpoint checkpoints/retinanet_x101_64x4d_fpn_1x_trainval_epoch_60.pth \
     --out work_dirs/retinanet_x101_64x4d_fpn_inferece \
     --cropsize 512 \
     --stride 256 \
     --testImgpath data/test_rrpn/test \
     --saveTxtpath work_dirs/retinanet_x101_64x4d_fpn_inferece/txt \
     --saveImgpath work_dirs/retinanet_x101_64x4d_fpn_inferece/img \
     --patchImgPath work_dirs/retinanet_x101_64x4d_fpn_inferece/patch \
