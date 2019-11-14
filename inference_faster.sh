#!/bin/bash                                                         
set -x                                                              
srun -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 \
    --job-name=inference --kill-on-bad-exit=1 \
python -u demo/inference_faster.py \
     --config configs/myconfig/faster_rcnn_hrnetv2p_w32_1x.py \
     --checkpoint checkpoints/faster_rcnn_hrnetv2p_w32_1x_trainval_epoch_60.pth \
     --out work_dirs/faster_rcnn_hrnetv2p_w32_1x_inferece \
     --cropsize 512 \
     --stride 256 \
     --testImgpath data/rrpn15_512/test \
     --saveTxtpath work_dirs/faster_rcnn_hrnetv2p_w32_1x_inferece/txt \
     --saveImgpath work_dirs/faster_rcnn_hrnetv2p_w32_1x_inferece/img \
     --patchImgPath work_dirs/faster_rcnn_hrnetv2p_w32_1x_inferece/patch \
