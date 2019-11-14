#!/bin/bash                                                         
set -x                                                              
srun -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 \
    --job-name=inference --kill-on-bad-exit=1 \
python -u demo/inference_faster.py \
     --config configs/myconfig/libra_faster_rcnn_r50_fpn_1x.py \
     --checkpoint work_dirs/libra_faster_rcnn_r50_fpn_1x_noclip_trainval/epoch_60.pth \
     --out work_dirs/libra_faster_rcnn_r50_fpn_1x_noclip_trainval \
     --cropsize 512 \
     --stride 256 \
     --testImgpath data/test_rrpn/test \
     --saveTxtpath work_dirs/libra_faster_rcnn_r50_fpn_1x_noclip_trainval/test/txt \
     --saveImgpath work_dirs/libra_faster_rcnn_r50_fpn_1x_noclip_trainval/test/img \
     --patchImgPath work_dirs/libra_faster_rcnn_r50_fpn_1x_noclip_trainval/test/patch \
