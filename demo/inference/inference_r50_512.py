import argparse
import os.path as osp
import mmcv
import os
import cv2
import math
import copy
import torch
import numpy as np
from PIL import Image, ImageDraw
from mmdet.apis import init_detector, inference_detector
from mmdet.core import multiclass_nms
from draw_box_in_img import draw_boxes_with_label_and_scores

ODAI_LABEL_MAP = {
        'back-ground': 0,
        'plane': 1,
        'baseball-diamond': 2,
        'bridge': 3,
        'ground-track-field': 4,
        'small-vehicle': 5,
        'large-vehicle': 6,
        'ship': 7,
        'tennis-court': 8,
        'basketball-court': 9,
        'storage-tank': 10,
        'soccer-ball-field': 11,
        'roundabout': 12,
        'harbor': 13,
        'swimming-pool': 14,
        'helicopter': 15,
    }

def get_label_name_map():
    reverse_dict = {}
    for name, label in ODAI_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict



def osp(savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

if __name__ == '__main__':
    from mmdet.apis import init_detector, inference_detector
    import mmcv
    from PIL import Image

    config_file = '/mnt/lustre/yanhongchang/project/mmdetection/configs/myconfig/v2.0/test/test_faster_rcnn_r50_fpn_1x_512.py'
    checkpoint_file = '/mnt/lustre/yanhongchang/project/mmdetection/checkpoints/rotated/rfaster_rcnn_r50_fpn_1x_trainval_epoch_50_epoch_50_685.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    # img = '/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/data/test_sp/images/P0063__1__0___0.png'  # or img = mmcv.imread(img), which will only load it once
    imgpath = '/mnt/lustre/yanhongchang/project/mmdetection/data/test/crop512_256'
    # imgpath = '/mnt/lustre/yanhongchang/project/mmdetection/data/test/crop1024_512'
    saveimgpath = '/mnt/lustre/yanhongchang/project/mmdetection/demo/work_dirs/out_img/faster_r50/tr_crop512/inference'
    savetxtpath = '/mnt/lustre/yanhongchang/project/mmdetection/demo/work_dirs/out_img/faster_r50/tr_crop512/txt'
    imglist = os.listdir(imgpath)
    osp(saveimgpath)
    osp(savetxtpath)
    for imgname in imglist:
        singleimgpath = os.path.join(imgpath, imgname)
        result = inference_detector(model, singleimgpath)
        bboxes = np.vstack(result)
        labels = [  # 0-15
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)
        if bboxes.shape[0] > 0:

            # outimg = Image.open(singleimgpath).convert('RGB')
            # outimg = np.array(outimg)
            # image = draw_boxes_with_label_and_scores(outimg, bboxes[:, :5], bboxes[:, 5], labels, 1)
            # image.save(os.path.join(saveimgpath, imgname))

            CLASS_DOTA = ODAI_LABEL_MAP.keys()
            LABEl_NAME_MAP = get_label_name_map()
            write_handle_r = {}

            for sub_class in CLASS_DOTA:
                if sub_class == 'back-ground':
                    continue
                write_handle_r[sub_class] = open((os.path.join(savetxtpath, 'Task1_%s.txt' % sub_class)), 'a+')

            boxes = []

            for rect in bboxes[:, :5]:
                box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), -rect[4]))
                box = np.reshape(box, [-1, ])
                boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

            rboxes = np.array(boxes, dtype=np.float32)

            for i, rbox in enumerate(rboxes):
                command = '%s %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n' % (
                imgname[:-4], bboxes[i, 5], rbox[0],
                rbox[1], rbox[2], rbox[3], rbox[4], rbox[5], rbox[6], rbox[7])

                write_handle_r[LABEl_NAME_MAP[int(labels[i]) + 1]].write(command)

            for sub_class in CLASS_DOTA:
                if sub_class == 'back-ground':
                    continue
                write_handle_r[sub_class].close()


