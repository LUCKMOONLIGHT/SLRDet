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


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--cropsize', help='patch image size', default=512, type=int)
    parser.add_argument('--stride', help='patch image stride', default=256, type=int)
    parser.add_argument('--testImgpath', help='test image path', default='work_dirs/retinanet_hrnet_fpn_inference', type=str)
    parser.add_argument('--saveTxtpath', help='test image path', default='work_dirs/retinanet_hrnet_fpn_inference', type=str)
    parser.add_argument('--saveImgpath', help='test image path', default='work_dirs/retinanet_hrnet_fpn_inference', type=str)
    parser.add_argument('--patchImgPath', help='test image path', default='work_dirs/retinanet_hrnet_fpn_inference', type=str)
    args = parser.parse_args()
    return args

def single_gpu_test(args, cfg, model):
    testImgList = os.listdir(args.testImgpath)
    for imgfile in testImgList:
        imgfile = imgfile.strip()
        img = Image.open(os.path.join(args.testImgpath, imgfile))
        image = img.convert('RGB')
        img = np.array(image)
        width, height, channel = img.shape
        rows = int(math.ceil(1.0 * (width - args.cropsize) / args.stride)) + 1
        cols = int(math.ceil(1.0 * (height - args.cropsize) / args.stride)) + 1
        multi_bboxes = list()
        multi_scores = list()
        for row in range(rows):
            if width > args.cropsize:
                y_start = min(row * args.stride, width - args.cropsize)
                y_end = y_start + args.cropsize
            else:
                y_start = 0
                y_end = width
            for col in range(cols):
                if height > args.cropsize:
                    x_start = min(col * args.stride, height - args.cropsize)
                    x_end = x_start + args.cropsize
                else:
                    x_start = 0
                    x_end = height
                subimg = copy.deepcopy(img[y_start:y_end, x_start:x_end, :])
                w, h, c = np.shape(subimg)
                outimg = np.zeros((args.cropsize, args.cropsize, 3))
                outimg[0:w, 0:h, :] = subimg
                result = inference_detector(model, outimg) #15
                bboxes = np.vstack(result)
                labels = [ #0-15
                    np.full(bbox.shape[0], i+1, dtype=np.int32)
                    for i, bbox in enumerate(result)
                ]
                labels = np.concatenate(labels)
                if len(bboxes) > 0:
                    # image = draw_boxes_with_label_and_scores(outimg, bboxes[:, :4], bboxes[:, 4], labels - 1, 0)
                    # image.save(os.path.join(args.patchImgPath, imgfile[:-4]+'_'+str(y_start)+'_'+str(x_start)+'.png'))
                    bboxes[:, :2] += [x_start, y_start]
                    multi_bboxes.append(bboxes[:, :5])
                    scores = np.zeros((bboxes.shape[0], len(ODAI_LABEL_MAP.keys()))) #0-15
                    for i, j in zip(range(bboxes.shape[0]), labels):
                        scores[i, j] = bboxes[i, 5]
                    multi_scores.append(scores)
        crop_num = len(multi_bboxes)
        if crop_num > 0:
            multi_bboxes = np.vstack(multi_bboxes)
            multi_scores = np.vstack(multi_scores)
            multi_bboxes = torch.Tensor(multi_bboxes)
            multi_scores = torch.Tensor(multi_scores)
            score_thr = 0.1
            nms=dict(type='nms', iou_thr=0.5)
            max_per_img = 2000
            det_bboxes, det_labels = multiclass_nms(multi_bboxes, multi_scores,
                                                            score_thr, nms,
                                                            max_per_img)
            if det_bboxes.shape[0] > 0:
                det_bboxes = np.array(det_bboxes)
                det_labels = np.array(det_labels) #0-14
                image = draw_boxes_with_label_and_scores(img, det_bboxes[:, :5], det_bboxes[:, 5], det_labels, 1)
                image.save(os.path.join(args.saveImgpath, imgfile))

                CLASS_DOTA = ODAI_LABEL_MAP.keys()
                LABEl_NAME_MAP = get_label_name_map()
                write_handle_r = {}
                osp(args.saveTxtpath)

                for sub_class in CLASS_DOTA:
                    if sub_class == 'back-ground':
                        continue
                    write_handle_r[sub_class] = open(os.path.join(args.saveTxtpath, 'Task1_%s.txt' % sub_class), 'a+')


                """
                :det_bboxes: format [x_c, y_c, w, h, theta, score]
                :det_labels: [label]
                """
                boxes = []

                for rect in det_bboxes[:, :5]:
                    box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), -rect[4]))
                    box = np.reshape(box, [-1, ])
                    boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

                rboxes = np.array(boxes, dtype=np.float32)

                for i, rbox in enumerate(rboxes):
                    command = '%s %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n' % (imgfile[:-4], det_bboxes[i, 5], rbox[0],
                                                                 rbox[1], rbox[2], rbox[3], rbox[4], rbox[5], rbox[6], rbox[7])

                    write_handle_r[LABEl_NAME_MAP[int(det_labels[i]) + 1]].write(command)

                for sub_class in CLASS_DOTA:
                    if sub_class == 'back-ground':
                        continue
                    write_handle_r[sub_class].close()



if __name__ == '__main__':
    args = parse_args()
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    if isinstance(args.config, str):
        config = mmcv.Config.fromfile(args.config)
    osp(args.saveTxtpath)
    osp(args.saveImgpath)
    osp(args.patchImgPath)

    single_gpu_test(args, config, model)
