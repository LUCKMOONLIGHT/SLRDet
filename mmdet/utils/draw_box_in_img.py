from __future__ import absolute_import, print_function, division

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import cv2

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen', 'LightBlue', 'LightGreen'
]
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

FONT = ImageFont.load_default()

def get_label_name_map():
    reverse_dict = {}
    for name, label in ODAI_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict


LABEL_NAME_MAP = get_label_name_map()

def draw_boxes_with_label_and_scores(img_array, boxes, scores, labels, img_name, path,  mode=0, labelAndScore=0):
    img_array = img_array.astype(np.uint8)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img_obj = Image.fromarray(img_array)
    raw_img_obj = img_obj.copy()

    draw_obj = ImageDraw.Draw(img_obj)
    if labelAndScore == 0:
        for box in boxes:
            draw_a_rectangel_in_img(draw_obj, box, color='White', width=2, mode=mode)
    else:
        for box, a_score, a_label in zip(boxes, scores, labels):
            draw_a_rectangel_in_img_with_label_and_scores(draw_obj, box, a_score, a_label, color='White', width=2, mode=mode)

    if len(boxes) > 0:
        out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.7)
        imgname = './work_dirs/out_img/'+path+'/'+img_name
        out_img_obj.save(imgname)



def draw_a_rectangel_in_img(draw_obj, box, color, width, mode):
    if mode == 0:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        top_left, top_right = (x1, y1), (x2, y1)
        bottom_left, bottom_right = (x1, y2), (x2, y2)

        draw_obj.line(xy=[top_left, top_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_left, bottom_left],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[bottom_left, bottom_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_right, bottom_right],
                      fill=color,
                      width=width)
    else:
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
        rect = ((x_c, y_c), (w, h), -theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                      fill=color,
                      width=width)

def draw_a_rectangel_in_img_with_label_and_scores(draw_obj, box, score, label, color, width, mode):
    if mode == 0:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        top_left, top_right = (x1, y1), (x2, y1)
        bottom_left, bottom_right = (x1, y2), (x2, y2)

        draw_obj.line(xy=[top_left, top_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_left, bottom_left],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[bottom_left, bottom_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_right, bottom_right],
                      fill=color,
                      width=width)
        # txt = LABEL_NAME_MAP[label] + ':' + str(round(score, 2))
        txt = ' '*int(label) + str(label)
        draw_obj.text(xy=(x1, y1),
                      text=txt,
                      fill='blue',
                      font=FONT)
    else:
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
        rect = ((x_c, y_c), (w, h), -theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                      fill=color,
                      width=width)
        txt = LABEL_NAME_MAP[label] + ':' + str(round(score, 2))
        # txt = ' '*int(label) + str(label)
        draw_obj.text(xy=(rect[0][0], rect[0][1]),
                      text=txt,
                      fill='blue',
                      font=FONT)

if __name__ == '__main__':
    # img_array = cv2.imread("1.png")
    img_array = Image.open('1.png')
    img_array = np.array(img_array, np.float32)
    boxes = np.array(
        [[500, 500, 50, 200, 0],
         [500, 500, 50, 200, 30],
         [500, 500, 50, 200, 60],
         [500, 500, 50, 200, 90],
         [500, 500, 50, 200, 120],
         [500, 500, 50, 200, 150]]
    )

    boxes2 = np.array(
        [[300, 300, 400, 400],
         [400, 400, 500, 500],
         [500, 500, 600, 600],
         [600, 600, 700, 700],
         [700, 700, 800, 800],
         [800, 800, 900, 900]]
    )

    # test only draw boxes
    labels = np.random.randint(len(boxes), size=boxes.shape[0])
    scores = np.random.rand(boxes.shape[0]) + 1
    imm = draw_boxes_with_label_and_scores(img_array, boxes2, mode=0)
    imm.save('3.png')