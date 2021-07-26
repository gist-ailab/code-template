import cv2
import argparse
import pickle
import os
from tqdm import tqdm
import numpy as np
import sys
from tracking import Sort

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin base_configs")
    parser.add_argument('--result_path', default='../sample/result_clip_2.pkl', type=str)
    parser.add_argument('--save_folder', default='./vis_clip2', type=str)
    parser.add_argument('--confidence_threshold', default=0.8, type=float)
    parser.add_argument('--target_class', default='', type=str)
    parser = parser.parse_args()
    return parser

def draw_box(img, box, text, color):
    box = map(int, box)
    x1, y1, x2, y2 = box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=color, thickness=1)
    return img

if __name__=='__main__':
    args = get_parser()
    os.makedirs(args.save_folder, exist_ok=True)

    # Enhancement
    show_enhance = False
    tracking = True
    if tracking:
        # max_age : Maximum number of frames to keep alive a track without associated detections.
        # min_hits : Minimum number of associated detections before track is initialised.
        # iou_threshold : Minimum IOU for match.
        tracker_0 = Sort(max_age=40, min_hits=100, iou_threshold=0.3)
        tracker_1 = Sort(max_age=40, min_hits=100, iou_threshold=0.3)
        tracker_2 = Sort(max_age=40, min_hits=20, iou_threshold=0.3)

    with open(args.result_path, 'rb') as f:
        result = pickle.load(f)

    target_class = args.target_class.strip().split(',')

    class_dict = {
        0: ['Connector', (36, 255, 12)],
        1: ['Marker', (37, 11, 255)],
        2: ['Fire', (255, 36, 12)],
    }

    image_list = sorted(list(result.keys()), key=lambda x: int(x.rstrip('.png').split('_')[-1]))
    for img_path in tqdm(image_list):
        if show_enhance:
            img = cv2.imread(img_path.replace('raw', 'enhancement'))
        else:
            img = cv2.imread(img_path)
        box_info = result[img_path]

        box_list_0 = []
        box_list_1 = []
        box_list_2 = []
        for r in box_info:
            box_ = r[0]
            score_ = r[1]
            class_ = r[2]

            if score_ < args.confidence_threshold:
                continue

            if target_class[0] != '' and str(class_) not in target_class:
                continue

            class_name = class_dict[class_][0]
            class_color = class_dict[class_][1]
            text_ = '%s-%d%%' %(class_name, int(score_ * 100))

            if tracking:
                if class_ == 0:
                    box_list_0.append([box_[0], box_[1], box_[2], box_[3], score_])
                elif class_ == 1:
                    box_list_1.append([box_[0], box_[1], box_[2], box_[3], score_])
                elif class_ == 2:
                    box_list_2.append([box_[0], box_[1], box_[2], box_[3], score_])

            else:
                img = draw_box(img, box_, text_, class_color)

        if tracking:
            if len(box_list_0) > 0:
                trackers_0 = tracker_0.update(np.array(box_list_0))
                for d in trackers_0:
                    d = d.astype(np.int32)

                    box = [d[0], d[1], d[2], d[3]]
                    id = d[4]

                    class_ = 0
                    class_name = class_dict[class_][0]
                    class_color = class_dict[class_][1]

                    text = 'Class:%s-ID:%d' %(class_name, id)
                    img = draw_box(img, box, text, class_color)

            if len(box_list_1) > 0:
                trackers_1 = tracker_1.update(np.array(box_list_1))
                for d in trackers_1:
                    d = d.astype(np.int32)

                    box = [d[0], d[1], d[2], d[3]]
                    id = d[4]

                    class_ = 1
                    class_name = class_dict[class_][0]
                    class_color = class_dict[class_][1]

                    text = 'Class:%s-ID:%d' % (class_name, id)
                    img = draw_box(img, box, text, class_color)

            if len(box_list_2) > 0:
                trackers_2 = tracker_2.update(np.array(box_list_2))
                for d in trackers_2:
                    d = d.astype(np.int32)

                    box = [d[0], d[1], d[2], d[3]]
                    id = d[4]

                    class_ = 2
                    class_name = class_dict[class_][0]
                    class_color = class_dict[class_][1]

                    text = 'Class:%s-ID:%d' % (class_name, id)
                    img = draw_box(img, box, text, class_color)

        save_name = os.path.basename(img_path)
        save_path = os.path.join(args.save_folder, save_name)
        cv2.imwrite(save_path, img)







