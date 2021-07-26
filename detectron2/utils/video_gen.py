import cv2
import numpy as np
from glob import glob
import os
import argparse

def video_write(image_list, save_name, frame=59.8):
    img = cv2.imread(image_list[0])
    height, width, layers = img.shape
    size = (width, height)
    out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'DIVX'), frame, size)
    print('Loaded')

    for image_path in image_list:
        print(image_path)
        img = cv2.imread(image_path)
        out.write(img)

    out.release()

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin base_configs")
    parser.add_argument('--input_dir', default='', type=str)
    parser.add_argument('--video_path', default='', type=str)
    parser.add_argument('--frame', default=32, type=int)
    parser = parser.parse_args()
    return parser

if __name__=='__main__':
    args = get_parser()

    image_list = glob(os.path.join(args.input_dir, '*.png'))
    image_list = sorted(image_list, key=lambda x: int(x.rstrip('.png').split('_')[-1]))
    video_write(image_list, save_name=args.video_path, frame=args.frame)