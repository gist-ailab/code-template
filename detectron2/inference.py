# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from torch.utils.data import Dataset, DataLoader

import torch
from collections import defaultdict
import pickle

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin base_configs")
    parser.add_argument(
        "--config_file",
        default="base_configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int
    )
    parser.add_argument(
        "--nms_threshold",
        default=0.5,
        type=float
    )
    parser.add_argument(
        "--gpu",
        default='1',
        type=str
    )
    return parser

class inference_dset(Dataset):
    def __init__(self, image_list, cfg, show_enhance=False):
        self.cfg = cfg
        self.image_list = image_list

        self.input_format = cfg.INPUT.FORMAT
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.show_enhance = show_enhance

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if self.show_enhance:
            original_image = read_image(self.image_list[index].replace('raw', 'enhance'), format="BGR")
        else:
            original_image = read_image(self.image_list[index], format="BGR")

        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]

        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        input = {"image": image, "height": height, "width": width}
        return input

    def collate_fn(self, samples):
        output = [sample for sample in samples]
        return output


if __name__ == "__main__":
    args = get_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Option
    cfg = setup_cfg(args)

    # Load Model
    model = build_model(cfg)
    model.eval()

    # Load Weight
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # Image List
    image_name_list = glob.glob(os.path.join(args.input, '*.png'))

    # Inference
    dset = inference_dset(image_name_list, cfg)
    dloader = DataLoader(dset, shuffle=False, batch_size=args.batch_size, collate_fn=dset.collate_fn)

    output = {}
    ix = 0

    for image_list in tqdm(dloader):
        with torch.no_grad():
            out = model(image_list)

            for instance in out:
                output[image_name_list[ix]] = []
                instance = instance['instances'].to('cpu')

                if len(instance) > 0:
                    boxes = instance.pred_boxes
                    scores = instance.scores
                    classes = instance.pred_classes

                    for b, s, c in zip(boxes, scores, classes):
                        box_info = [[float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(s), int(c)]
                        output[image_name_list[ix]].append(box_info)

                ix += 1

    # Save the inference Result
    save_basename = args.input.split('/')[-1]
    save_name = os.path.join(args.output, 'result_%s.pkl' %save_basename)
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    with open(save_name, 'wb') as f:
        pickle.dump(output, f)