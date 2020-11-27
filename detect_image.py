import argparse
import time
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class YoloConfig:
    def __init__(self, pred, im0, img, names, colors):
        self.img = img
        self.im0 = im0
        self.names = names
        self.colors = colors

        self.pred = pred
        self.agnostic_nms = False
        self.classes = None
        self.iou_thres = 0.45
        self.conf_thres = 0.25

    def numpy(self):
        self.pred = self.pred.cpu().numpy()
        self.img = self.img.cpu().numpy()
        return self

    def torch(self):
        xxx = deepcopy(self)
        return deepcopy(YoloConfig(torch.tensor(xxx.pred).cuda(), xxx.im0,
                          torch.tensor(xxx.img).cuda(), xxx.names, xxx.colors))


def run_model(image_xx):
    augment = False
    weights = "yolov5x.pt"
    imgsz = 640
    device = "0"

    # Initialize
    device = select_device(device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    random.seed(1)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # code from dataloader to scale image
    im0 = image_xx
    img = letterbox(image_xx, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=augment)[0]
    return YoloConfig(pred, im0, img, names, colors)


def detect(y_config):
    # Apply NMS
    pred = non_max_suppression(y_config.pred, y_config.conf_thres, y_config.iou_thres, classes=y_config.classes,
                               agnostic=y_config.agnostic_nms)

    bboxes = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(y_config.im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(y_config.img.shape[2:], det[:, :4], y_config.im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                bboxes.append((y_config.names[int(cls)], [el.item() for el in xyxy], conf.item()))

    return bboxes


if __name__ == '__main__':
    im = np.random.random((480, 480, 3))
    res = run_model(im)
    res = detect(res)
    print(res)
