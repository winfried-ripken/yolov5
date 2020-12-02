from copy import deepcopy

import cv2
import numpy as np
import os
import torch
from numpy import random
import pickle

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords


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

    def torch(self, device):
        xxx = deepcopy(self)

        if device.startswith("cuda"):
            return deepcopy(YoloConfig(torch.tensor(xxx.pred).cuda(), xxx.im0,
                                       torch.tensor(xxx.img).cuda(), xxx.names, xxx.colors))
        else:
            return deepcopy(YoloConfig(torch.tensor(xxx.pred), xxx.im0,
                                       torch.tensor(xxx.img), xxx.names, xxx.colors))


def load_yolo_model(device="cuda:0", torchscript=False):
    weights = "yolov5x.pt"
    device = torch.device(device)

    if torchscript:
        model = torch.jit.load("yolov5x.torchscript.pt", map_location=device)
    else:
        # Initialize
        model = attempt_load(weights, map_location=device)  # load FP32 model

    return model


def run_model(image_xx, model=None, device="cuda:0", imgsz=640):
    if model is None:
        model = load_yolo_model(device)

    augment = False
    device = torch.device(device)

    # Load model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    random.seed(1)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # code from dataloader to scale image
    im0 = image_xx
    img = letterbox(image_xx, new_shape=imgsz, scaleFill=True)[0]

    # Convert
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=augment)[0]
    return YoloConfig(pred, im0, img, names, colors)


def run_model_tscript(image_xx, model=None, device="cuda:0"):
    if model is None:
        model = load_yolo_model(device, True)

    imgsz = (640, 480)
    device = torch.device(device)

    # Get names and colors
    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
    with open('colors.pkl', 'rb') as f:
        colors = pickle.load(f)

    # code from dataloader to scale image
    im0 = image_xx
    img = cv2.resize(im0, imgsz)

    # Convert
    img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]
    return YoloConfig(pred, im0, img, names, colors)


def detect(y_config, output_indices=False):
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
                cc = int(cls) if output_indices else y_config.names[int(cls)]
                bboxes.append([cc, [el.item() for el in xyxy], conf.item()])

    return bboxes


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    im = np.random.random((720, 1280, 3))
    res = run_model(im, device="cpu")
    res = detect(res)
    print(res)
