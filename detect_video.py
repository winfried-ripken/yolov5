import argparse

import cv2
import torch
from imageio import get_writer

from detect_image import load_yolo_model, run_model, detect
from utils.plots import plot_one_box


def get_class_to_label():
    # COCO LABELS
    # 0 - person
    # 1 - bicycle
    # 2 - car
    # 3 - motorcycle
    # 5 - bus
    # 7 - truck

    return {0: "pedestrian",
            1: "bicycle",
            2: "vehicle",
            3: "motorcycle",
            5: "vehicle",
            7: "vehicle"}


def process_model_output(det, conf_thres):
    class_to_label = get_class_to_label()
    lbls = list(set(class_to_label.values()))
    det.classes = list(class_to_label.keys())
    det.conf_thres = conf_thres
    bboxes = detect(det, output_indices=True)

    for bbox in bboxes:
        lbl = class_to_label[bbox[0]]

        label = f'{lbl} {bbox[2]:.2f}'
        plot_one_box(x=bbox[1], img=det.im0, label=label,
                     color=det.colors[lbls.index(lbl)])

        bbox[0] = lbl

    return det.im0, bboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='video.mp4', help='source video')
    parser.add_argument('--result', type=str, default='predictions.mp4', help='result video')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--device', default='cuda:0', help='e.g. cuda:0 or cpu')

    opt = parser.parse_args()
    print(opt)

    model = load_yolo_model(device=opt.device)

    vid_result = get_writer(opt.result, fps=30)
    cap = cv2.VideoCapture(opt.source)
    ret, frame = cap.read()

    i = 0
    with torch.no_grad():
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, _ = process_model_output(run_model(frame, model=model, device=opt.device, imgsz=640), opt.conf_thres)

            i += 1
            if i % 5 == 0:
                print(f"\rprocessed {i} frames", end="")

            vid_result.append_data(frame)
            ret, frame = cap.read()

    vid_result.close()
