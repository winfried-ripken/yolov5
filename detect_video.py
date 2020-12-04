import argparse

import json
from pathlib import Path

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

    return {0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"}


def get_label_to_id_ours():
    return {"person": 0,
            "bicycle": 1,
            "car": 2,
            "motorcycle": 3,
            "bus": 4,
            "truck": 5}


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


class CocoLabelResultWriter:
    def __init__(self, json_file):
        self.json_file = json_file
        self.data = {}
        self.annotation_id_counter = 0
        self.id_to_c = get_label_to_id_ours()
        self.make_header()

    def dump(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f)

    def make_header(self):
        categories = []

        for x in self.id_to_c:
            categories.append({"name": x, "id": self.id_to_c[x]})

        self.data["categories"] = categories
        self.data["images"] = []
        self.data["annotations"] = []

    def append_frame(self, index, frame, detection):
        w, h = frame.shape[:2]

        self.data["images"].append({
            "width": w,
            "height": h,
            "id": index,
            "file_name": f"frame_{index:06d}"
        })

        for d in detection:
            self.data["annotations"].append({
                "id": self.annotation_id_counter,
                "category_id": self.id_to_c[d[0]],
                "bbox": [d[1][0], d[1][1], d[1][2] - d[1][0], d[1][3] - d[1][1]],
                "image_id": index
            })

            self.annotation_id_counter += 1


class CustomJsonLabelResultWriter:
    def __init__(self, json_file):
        self.json_file = json_file
        self.data = []

    def dump(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def append_frame(self, index, frame, detection):
        detections = []
        for d in detection:
            detections.append([
                d[0],
                [d[1][0], d[1][1], d[1][2], d[1][3]],
                d[2]])

        self.data.append(
            {
                "frame_counter": index,
                "detections": detections
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='video.mp4', help='source video')
    parser.add_argument('--result', type=str, default='predictions.mp4', help='result video')
    parser.add_argument('--coco-out', type=str, default='coco_labels.json', help='result labels')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--device', default='cuda:0', help='e.g. cuda:0 or cpu')

    opt = parser.parse_args()
    print(opt)

    if not Path(opt.source).exists():
        raise ValueError(f"invalid argument for --source {opt.source}. Path does not exist")

    writer = CustomJsonLabelResultWriter(opt.coco_out)
    model = load_yolo_model(device=opt.device)

    vid_result = get_writer(opt.result, fps=30)
    cap = cv2.VideoCapture(opt.source)
    ret, frame = cap.read()

    i = 0
    with torch.no_grad():
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, det_result = process_model_output(run_model(frame, model=model, device=opt.device, imgsz=640),
                                                     opt.conf_thres)
            writer.append_frame(i, frame, det_result)

            i += 1
            if i % 5 == 0:
                print(f"\rprocessed {i} frames", end="")

            vid_result.append_data(frame)
            ret, frame = cap.read()

    writer.dump()
    vid_result.close()
