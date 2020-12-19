import json
import numpy as np
import cv2
import torch
from imageio import get_writer
from detect_image import load_yolo_model, run_model, detect

if __name__ == '__main__':
    with open("toolchain/yolo_detection_export.json", "r") as f:
        config = json.load(f)

    device = "cuda:1"
    conf_thres = 0.25
    model = load_yolo_model(device=device)

    numpy_result = []
    cap = cv2.VideoCapture(config["vid_in"])
    ret, frame = cap.read()

    i = 0
    with torch.no_grad():
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_frame = np.zeros_like(frame, dtype=np.uint8)

            det = run_model(frame, model=model, device=device, imgsz=640)
            class_to_label = {0: 1,  # "person"
                              1: 2,  # "bicycle"
                              2: 4,  # "car"
                              3: 8,  # "motorcycle"
                              5: 16,  # "bus"
                              7: 32}  # "truck"

            det.classes = list(class_to_label.keys())
            det.conf_thres = conf_thres
            bboxes = detect(det, output_indices=True)

            for bbox in bboxes:
                lbl = class_to_label[bbox[0]]

                result_frame[int(bbox[1][1]):int(bbox[1][3]), int(bbox[1][0]):int(bbox[1][2])] += lbl

            i += 1
            if i % 5 == 0:
                print(f"\rprocessed {i} frames", end="")

            numpy_result.append(result_frame.astype(np.uint8))
            ret, frame = cap.read()

    np.save(config["numpy_out"], np.stack(numpy_result))
