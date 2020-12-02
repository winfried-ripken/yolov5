import argparse
import cv2
import numpy as np
import streamlit as st
from imageio import get_reader

from detect_image import run_model, load_yolo_model
from detect_video import process_model_output


@st.cache(suppress_st_warning=True)
def get_yolo_model(device):
    print("loading model")
    model = load_yolo_model(device=device)
    return model


@st.cache(suppress_st_warning=True)
def yolo_compute(index, device):
    f = video.get_data(index)
    frameX = np.array(f)
    model = get_yolo_model(device)

    return run_model(frameX, model, device).numpy()


def frame_count(video_path, manual=False):
    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status:
                break
            frames += 1
        return frames

    cap = cv2.VideoCapture(video_path)
    # Slow, inefficient but 100% accurate method
    if manual:
        frames = manual_count(cap)
    # Fast, efficient but inaccurate method
    else:
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(cap)
    cap.release()
    return frames


parser = argparse.ArgumentParser(description='Check the confidence threshold')
parser.add_argument('--source', type=str, default='video.mp4', help='source video')
parser.add_argument('--device', default='cuda:0', help='e.g. cuda:0 or cpu')

opt = parser.parse_args()
print(opt)

st.title(f"Find the confidence threshold for {opt.source}")
num_frames = frame_count(opt.source)
video = get_reader(opt.source)

img_placeholder = st.empty()
curr_frame = st.sidebar.slider("frame", 0, int(num_frames) - 1, 0, 1)
show_pred = st.sidebar.checkbox("show predictions", True)

res = yolo_compute(curr_frame, opt.device)
res = res.torch(opt.device)
tconf = st.sidebar.slider("confidence threshold", 0.0, 1.0, 0.25, 0.01)

if show_pred:
    frame, _ = process_model_output(res, tconf)
else:
    frame = res.im0

img_placeholder.image(frame, width=640)
