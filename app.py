import cv2
import numpy as np
import streamlit as st
from imageio import get_reader
from detect_image import run_model, detect, run_model_tscript, load_yolo_model
from utils.plots import plot_one_box

device = "cuda:1"


@st.cache(suppress_st_warning=True)
def yolo_compute(index, ts=False, weights="yolov5x.pt"):
    f = video.get_data(index)
    frameX = np.array(f)

    if ts:
        frameX = cv2.resize(frameX, (640, 480))
        return run_model_tscript(frameX, device=device).numpy()
    else:
        model = load_yolo_model(weights=weights, device=device)
        return run_model(frameX, model, device=device).numpy()


def detect_class(y_config, classes, thresh):
    y_config.classes = classes
    y_config.conf_thres = thresh

    bboxes = detect(y_config)

    for bbox in bboxes:
        label = f'{bbox[0]} {bbox[2]:.2f}'
        plot_one_box(x=bbox[1], img=y_config.im0, label=label,
                     color=y_config.colors[y_config.names.index(bbox[0])])


st.title("Check the labels for all objects")

img_placeholder = st.empty()
frame_placeholder = st.sidebar.empty()

video = get_reader("/home/winfried/pa_data/ananth.mp4")
use_ts = st.sidebar.checkbox("use torchscript")

fs = frame_placeholder.slider("frame", 0, 1000, 0, 5)

wconf = None
if not use_ts:
    wconf = st.sidebar.radio("weights", ["yolov5x.pt", "best.pt"])
res = yolo_compute(fs, ts=use_ts, weights=wconf).torch(device=device)  # , classes=[2], conf_thres=value

scl = st.sidebar.radio("show classes", ["all", "car", "pedestrians", "traffic lights"])
tcar = st.sidebar.slider("car threshold", 0.0, 1.0, 0.25, 0.01)
tped = st.sidebar.slider("pedestrians threshold", 0.0, 1.0, 0.25, 0.01)
ttrl = st.sidebar.slider("traffic lights threshold", 0.0, 1.0, 0.25, 0.01)

classes = []

if scl == "car" or scl == "all":
    detect_class(res, [2, 7], tcar)
if scl == "pedestrians" or scl == "all":
    detect_class(res, [0], tped)
if scl == "traffic lights" or scl == "all":
    detect_class(res, [9], ttrl)

img_placeholder.image(res.im0, width=640)

submit = st.sidebar.button('All correct')
mistakes = st.sidebar.button('Needs relabeling')
