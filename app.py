import numpy as np
import streamlit as st
from imageio import get_reader

from detect_image import detect_img

st.title("Check the labels for all cars")

submit = st.button('All correct')
mistakes = st.button('Needs relabeling')

img_placeholder = st.empty()
slider_ph = st.empty()

if submit:
    st.write("button pressed")

video = get_reader("/home/winfried/pa_data/ananth.mp4")
value = slider_ph.slider("model confidence threshold", 0.0, 1.0, 0.25, 0.01)

frameX = np.array(video.get_data(0))
result = detect_img(frameX, classes=[2], conf_thres=value)

img_placeholder.image(result, width=640)