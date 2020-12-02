This is the yolov5 repo with some changes (https://github.com/ultralytics/yolov5). We are using the original weights as for now, but plan to train for some classes on our own. 

Usage:

#### Install all dependencies 
See requirements.txt

#### To find the best confidence threshold for your data:
```
./select_conf_threshold.sh 
--source ../path_to_your_video.mp4
--device cuda:0 
```
This will open a new browser tab, where you can interactively experiment with the threshold. Device should be a valid cuda/gpu index or "cpu".

#### Process video files:
```
python detect_video.py 
--source ../path_to_your_video.mp4
--device cuda:0 
--conf-thres 0.25 
--coco-out coco_labels.json 
--result predictions.mp4
```
Convert the whole video. It will generate coco label files and a processed video with boxes overlay. The confidence threshold is set with "conf-thres".