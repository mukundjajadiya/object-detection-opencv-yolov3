# Object Detection

### Object Detection detects the objects on the frame and tries to identify them.

This project uses yolov3 dependencies,

- [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
- [yolov3.config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

## Preview of Object Detection
Object Detection on image:
<p align="center">
  <img width="100%" height="100%" src="https://github.com/mukundjajadiya/object-detection-opencv-yolov3/blob/main/data/GIF/object_detection_on_image.png" alt="image">
</p>

Object Detection on video:
![](https://github.com/mukundjajadiya/object-detection-opencv-yolov3/blob/main/data/GIF/object_detection_on_video.gif)

## **Step: 1** Clone Object Detection repository
```bash
git clone https://github.com/mukundjajadiya/object-detection-opencv-yolov3.git
```
## **Step: 2** change directory
```bash
cd object-detection-opencv-yolov3
```
## **Step: 3** Installation
You have to run the following commands to install dependencies

```bash
pip install -r requirements.txt
```

## **Step: 4** Usage
Before you run app.py file you have to required yolov3 dependencies,
- [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
- [yolov3.config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

After downloading the above dependency set path in `app.py`,
- yolov3_weights_path
- yolov3_cfg_path
- yolov3_coco_names_path
  

### Object detection on video

```python
from ObjectDetection import ObjectDetection

OD = ObjectDetection(yolov3_weights_path="data/yolov3.weights",
                     yolov3_cfg_path="data/yolov3.cfg",
                     yolov3_coco_names_path="data/coco.names")

# Object detection on video
OD.video_object_detection(video_path = "data/videos/1.mp4")
```

or

### Object detection on camera

```bash
from ObjectDetection import ObjectDetection

OD = ObjectDetection(yolov3_weights_path="data/yolov3.weights",
                     yolov3_cfg_path="data/yolov3.cfg",
                     yolov3_coco_names_path="data/coco.names")

# Object detection on camera
OD.video_object_detection(camera_index = 0)
```

or

### Object detection on an image

```bash
from ObjectDetection import ObjectDetection

OD = ObjectDetection(yolov3_weights_path="data/yolov3.weights",
                     yolov3_cfg_path="data/yolov3.cfg",
                     yolov3_coco_names_path="data/coco.names")

# Object detection on an image
OD.image_object_detection(image_path = "data/images/1.jpg")
```
Run app.py python file to detect an objects 
```bash
python app.py
```
## License

[MIT](https://choosealicense.com/licenses/mit/)
