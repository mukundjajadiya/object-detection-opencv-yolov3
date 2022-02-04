from ObjectDetection import ObjectDetection

OD = ObjectDetection(yolov3_weights_path="yolov3/yolov3.weights",
                     yolov3_cfg_path="yolov3/yolov3.cfg",
                     yolov3_coco_names_path="yolov3/coco.names")

# Object detection on video
OD.video_object_detection(video_path="videos/1.mp4")

# Object detection on camera
OD.video_object_detection(camera_index=0)

# Object detection on image
OD.image_object_detection(image_path="images/1.jpg")
