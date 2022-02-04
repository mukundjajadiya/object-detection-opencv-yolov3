from email.mime import image
import os
from tkinter import image_names
import cv2 as cv
import numpy as np


class ObjectDetection:
    def __init__(self,
                 yolov3_weights_path="",
                 yolov3_cfg_path="",
                 yolov3_coco_names_path=""
                 ):

        self.__yolov3_weights_path = yolov3_weights_path
        self.__yolov3_cfg_path = yolov3_cfg_path
        self.__yolov3_coco_names_path = yolov3_coco_names_path

        if not os.path.exists(self.__yolov3_weights_path):
            print(
                f"[ERROR] yolov3_weights_path: {self.__yolov3_weights_path} does not exist.")
            return
        if not os.path.exists(self.__yolov3_cfg_path):
            print(
                f"[ERROR] yolov3_cfg_path: {self.__yolov3_cfg_path} does not exist.")
            return
        if not os.path.exists(self.__yolov3_coco_names_path):
            print(
                f"[ERROR] yolov3_coco_names_path: {self.__yolov3_coco_names_path} does not exist.")
            return

        with open(self.__yolov3_coco_names_path, 'r') as f:
            self.__classes = f.read().splitlines()

    def video_object_detection(self, video_path=None, camera_index=None):
        cv_window_name = ""

        if video_path != None:
            if os.path.exists(video_path):
                cap = cv.VideoCapture(video_path)
                cv_window_name = video_path.split("/")[-1]
            else:
                print(f"[ERROR] Video file path: {video_path} does not exist.")
                return
        else:
            cv_window_name = f"Camera {camera_index}"

            cap = cv.VideoCapture(camera_index)

            if cap is None or not cap.isOpened():       
                print(f"[ERROR] Camera {camera_index} is invalid.")
                return
                

        net = cv.dnn.readNet(self.__yolov3_weights_path,
                             self.__yolov3_cfg_path)

        RECTANGLE_COLOR = (0, 255, 0)  # Green color
        RECTANGLE_THICKNESS = int(2)

        FONT = cv.FONT_HERSHEY_SIMPLEX
        FONT_SIZE = 0.5
        FONT_COLOR = (255, 255, 255)  # White color
        FONT_THICKNESS = int(1)
        print(f"[INFO] Object Detecting on {cv_window_name}....")
        while True:
            try:
                _, img = cap.read()
                height, width, _ = img.shape

                blob = cv.dnn.blobFromImage(img, 1/255,
                                            (416, 416),
                                            (0, 0, 0),
                                            swapRB=True,
                                            crop=False)

                net.setInput(blob)
                output_layers_names = net.getUnconnectedOutLayersNames()
                layer_outputs = net.forward(output_layers_names)

                boxes = []
                confidences = []
                class_ids = []

                for output in layer_outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5:
                            center_x = int(detection[0]*width)
                            center_y = int(detection[1]*height)
                            w = int(detection[2]*width)
                            h = int(detection[3]*height)

                            x = int(center_x - w/2)
                            y = int(center_y - h/2)

                            boxes.append([x, y, w, h])
                            confidences.append((float(confidence)))
                            class_ids.append(class_id)

                indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                for i in indexes.flatten():
                    x, y, w, h = boxes[i]

                    label = str(self.__classes[class_ids[i]])

                    confidence = str(round(confidences[i], 2))

                    cv.rectangle(img,
                                 (x, y),
                                 (x+w, y+h),
                                 RECTANGLE_COLOR,
                                 RECTANGLE_THICKNESS)

                    cv.putText(img,
                               label + " " + confidence,
                               (x, y-5),
                               FONT,
                               FONT_SIZE,     # Font size
                               FONT_COLOR,    # Font color
                               FONT_THICKNESS  # Font thickness
                               )

                    cv.putText(img,
                               "Enter \"esc\" key to exit",
                               (5, 15),
                               FONT,
                               FONT_SIZE,     # Font size
                               (255, 255, 255),    # Font color
                               FONT_THICKNESS  # Font thickness
                               )

                cv.imshow(cv_window_name, img)
                key = cv.waitKey(1)
                if key == 27:
                    break
            except Exception as e:
                print(f"[INFO] Object not found in current frame.")

        cap.release()

        cv.destroyAllWindows()

    def image_object_detection(self, image_path=None):
        if image_path:
            if not os.path.exists(image_path):
                print(f"[ERROR] Image path: {image_path} does not exist.")
                return

        image_name = image_path.split("/")[-1]

        net = cv.dnn.readNet(self.__yolov3_weights_path,
                             self.__yolov3_cfg_path)

        RECTANGLE_COLOR = (0, 255, 0)  # Green color
        RECTANGLE_THICKNESS = int(2)

        FONT = cv.FONT_HERSHEY_SIMPLEX
        FONT_SIZE = 0.5
        FONT_COLOR = (255, 255, 255)  # White color
        FONT_THICKNESS = int(1)
        print(f"[INFO] Object Detecting on {image_name}....")

        try:
            img = cv.imread(image_path)
            height, width, channels = img.shape
            blob = cv.dnn.blobFromImage(img, 1/255,
                                        (416, 416),
                                        (0, 0, 0),
                                        swapRB=True,
                                        crop=False)

            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layer_outputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in indexes.flatten():
                x, y, w, h = boxes[i]

                label = str(self.__classes[class_ids[i]])

                confidence = str(round(confidences[i], 2))

                cv.rectangle(img,
                             (x, y),
                             (x+w, y+h),
                             RECTANGLE_COLOR,
                             RECTANGLE_THICKNESS)

                cv.putText(img,
                           label + " " + confidence,
                           (x, y-5),
                           FONT,
                           FONT_SIZE,     # Font size
                           FONT_COLOR,    # Font color
                           FONT_THICKNESS  # Font thickness
                           )

                cv.putText(img,
                           "Enter \"esc\" key to exit",
                           (5, 15),
                           FONT,
                           FONT_SIZE,     # Font size
                           (255, 255, 255),    # Font color
                           FONT_THICKNESS  # Font thickness
                           )

            cv.imshow(image_name, img)
            cv.waitKey(0)
        except Exception as e:
            print(f"[ERROR] {e}")
            print(f"[INFO] Object not found in current frame.")

        cv.destroyAllWindows()
