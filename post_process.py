# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""
import time
import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import (LOGGER, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors

class LoadWebcam:
    def __init__(self, img_size=[480,640], stride=32):
        self.img_size = img_size
        self.stride = stride
        self.cap = cv2.VideoCapture(0)  # video capture object

    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration
        # Read frame
        ret_val, img0 = self.cap.read()
        # img = cv2.resize(img0, self.img_size, interpolation=cv2.INTER_CUBIC)
        img = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img, img0
    def __len__(self):
        return 0

@torch.no_grad()
def run(weights='./model/yolov3-tiny.onnx',
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=True,  # class-agnostic NMS
        line_thickness=3,  # bounding box thickness (pixels)
        dnn=True,  # use OpenCV DNN for ONNX inference
        ):

    # Load model
    model = DetectMultiBackend(weights, dnn=dnn)
    names = ['PERSON']


    # Dataloader
    dataset = LoadWebcam()
    sum = 0
    for im, im0s in dataset:
        if dataset.count>0 and dataset.count % 100==0:
            print(sum/dataset.count)
        im = torch.from_numpy(im).float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]
        if len(im0s.shape) == 3:
            im0s = im0s[None]
        # Inference
        t1 = time.time()
        pred = model(im)
        t2 = time.time()
        infertime = round(t2-t1,2)
        sum += infertime

        # print("infer time is: {}".format(infertime))
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    # label = f'{names[c]} {conf:.2f}'
                    label = f'aaaaa'
                    annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()

            cv2.imshow('demo', im0)
            cv2.waitKey(1)  # 1 millisecond
run()
