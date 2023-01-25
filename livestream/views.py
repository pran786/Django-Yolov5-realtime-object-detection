from django.shortcuts import render
from django.http import StreamingHttpResponse
import yolov5
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
import cv2
from PIL import Image as im
import torch
from yolov5.utils.general import *
# Create your views here.
def index(request):
    return render(request,'index.html')

model = yolov5.load('yolov5s.pt')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = select_device('cpu') # 0 for gpu, '' for cpu

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
hide_labels=False
hide_conf = False
def stream():
    cap = cv2.VideoCapture(0)
    model.conf = 0.25
    model.iou = 0.5
   # model.classes = [0,64,39]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break

        results = model(frame, augment=True)
        # proccess
        annotator = Annotator(frame, line_width=2, pil=not ascii) 
        det = results.pred[0]
        if det is not None and len(det):  
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True)) 

        im0 = annotator.result() 
       
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')  

def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')    


def index(request):
    return render(request,'index.html')


# def video_feed(request):
#     return StreamingHttpResponse(stream(),content_type='multipart/x-mixed-replace; boundary=frame')

# def stream():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret,frame = cap.read()
#         if not ret:
#             print('failed to read camera')
#         img_bytes = cv2.imencode('.jpg',frame)[1].tobytes()
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+ img_bytes +b'\r\n')