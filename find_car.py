import sys
import matplotlib.pyplot as plt
import bisect
import numpy as np
import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv3
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox
import cv2

chainer.config.cv_resize_backend = "cv2"

def scale_to_width(img, width):
    scale = width / img.shape[1]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)

def predict_bbox(img):
    chainer_img = trans_img_chainer(img)
    model = YOLOv3(pretrained_model="voc0712")
    bboxes, labels, scores = model.predict([chainer_img])
    return bboxes, labels, scores

def calc_ratio(box):
    height = int(box[2] - box[0])
    width = int(box[3] - box[1])
    return height, width

def trans_img_chainer(img):
    print(img)
    buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = np.asanyarray(buf, dtype=np.float32).transpose(2, 0, 1)
    return dst

def triming_pic(img,box):
    return img[int(box[0]):int(box[2]),int(box[1]):int(box[3])]

def focus_car(img,bboxes,scores,labels):
    for label_index, label in enumerate(labels[0]):
        print(label)
        if label == 6:
            index = label_index
            break
        else:
            pass
    
    box = bboxes[0][index]
    trim_pic = triming_pic(img,box)
    zoom_to_trim_pic = scale_to_width(trim_pic,1920)
    return zoom_to_trim_pic
    

def stream_video(video_path,mode):
    video_path = sys.argv[1]
    video = cv2.VideoCapture(video_path)

    while(video.isOpened()):
        ret,img = video.read()
        bboxes, labels,scores = predict_bbox(img)
        zoom_to_trim_pic = focus_car(img,bboxes,scores,labels)
        cv2.imwrite("focus_car.jpg",zoom_to_trim_pic)

        cv2.imshow("car",zoom_to_trim_pic)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

path = sys.argv[1]
stream_video(path,0)