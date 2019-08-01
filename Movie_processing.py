import matplotlib.pyplot as plt
import cv2
import os
import sys

def capture_frame_from_video(video_path,dir_path):
    cap = cv2.VideoCapture(video_path)
    count = 0

    if not cap.isOpened():
        print("Not Video your path")
        return
    
    os.makedirs(dir_path,exist_ok=True)

    while(1):
        ret,flame = cap.read()
        if count == 10:
            break
        if not ret:
            print("Not read video")
            return
        cv2.imwrite("{0}/{1}.jpg".format(dir_path,count),flame)
        count += 1


def find_BBox(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = sys.argv[1]
video = cv2.VideoCapture(path)
while(video.isOpened()):
    ret,cap = video.read()
    cv2.imshow("movie",cap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break