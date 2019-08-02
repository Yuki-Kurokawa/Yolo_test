import cv2
import sys
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

pic_path = sys.argv[1]
img = cv2.imread(pic_path)
print("read image")
img_lbl, regions = selectivesearch.selective_search(img,scale=500,min_size=100)

boxes = set()

for region in regions:
    if region["rect"] in boxes:
        continue

    if region["size"] <= 1000:
        continue
    
    boxes.add(region["rect"])

for box in boxes:
    print(box)

print(len(boxes))
