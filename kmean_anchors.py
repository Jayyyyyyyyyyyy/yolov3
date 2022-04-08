#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:43:39 2019
@author: deniz
cat * > labels data
"""

import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;

sns.set()  # for plot styling

train_path = "all_anchors"


BBDlabeldict = {"person": 0,
                "drivable area": [],
                "lane": []}

w, h = [], []
with open(train_path, "r") as ftr:
    for line in ftr:
        res = line.strip().split(" ")
        x1, y1, width, height = float(res[1]), float(res[2]), int(float(res[3])*320), int(float(res[4])*240)
        w.append(width)
        h.append(height)
w = np.asarray(w)
h = np.asarray(h)

x = [w, h]
x = np.asarray(x)
x = x.transpose()
##########################################   K- Means
##########################################

from sklearn.cluster import KMeans

kmeans3 = KMeans(n_clusters=9)
kmeans3.fit(x)
y_kmeans3 = kmeans3.predict(x)

##########################################
centers3 = kmeans3.cluster_centers_

yolo_anchor_average = []
for ind in range(9):
    yolo_anchor_average.append(np.mean(x[y_kmeans3 == ind], axis=0))

yolo_anchor_average = np.array(yolo_anchor_average)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, s=2, cmap='viridis')
plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c='red', s=50);
yoloV3anchors = yolo_anchor_average
yoloV3anchors[:, 0] = yolo_anchor_average[:, 0] / 768 * 608
yoloV3anchors[:, 1] = yolo_anchor_average[:, 1] / 576 * 608
yoloV3anchors = np.rint(yoloV3anchors)
fig, ax = plt.subplots()
for ind in range(9):
    rectangle = plt.Rectangle((160 - yoloV3anchors[ind, 0] / 2, 120 - yoloV3anchors[ind, 1] / 2), yoloV3anchors[ind, 0],
                              yoloV3anchors[ind, 1], fc='b', edgecolor='b', fill=None)
    ax.add_patch(rectangle)
ax.set_aspect(1.0)
plt.axis([0, 320, 0, 240])
plt.show()
yoloV3anchors.sort(axis=0)
print("Your custom anchor boxes are {}".format(yoloV3anchors))

# F = open("YOLOV3_BDD_Anchors.txt", "w")
# F.write("{}".format(yoloV3anchors))
# F.close()