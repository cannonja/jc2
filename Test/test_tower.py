import numpy as np
import pandas as pd
import os
from mr.datasets.tower import StanfordTower as st
from mr.datasets.common import ImageSet
from mr.datasets.imageDataset import ImageDataset
import matplotlib.image as img
import matplotlib.pyplot as plt


folder = '/stash/tlab/datasets/Tower/train/dev_test'

## Initialize class and read annotation file
t = st()
df = t.read_file(folder)

## Make sure I understand how to use ImageDataset
expected = os.listdir(folder + '/ims')
expected.sort()
vid = ImageDataset(folder + '/ims', expected)
imgs = vid.resize(177, 100).split(0, noLabels=True)[0][0]
img.imsave(folder + '/test000000.png', imgs[0].reshape(100, 177, 3))
img.imsave(folder + '/test000001.png', imgs[1].reshape(100, 177, 3))

















'''
## Test on first frame in the first video clip
print("Testing intersection methods")
vertices = df.iloc[0, 1:9].values.reshape(4,2)
box = t.box_2_poly(vertices)
pix_in = t.pix_2_poly(568, 27)
pix_out = t.pix_2_poly(490, 52)


## Check intersections
print("pix_in returns {}".format(box.intersects(pix_in)))
print("pix_out returns {}".format(box.intersects(pix_out)))


print("Testing class_check")
row = df[df.Frame == 0].iloc[0, 1:10].values
print("pix_in returns {}".format(t.class_check(568, 27, row)))
print("pix_out returns {}".format(t.class_check(490, 52, row)))


## Check masking
print("Testing assign classes")
boxes = df[df.Frame == 0].iloc[:, 1:10]

wn, hn, bn = t.resize_boxes(1920, 1088, 60, boxes.iloc[:, :-1])
boxes2 = pd.DataFrame(boxes).copy()
boxes2.iloc[:, :-1] = bn
mask = t.assign_classes(wn, hn, boxes2)
img.imsave(folder + '/mask0.png', mask, cmap = plt.get_cmap('gray_r'), vmin = 0,
                                vmax = 255)
'''
'''
mask = t.assign_classes(1920, 1088, boxes)
#mask = t.assign_classes(30, 30, boxes)
np.savetxt(folder + '/mask_data.csv', mask, delimiter = ',')
'''
