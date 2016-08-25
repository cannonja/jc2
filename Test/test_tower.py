import numpy as np
import pandas as pd
from mr.datasets.tower import StanfordTower as st
import matplotlib.image as img
import matplotlib.pyplot as plt


folder = '/stash/tlab/datasets/Tower/train/1'

## Initialize class and read annotation file
t = st()
df = t.read_file(folder)

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
mask = t.assign_classes(1920, 1088, boxes)
#mask = t.assign_classes(30, 30, boxes)
img.imsave(folder + '/mask0.png', mask, cmap = plt.get_cmap('gray_r'), vmin = 0,
                                vmax = 255)
np.savetxt(folder + '/mask_data.csv', mask, delimiter = ',')


