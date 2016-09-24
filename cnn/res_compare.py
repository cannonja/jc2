import numpy as np
import pandas as pd
import os
from mr.datasets.tower import StanfordTower as st
from mr.datasets.tower import TowerScaffold
from mr.datasets.common import ImageSet
from mr.datasets.imageDataset import ImageDataset
from mr.learn.scaffold import Scaffold
from mr.learn.convolve import ConvolveLayer
from mr.learn.convolve import PoolLayer
from mr.learn.unsupervised.lca import Lca
from mr.learn.supervised.perceptron import Perceptron
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import pickle



pickle_file = open("/u/jc2/dev/jc2/cnn/model_{}.p".format(w_new), 'wb')
pickle_data = (t, test, train, vp, model, model2, xP, y)
pickle.dump(pickle_data, pickle_file)
pickle_file.close()

res = [28, 35, 50, 70]
res_names = [str(i) for i in res]
t2 = TowerScaffold()
xPs = []
ys = []
dices = []

## Get data
for i in res:
    pickle_file = open("/u/jc2/dev/jc2/cnn/model_{}.p".format(i), 'rb')
    t, _, _, _, _, _, xP, y = pickle.load(pickle_file)
    xPs.append(xP.copy())
    ys.append(ys.copy())
    pickle_file.close()
    dices.append(t2._get_Dices(xP, y, 5))

fig = t2.plot_Dice_res(dices, res_names, t.classes)
fig.savefig('dice_res.png')






