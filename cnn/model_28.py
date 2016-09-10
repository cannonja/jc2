import numpy as np
import pandas as pd
import os
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import pickle
from mr.datasets.common import ImageSet
from mr.datasets.imageDataset import ImageDataset


from mr.datasets.tower import StanfordTower as st
from mr.datasets.tower import TowerScaffold
from mr.learn.scaffold import Scaffold
from mr.learn.convolve import ConvolveLayer
from mr.learn.convolve import PoolLayer
from mr.learn.unsupervised.lca import Lca
from mr.learn.supervised.perceptron import Perceptron



folder = '/stash/tlab/datasets/Tower'
file_pre = 'Neovision2-Training-Tower-'
w_new = 20
tau = 1
train_folders = ['1', '2', '3', '4', '5']
test_folders = ['13', '14']

"""pickle_data = (model, model2, vp, xP, y)"""

pickle_file = open('model_28.p', 'rb')
pickle_data = pickle.load(pickle_file)
pickle_file.close()

model, model2, vp, xP, y = pickle_data
model2 = TowerScaffold()


videos_to_train = [os.path.join(folder, 'dev_test', 'train2', 'ims')]
videos_to_test = [os.path.join(folder, 'dev_test', 'test2', 'ims')]
train_csv = [os.path.join(folder, 'dev_test', 'train2', 'train.csv')]
test_csv = [os.path.join(folder, 'dev_test', 'test2', 'train.csv')]
t = st(vp[0], videos_to_train, videos_to_test, train_csv, test_csv)

dices = model2._get_Dices(xP, y, 5)
fig = model2.plot_Dices(dices, t.classes)
plt.show()
