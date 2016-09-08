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



folder = '/stash/tlab/datasets/Tower'
file_pre = 'Neovision2-Training-Tower-'
w_new = 20
tau = 1
train_folders = ['1', '2', '3', '4', '5']
test_folders = ['13', '14']


## Debug - one image
videos_to_train = [os.path.join(folder, 'dev_test', 'train2', 'ims')]
videos_to_test = [os.path.join(folder, 'dev_test', 'test2', 'ims')]
train_csv = [os.path.join(folder, 'dev_test', 'train2', 'train.csv')]
test_csv = [os.path.join(folder, 'dev_test', 'test2', 'train.csv')]

'''
## Debug - more images.
videos_to_train = [os.path.join(folder, 'dev_test', 'train', 'ims')]
videos_to_test = [os.path.join(folder, 'dev_test', 'test', 'ims')]
train_csv = [os.path.join(folder, 'dev_test', 'train', 'train.csv')]
test_csv = [os.path.join(folder, 'dev_test', 'test', 'test.csv')]
'''

'''
## Preprocess
videos_to_train = [os.path.join(folder, 'train', i,
                            file_pre + str(i).zfill(3)) for i in train_folders]
videos_to_test = [os.path.join(folder, 'test', i,
                            file_pre + str(i).zfill(3)) for i in test_folders]
train_csv = [os.path.join(folder, 'train', i,
                file_pre + str(i).zfill(3) + '.csv') for i in train_folders]
test_csv = [os.path.join(folder, 'test', i,
                file_pre + str(i).zfill(3) + '.csv') for i in test_folders]
'''




## Initialize class and read annotation file
start = datetime.datetime.now()
print ("Loading video data")
t = st(w_new, videos_to_train, videos_to_test, train_csv, test_csv, tau)
print("Start Split!!!!")
train, test, vp = t.split()
stop = datetime.datetime.now()
print ("Total min to load: {}\n\n".format((stop-start).total_seconds() / 60))

pre_labs = t.masks[0][0].flatten()
post_labs = train[1][0]
print ("Pre labels:\nCount 1's = {}\nMin = {}\nMax = {}\nMean = {}\n".format(
    np.sum(pre_labs), pre_labs.min(), pre_labs.max(), pre_labs.mean()))
print ("Post labels:\nCount > 0 = {}\nMin = {}\nMax = {}\nMean = {}\n".format(
    np.sum(post_labs > 0), post_labs.min(), post_labs.max(), post_labs.mean()))





'''
## Set up model
print ("Building model")
model = Scaffold()
c = ConvolveLayer(layer = Lca(15), visualParams = vp, convSize = 10,
            convStride = 3)
c.init(len(train[0][0]), None)
model.layer(c)
p = PoolLayer(visualParams = c.visualParams)
model.layer(p)
model.layer(Perceptron())

## Train and test model
print ("Training model")
start = datetime.datetime.now()
model.fit(*train)
stop = datetime.datetime.now()
train_min = (stop - start).total_seconds() / 60
print ("Total min to train: {}".format(train_min))
'''

'''
pickle_file = open('/u/jc2/dev/jc2/cnn/model_data.p', 'wb')
pickle_data = [(train, test, vp), model]
pickle.dump(pickle_data, pickle_file)
pickle_file.close()
'''

'''
path = 'visualize.png'
path2 = 'visualize2.png'
model.visualize(vp, path)
model.visualize(vp, path2, inputs = test[0][0])
'''

