import numpy as np
import pandas as pd
import os
from mr.datasets.tower import StanfordTower as st
from mr.datasets.common import ImageSet
from mr.datasets.imageDataset import ImageDataset
from mr.learn.scaffold import Scaffold
from mr.learn.convolve import ConvolveLayer
from mr.learn.convolve import PoolLayer
from mr.learn.unsupervised.lca import Lca
from mr.learn.supervised.perceptron import Perceptron
import matplotlib.image as img
import matplotlib.pyplot as plt
import datetime
import pickle



folder = '/stash/tlab/datasets/Tower'
file_pre = 'Neovision2-Training-Tower-'
w_new = 36
train_folders = ['1']
test_folders = ['13']
'''
pickle_file = open('/u/jc2/dev/jc2/cnn/resources.p', 'rb')
pickle_data = pickle.load(pickle_file)
pickle_file.close()
train, test, vp = pickle_data
'''


## Testing preprocess - fewer images.  Need to find memory error
videos_to_train = [os.path.join(folder, 'dev_test', 'train', 'ims')]
videos_to_test = [os.path.join(folder, 'dev_test', 'test', 'ims')]
train_csv = [os.path.join(folder, 'dev_test', 'train', 'train.csv')]
test_csv = [os.path.join(folder, 'dev_test', 'test', 'test.csv')]







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
t = st(w_new, videos_to_train, videos_to_test, train_csv, test_csv )
train, test, vp = t.split()
stop = datetime.datetime.now()
print ("Total min to load: {}".format((stop-start).total_seconds() / 60))




## Set up model
print ("Building model")
model = Scaffold()
c = ConvolveLayer(layer = Lca(15), visualParams = vp, convSize = 4,
            convStride = 2)
c._init(len(train[0][0]), None)
model.layer(c)
p = PoolLayer(visualParams = c.visualParams)
model.layer(p)
model.layer(Perceptron())

## Train and test model
print ("Training model")
start = datetime.datetime.now()
model.fit(*train)
print (model.layers[0].nOutputs)
print (model.layers[0].nOutputsConvolved)
stop = datetime.datetime.now()
train_min = (stop - start).total_seconds() / 60
print ("Total min to train: {}".format(train_min))
print ("Testing model")
start = datetime.datetime.now()
print (model.score(*test))
stop = datetime.datetime.now()
test_min = (stop - start).total_seconds() / 60
print ("Total min to train: {}".format(test_min))



