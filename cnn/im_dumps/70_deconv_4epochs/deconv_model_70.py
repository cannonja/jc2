import numpy as np
import pandas as pd
import os
from natsort import natsorted
import skvideo.io
from mr.datasets.tower import StanfordTower as st
from mr.datasets.tower import TowerScaffold
from mr.datasets.common import ImageSet
from mr.datasets.imageDataset import ImageDataset
from mr.learn.scaffold import Scaffold
from mr.learn.convolve import ConvolveLayer
from mr.learn.convolve import PoolLayer
from mr.learn.convolve import DeconvolveLayer
from mr.learn.unsupervised.lca import Lca
from mr.learn.supervised.perceptron import Perceptron
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import pickle



folder = '/stash/tlab/datasets/Tower'
file_pre = 'Neovision2-Training-Tower-'
w_new = 70
num_epochs = 10
cnn_params = (10, 5)  #(convSize, convStride)
deconv_params = (10, 5)  #(deconvSize, deconvStride)
num_classes = 5
tau = 1
train_folders = ['1', '2', '3', '4', '5']
test_folders = ['13', '14']



'''
## Debug
videos_to_train = [os.path.join(folder, 'dev_test', 'train2', 'ims')]
videos_to_test = [os.path.join(folder, 'dev_test', 'test2', 'ims')]
train_csv = [os.path.join(folder, 'dev_test', 'train2', 'train.csv')]
test_csv = [os.path.join(folder, 'dev_test', 'test2', 'train.csv')]
'''

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


## Initialize class and read annotation file
start = datetime.datetime.now()
print ("Loading video data")
t = st(w_new, videos_to_train, videos_to_test, train_csv, test_csv, tau)
train, test, vp = t.split()
stop = datetime.datetime.now()
print ("Total min to load: {}".format((stop-start).total_seconds() / 60))

## Set up model
print ("Building model")
model = Scaffold()
c = ConvolveLayer(layer = Lca(15), visualParams = vp, convSize = cnn_params[0],
            convStride = cnn_params[1])
c.init(len(train[0][0]), None)
model.layer(c)
d = DeconvolveLayer(layer = Perceptron(), visualParams = vp[:2] + (c.visualParams[2], num_classes),
        convSize = c.convSize, convStride = c.convStride, deconvSize = deconv_params[0],
        deconvStride = deconv_params[1])
model.layer(d)
model.init(len(train[0][0]), len(train[1][0]))


## Train and test model
start = datetime.datetime.now()
model2 = TowerScaffold()

for i in range(num_epochs):
    print ("Starting Epoch {}".format(i + 1))
    e_start = datetime.datetime.now()
    model.partial_fit(*train)
    e_stop = datetime.datetime.now()
    e_min = (e_stop - e_start).total_seconds() / 60
    print ("Epoch {} min: {}".format(i + 1, e_min))

    xP = model.predict(test[0], False)
    xP[xP > 0.5] = 1
    xP[xP <= 0.5] = 0
    y = np.asarray(test[1]).copy()
    dice = model2._calc_Dice(xP, y)
    print ("Avg Dice after Epoch #{}: {}".format(i + 1, np.mean(dice)))

    print("Dumping Dice plot\n")
    #t.dump_ims(orig, labels, predictions)
    fig = model2.plot_Dices(dice, t.classes)
    fig.savefig("dice_epoch_{}.png".format(i + 1))


stop = datetime.datetime.now()
train_min = (stop - start).total_seconds() / 60
print ("Total min to run all epochs: {}".format(train_min))

## Dump visualizations

apath1 = os.path.join(folder, 'test', '13', 'Neovision2-Training-Tower-013')
apath2 = os.path.join(folder, 'test', '14', 'Neovision2-Training-Tower-014')
test_files1 = [os.path.join(apath1, i) for i in natsorted(os.listdir(apath1))]
test_files2 = [os.path.join(apath2, i) for i in natsorted(os.listdir(apath2))]
test_files = test_files1 + test_files2



## Run program to save images, masks, predictions, and dice plot

print("Constructing lists")
print("\torig...........")
orig = [np.asarray(Image.open(i).resize((vp[0], vp[1]))) for i in test_files]
print("\tlabels.........")
labels = [t.combine_classes(y[i, :].reshape(5, np.prod(vp[:2])),
                vp[0], vp[1]) for i in range(y.shape[0])]
print("\tpredictions....")
predictions = [t.combine_classes(xP[i, :].reshape(5, np.prod(vp[:2])),
                vp[0], vp[1]) for i in range(xP.shape[0])]




vid_data = np.zeros((len(orig), vp[1], vp[0], vp[2]), dtype = np.uint8)
frames = natsorted(os.listdir('im_dump/orig'))
in_data = [np.asarray(Image.open(os.path.join('im_dump', 'orig', i))) for i in frames]
for i in range(len(in_data)):
    vid_data[i, :, :, :] = in_data[i]
skvideo.io.vwrite('orig.mp4', vid_data)

vid_data = np.zeros((len(labels), vp[1], vp[0], vp[2]), dtype = np.uint8)
frames = natsorted(os.listdir('im_dump/labels'))
in_data = [np.asarray(Image.open(os.path.join('im_dump', 'labels', i))) for i in frames]
for i in range(len(in_data)):
    vid_data[i, :, :, :] = in_data[i]
skvideo.io.vwrite('labels.mp4', vid_data)

vid_data = np.zeros((len(predictions), vp[1], vp[0], vp[2]), dtype = np.uint8)
frames = natsorted(os.listdir('im_dump/predictions'))
in_data = [np.asarray(Image.open(os.path.join('im_dump', 'predictions', i))) for i in frames]
for i in range(len(in_data)):
    vid_data[i, :, :, :] = in_data[i]
skvideo.io.vwrite('predictions.mp4', vid_data)

