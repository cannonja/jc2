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
## 70 x 40
## Debug
videos_to_train = [os.path.join(folder, 'dev_test', 'train2', 'ims')]
videos_to_test = [os.path.join(folder, 'dev_test', 'test2', 'ims')]
train_csv = [os.path.join(folder, 'dev_test', 'train2', 'train.csv')]
test_csv = [os.path.join(folder, 'dev_test', 'test2', 'train.csv')]
t = st(6, videos_to_train, videos_to_test, train_csv, test_csv)


apath1 = os.path.join(folder, 'test', '13', 'Neovision2-Training-Tower-013')
apath2 = os.path.join(folder, 'test', '14', 'Neovision2-Training-Tower-014')
test_files1 = [os.path.join(apath1, i) for i in os.listdir(apath1)]
test_files2 = [os.path.join(apath2, i) for i in os.listdir(apath2)]
test_files = test_files1 + test_files2

##pickle_data = (model, model2, vp, xP, y, dice)
print ("Unpickling...")
pickle_file = open("/u/jc2/dev/jc2/cnn/model_70.p", 'rb')
model, model2, vp, xP, y = pickle.load(pickle_file)
dice = model2._get_Dices(xP, y, len(t.classes))
pickle_file.close()


print("Dumping visuals")
#orig = [(i * 255.).reshape(vp[1], vp[0], vp[2]).astype(np.uint8) for i in test[0]]
orig = [np.asarray(Image.open(i).resize((70, 40))) for i in test_files]
labels = [y[i].reshape(5, np.prod(vp[:2])) for i in range(y.shape[1])]
#labels = t.vis_masks[1][-len(orig):]
predictions = [t.combine_classes(xP[i, :].reshape(5, np.prod(vp[:2])),
                vp[0], vp[1]) for i in range(xP.shape[0])]
t.dump_ims(orig, labels, predictions)
fig = model2.plot_Dices(dice, t.classes)
fig.savefig('dice.png')








