import imp
import os
import sys
from scipy import misc
import numpy as np
import socket
import pandas
import matplotlib.pyplot as plt
import random
import pdb
from natsort import natsort


machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Projects\\net')
    file_path = 'C:\\Users\\Jack2\\Desktop'
    train_path = file_path + '\\Data Science Projects\\Kaggle Projects\\Ultrasound  Nerve Segmentation\\data\\train'
    im_path = train_path + '.csv'
    mask_path = file_path + '\\Data Science Projects\\Kaggle Projects\\Ultrasound  Nerve Segmentation\\data\\train_mask.csv'    
    os.chdir(train_path)
    train_all = natsort.natsorted(os.listdir())
elif (machine == 'Tab'):
    #Little laptop
    file_path = 'C:\\Users\\Jack\\Desktop'
else:
    #PSU machines (linux lab)
    base1 = os.path.expanduser('~/dev/jc2')
    base2 = os.path.expanduser('~/Desktop')
    sys.path.append(os.path.join(base1, 'Projects/net'))
    file_path = base2 + '/kaggle/ultrasound'
    train_path = file_path + '/data/train'
    im_path = train_path + '.csv' 
    mask_path = file_path + '/data/train_mask.csv'
    os.chdir(train_path)
    train_all = natsort(os.listdir())

from feed_forward import ff_net

################################## Read training images and masks ####################################################

'''Image 19_8.tif appears to not have a mask and 19_9 seems to be missing'''
train_im = train_all[slice(0,len(train_all),2)]
train_mask = train_all[slice(1,len(train_all),2)]

im_dim = misc.imread(train_im[0]).shape
ims = np.zeros((np.prod(im_dim), len(train_im)))
for i in range(len(train_im)):
    ims[:, i] = misc.read(train_im[i]).flatten()
numpy.savetxt(im_path, ims, delimiter=',')


masks = np.zeros((np.prod(im_dim), len(train_mask)))
for i in range(len(im_data)):
    masks[:, i] = misc.read(train_mask[i]).flatten()
numpy.savetxt(mask_path, masks, delimiter=',')














































