import imp
import os
import sys
from scipy import ndimage
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
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack2\\Desktop'
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
    os.chdir(train_path)

from feed_forward import ff_net

################################## Read training images and masks ####################################################

'''Image 19_8.tif appears to not have a mask and 19_9 seems to be missing'''
train_all = natsort(os.listdir())
train_im = train_all[slice(0,len(train_all),2)]
train_mask = train_all[slice(1,len(train_all),2)]
'''
print (len(train_all), train_all[:10], train_all[-10:])
print (len(train_im), len(train_mask), len(train_im) + len(train_mask))
print (train_im[:10], train_mask[:10])
print (train_im[-10:], train_mask[-10:])
'''














































