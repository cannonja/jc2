import os
import sys
from PIL import Image
import numpy as np
import socket
import pandas
import matplotlib.pyplot as plt
import time
import datetime


machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack2\\Desktop'
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
else:
    #PSU machines (linux lab)
    base1 = os.path.expanduser('~/dev/jc2')
    base2 = os.path.expanduser('~/Desktop')
    sys.path.append(os.path.join(base1, 'MNIST_Load'))
    sys.path.append(os.path.join(base1, 'Rozell'))
    os.chdir(os.path.join(base1, 'MNIST_Load'))
    file_path = base2

import mnist_load as mnist
import sparse_algo as sp
import r_network_class as lca


################### Set parameters ##############################################################
lamb = .1
tau = 10
delta = 0.001
u_stop = .01
t_type = 'S'
alpha = 0.1

################### Load MNIST image and label data #############################################
num_images = 20
start_pos = 0
image_file = 'train-images.idx3-ubyte'
label_file = 'train-labels.idx1-ubyte'
image_data = mnist.load_images(image_file, num_images, start_pos)
label_data = mnist.load_labels(label_file, num_images, start_pos)

#Initialize network dictionary and parameters
D = 
rozell = lca.r_network(D)
rozell.set_parameters(lamb, tau, delta, u_stop, t_type)


################### Run each training image through network #######################################
################### For each image, generate sparse code then update trained ######################

