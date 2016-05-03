import os
import sys
from PIL import Image
import numpy as np
import socket
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime


machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Classify')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack2\\Desktop'
    dict_path = file_path + '\\Git_Repos\\jc2\\Classify\\trained_data'
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
    sys.path.append(os.path.join(base1, 'Classify'))
    os.chdir(os.path.join(base1, 'MNIST_Load'))
    file_path = base2
    dict_path = base1 + '/Classify/trained_data'

import mnist_load as mnist
import sparse_algo as sp
import r_network_class as lca
import classify


################### Set parameters ##############################################################
lamb = .1
tau = 10
delta = 0.001
u_stop = .01
t_type = 'S'
alpha = 0.1

################### Load MNIST image and label data #############################################
num_images = 5
start_pos = 0
image_file = 'train-images.idx3-ubyte'
label_file = 'train-labels.idx1-ubyte'
image_data = mnist.load_images(image_file, num_images, start_pos)
label_data = mnist.load_labels(label_file, num_images, start_pos)

################### Initialize Rozell network and set parameters ################################
dict_data = pd.read_csv(dict_path, header = None)
rozell = lca.r_network(dict_data.values)
rozell.set_parameters(lamb, tau, delta, u_stop, t_type)

################### Generate matrix of weights for mapping sparse code ###########################
################### to output layer.  Then forward prop phase          ###########################
weights = np.random.rand(10, 51) #10 nodes in layer j+1 and 50 nodes in layer j
correct = 0
for i, j in zip(image_data, label_data):
    #Run Rozell
    rozell.set_stimulus(i.flatten())
    sparse_code = rozell.generate_sparse()
    sparse_code = np.insert(sparse_code, 0, 1, axis = 0) #Add bias term

    #Integrate weights and sparse code, then feed to sigmoid
    z = np.dot(weights, sparse_code)
    activation = classify.sigmoid(z)[:, np.newaxis]

    #Convert ouptut vector to binary max value set to one
    #All others set to zero
    prediction = np.where(activation == activation.max())[0][0]
    label = np.zeros((10,1))
    label[j] = 1
    '''
    if (prediction == j):
        correct += 1
    '''
    error = activation - label
    MSE = np.dot(np.transpose(error), error) / error.shape[0]
    RMSE = np.sqrt(MSE)


    
    
    












