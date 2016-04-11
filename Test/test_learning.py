import imp
import os
import sys
from PIL import Image
import numpy as np
import socket
import pandas
import matplotlib.pyplot as plt

machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
else:
    #PSU machines (linux lab)
    base = os.path.expanduser('~/dev/jc2')
    sys.path.append(os.path.join(base, 'MNIST_Load'))
    sys.path.append(os.path.join(base, 'Rozell'))
    os.chdir(os.path.join(base, 'MNIST_Load'))


import mnist_load as mnist
import sparse_algo as sp
import r_network_class as lca


################### Set parameters ##############################################################
lamb = .1
tau = 10
delta = 0.001
u_stop = .01
t_type = 'S'
alpha = 0.811

################### Load dictionary from first 50 MNIST images ##################################
################### Load training set from last 59950 MNIST images ##############################
num_images = 1000
image_file = 'train-images.idx3-ubyte'
dict_data = mnist.load_images(image_file, 50)
training_data = mnist.load_images(image_file, num_images, 49)
D = sp.build_dictionary(dict_data)

#Initialize network dictionary and parameters
network = lca.r_network(D)
network.set_parameters(lamb, tau, delta, u_stop, t_type)


################### Run each training image through network #######################################
################### For each image, generate sparse code then update trained ######################
print(network.trained[:,1])
for i in range(num_images):
    stimulus = training_data[i].flatten()
    network.set_stimulus(stimulus)
    network.generate_sparse()

print(network.trained[:,1])







'''
#Run Rozell and generate sparse code

#For each image, run Rozell then generate error table and image grid
for i in range(num_images):
    #Get number of rows for error table and image grid
    #Then set stimulus for Rozell
    rows = len(lambdas)
    signal = signal_data[i].flatten()
    network.set_stimulus(signal)
    
    #Run Rozell and get error and image grid data
    error, im1, im2 = network.reconstruct(lambdas)

    #Add error table characteristics
    #Plot both E(t) and Sparsity vs. lambdas
    #E(t) on top plot and Sparsity on bottom
    error.columns = error_names
    error.set_index(lambdas)
    print(error)
    plt.subplot(211)
    plt.plot(lambdas, error['E(t)'], 'r')
    plt.subplot(212)
    plt.plot(lambdas, error['Sparsity'], 'c')
    plt.show()
    
    #Generate and show grid images
    grid = network.fill_grid(rows, im1)
    grid2 = network.fill_grid(rows, im2)    
    im_grid = Image.fromarray(grid)
    im_grid2 = Image.fromarray(grid2)
    im_grid.show()
    im_grid2.show()
'''
