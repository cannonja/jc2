import imp
import os
import sys
from PIL import Image
import numpy as np
import socket
import pandas
import matplotlib.pyplot as plt
import pdb

machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    dict_path = 'C:\\Users\\Jack2\\Desktop\\dict.png'
    stim_path = 'C:\\Users\\Jack2\\Desktop\\stim.png'
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


##############################Test return_sparse##########################################################################################

#Set parameters
lamb = 1.0
tau = 10.0
delta = 0.01
u_stop = .001
t_type = 'S'
num_images = 1

#Load MNIST dictionary and signal
image_file = 't10k-images.idx3-ubyte'  #'train-images.idx3-ubyte'
signal_data = mnist.load_images(image_file, num_images, 50)
Image.fromarray(signal_data[0]).save(stim_path)

'''
#Scale stimulus and dictionary before running Rozell
signal_data[0] = signal_data[0].astype(float)
signal_data[0] /= 255.
dict_data = mnist.load_images(image_file, 50, 1)
#dict_data = pandas.read_csv('trained_data.csv', header=None, names=None)
#dict_data = [ np.asarray(r, dtype=float) / 255. for r in dict_data ]
for i in range(len(dict_data)):
    dict_data[i] = dict_data[i].astype(float)
    dict_data[i] /= 255.


D = sp.build_dictionary(dict_data)
#D = dict_data.values




#Run Rozell and generate sparse code
network = lca.r_network(D)
#network.save_dictionary(5, 10, dict_path)
network.set_parameters(lamb, tau, delta, u_stop, t_type)
error_names = ['E(t)', 'Resid', 'Cost', 'Sparsity']
#lambdas = np.arange(0.1, 10.5, 1)
lambdas = [3.]

#pdb.set_trace()
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
    #error.set_index(lambdas)
    print(error)
    plt.subplot(211)
    plt.plot(lambdas, error['E(t)'], 'r')
    plt.ylabel('E(t)')
    plt.subplot(212)
    plt.plot(lambdas, error['Sparsity'], 'c')
    plt.ylabel('Sparsity')
    plt.suptitle('E(t) and Sparsity vs. Lambda', size=16)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    #Generate and show grid images
    grid = network.fill_grid(rows, im1)
    grid2 = network.fill_grid(rows, im2)
    grid *= 255.
    grid2 *= 255.
    im_grid = Image.fromarray(grid)
    im_grid2 = Image.fromarray(grid2)
    im_grid.show()
    im_grid2.show()

'''
