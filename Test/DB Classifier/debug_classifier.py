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
lamb = 8
tau = 10
delta = 0.001
u_stop = 0.001
t_type = 'S'
num_images = 1


#Load MNIST dictionary and signal
image_file = 't10k-images.idx3-ubyte'  #'train-images.idx3-ubyte'
signal_data = mnist.load_images(image_file, num_images, 9000)
dict_data = mnist.load_images(image_file, 50, 1)
#dict_data = pandas.read_csv('trained_data.csv', header=None, names=None).values

signal_data[0] = signal_data[0].astype(float)
signal_data[0] = np.multiply(signal_data[0], 1/255.)
for i in range(len(dict_data)):
    dict_data[i] = dict_data[i].astype(float)
    dict_data[i] /= 255.
D = sp.build_dictionary(dict_data)
#D = dict_data

#Run Rozell and generate sparse code
network = lca.r_network(D)
network.set_parameters(lamb, tau, delta, u_stop, t_type)
lambdas = [lamb]

#pdb.set_trace()
#Set stimulus for Rozell and show original input image
og_image = signal_data[0]
signal = og_image.flatten()
network.set_stimulus(signal)

#Generate sparse code, show original image and reconstruction
code = network.generate_sparse()
print (np.count_nonzero(code) / 50)
recon_data = np.dot(network.dictionary, code).reshape((28,28))
recon_data *= 255.
og_image *= 255.

for i in (og_image, recon_data):
    im = Image.fromarray(i)
    im.show()




